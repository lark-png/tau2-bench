# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import inspect
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

# 🎯 绝对导入优化：保证在项目工程中作为独立脚本运行时依赖解析完全正常
from moshi.client_utils import log
from moshi.models import loaders, MimiModel, LMModel, LMGen
from moshi.run_inference import get_condition_tensors


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, model_type: str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, cfg_coef: float, device: str | torch.device, **kwargs):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        # 🎯 新增：预计算 1 帧标准的 Mimi 真实静音 Token
        # 尺寸为 [1, 1, self.frame_size]
        silent_pcm = torch.zeros((1, 1, self.frame_size), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # encode 返回 [1, 8, 1]，表示 1 batch, 8 codebooks, 1 frame 的 discrete codes
            self.silent_mimi_codes = self.mimi.encode(silent_pcm).to(device=self.device, dtype=torch.long) # type: ignore

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self._silent_frame_count = 0
        self._last_spoken_piece = "[START_OF_SESSION]"

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])

        torch.cuda.synchronize()

    async def decode_and_send(
        self,
        tokens: torch.Tensor,
        ws: web.WebSocketResponse,
        opus_writer: sphn.OpusStreamWriter
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
        main_pcm = self.mimi.decode(tokens[:, 1:])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(b"\x01" + opus_bytes)
        text_token = tokens[0, 0, 0].item()
        if text_token not in (0, 3):
            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")

            # 🎯 结算刚刚结束的那段静音（如果停顿超过 3 帧 / 240ms，打印高亮统计）
            if self._silent_frame_count >= 3:
                duration_s = self._silent_frame_count * 0.08
                duration_ms = self._silent_frame_count * 80
                print(
                    f"\n{'!'*20} [SILENCE GAP DETECTED] {'!'*20}\n"
                    f"📍 上一个词: '{self._last_spoken_piece}'\n"
                    f"⏱️ 静音持续: {self._silent_frame_count} 帧 ({duration_ms}ms / {duration_s:.2f} 秒)\n"
                    f"📍 接下来的词: '{_text}'\n"
                    f"{'!'*65}\n",
                    flush=True
                )
            
            # 重置计数器，并记录当前说话的词
            self._silent_frame_count = 0
            self._last_spoken_piece = _text

            msg = b"\x02" + bytes(_text, encoding="utf8")
            log("info", f"text token '{_text}'")
            await ws.send_bytes(msg)

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter
    ):
        all_pcm_data = None
        skip_frames = 1
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        be = time.time()
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        if skip_frames:
                            # The first input audio frame is ignored, as from the point of
                            # view of the model it is in the past. We still `mimi.encode` for simplicity,
                            # however as the first encoded frame has a specific structure (due to the left padding),
                            # we reset the streaming state of the encoder to reapply the padding on the next call.
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            await self.decode_and_send(tokens, ws, opus_writer)
                        log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")
                
                elif kind == 2:  # 🎯 神经网络底层文本注入通道
                    payload = message[1:].decode("utf-8")
                    log("info", f"Received tool response to inject: '{payload}'")

                    # 🎯 显微镜打印 1：打印即将注入的文本
                    print(f"\n{'='*25} [INJECTING TO MOSHI BRAIN] {'='*25}", flush=True)
                    print(f"Payload: {payload}", flush=True)
                    
                    be_inject = time.time()
                    # 1. 使用 SentencePiece 对输入的工具返回文本进行 Token 编码
                    tokens_list = self.text_tokenizer.encode(payload)
                    print(f"Total tokens to inject: {len(tokens_list)}", flush=True)
                    
                    # 2. 依次将 Token 作为输入，进行 Teacher-Forcing 强制改写
                    for t in tokens_list:
                        # 🎯 优化 1：使用真实的 Mimi 静音 Token 作为用户输入，而不是全零 Tensor 
                        # self.silent_mimi_codes 形状为 [1, 8, 1]，代表了真实的静音
                        tokens = self.lm_gen.step(self.silent_mimi_codes)
                        
                        # 推动生成器进行自回归计算后，开始强注核心
                        state = self.lm_gen._streaming_state
                        current_pos = state.offsets[0].item() % state.cache.shape[2]
                        
                        # 🎯 强注文本：改写刚才模型自回归写入的代码，将其改为工具 Token t
                        state.cache[0, 0, current_pos] = t
                        
                        # 🎯 优化 2：强注 Moshi 音频：为了 100% 匹配您的 SFT 数据，
                        # 我们把 Moshi 自身的音频通道 (Codebook 1 至 8) 也全部覆盖为 Mimi 真实的静音 Token！
                        # state.cache 维度 1 的 1:9 对应主说话人的 8 个音频 codebooks
                        state.cache[0, 1:9, current_pos] = self.silent_mimi_codes[0, :, 0]
                        
                        # 修改临时返回的 tokens 变量，确保模型下一帧能获取到正确的历史缓存对齐
                        if tokens is not None:
                            tokens[0, 0, 0] = t
                            # 🎯 文本注入期间丢弃模型产生的 Opus 音频，首字延迟降到零
                    
                    inject_duration_ms = 1000 * (time.time() - be_inject)
                    log("info", f"Successfully Teacher-Forced {len(tokens_list)} text tokens & Silence Audio into Moshi Cache in {inject_duration_ms:.1f}ms.")
                    self._silent_frame_count = 0
                    self._last_spoken_piece = "[AFTER_TOOL_INJECTION]"
                
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            log("info", "connection closed")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        log("info", "accepted connection")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await self.recv_loop(ws, opus_reader, opus_writer)
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                        help="Do not fuse LoRA layers intot Linear layers.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log("error", "Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    log("info", "moshi loaded")

    state = ServerState(checkpoint_info.model_type, mimi, text_tokenizer, lm, args.cfg_coef, args.device,
                        **checkpoint_info.lm_gen_config)
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host , port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()