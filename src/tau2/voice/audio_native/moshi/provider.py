import asyncio
import json
import logging
import urllib.parse
from typing import Optional, List, Dict
import websockets
import re  # 🎯 新增导入：用于特殊 Token 的正则匹配
import io
import scipy.signal  # 用于将 16kHz 重采样到 24kHz
import soundfile as sf  # 用于在内存中将原始 PCM 压制为标准的 Ogg/Opus 字节流
from .events import MoshiAudioEvent, MoshiTextEvent, MoshiToolCallEvent, BaseMoshiEvent

logger = logging.getLogger(__name__)

class MoshiRealtimeProvider:
    """Moshi Realtime local provider with WebSocket-based communication."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8998,
    ):
        self.ws_url = f"ws://{host}:{port}/api/chat"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        # 用于在本地存下框架传给我们的工具和守则，方便后续分析
        self.tools: List = []
        self.system_prompt: str = ""
        # 🎯 新增这行：用于缓存预编码好的 Opus 音频分片
        self._audio_chunks_queue: List[bytes] = []

    @property
    def is_connected(self) -> bool:
        """检查当前 WebSocket 管道是否连接正常 (自适应新旧版本 websockets)。"""
        if self.ws is None:
            return False
            
        # 1. 兼容较新版本的 websockets (v13.0/v14.0+)
        if hasattr(self.ws, "state"):
            from websockets.protocol import State
            return self.ws.state is State.OPEN
            
        # 2. 兼容较老版本的 websockets (v12.0 及以下)
        return getattr(self.ws, "open", False)

    async def connect(self) -> None:
        """建立基本连接。初始化时，使用默认提示词。"""
        if self.is_connected:
            return
        # 初始默认连接
        await self._connect_with_prompt("You are a helpful assistant.")

    async def _connect_with_prompt(self, system_prompt: str) -> None:
        """根据传入的系统提示词建立连接。
        
        注意：因为官方原生的 moshi.server 并不支持通过 URL 传参，
        为了防止握手失败，我们直连官方干净的 WebSocket 接口。
        你未来的微调版本如果需要传参，可以在这里恢复拼接。
        """
        # 直连官方标准 WebSocket 接口，不带任何 ?text_prompt= 后缀
        connection_url = self.ws_url
        
        logger.info(f"Connecting to local Moshi server: {connection_url}...")
        try:
            # 建立直连
            self.ws = await websockets.connect(connection_url)
            logger.info("Successfully connected to Moshi server.")
        except Exception as e:
            logger.error(f"Failed to connect to Moshi server: {e}")
            raise e

    async def disconnect(self) -> None:
        """安全地切断与本地 Moshi 的物理连接。"""
        if self.ws:
            logger.info("Disconnecting from Moshi server...")
            await self.ws.close()
            self.ws = None
            logger.info("Disconnected successfully.")

    # ------ 后面我们要实现的工具注册、发送音频、接收事件的方法占位 ------
    async def configure_session(
        self,
        system_prompt: str,
        tools: List,
        *args,
        **kwargs,
    ) -> None:
        """注册工具和系统提示词（核心：通过断开并重连来实现）。"""
        # 1. 将工具和守则在本地存一份备用
        self.tools = tools
        self.system_prompt = system_prompt

        logger.info("Configuring Moshi session by reconnecting with new system prompt...")
        
        # 2. 断开初始的默认物理连接
        await self.disconnect()
        
        # 3. 带上真实的 system_prompt 重新连上本地 Moshi
        await self._connect_with_prompt(system_prompt)
        logger.info("Moshi session configured successfully with true system prompt.")

    def pre_encode_tick_audio(self, user_audio: bytes) -> None:
        """🎯 性能优化核心：预先对整包 200ms 的用户音频进行一次性 Ogg/Opus 压缩，避免分包重复计算。"""
        if not user_audio:
            self._audio_chunks_queue = []
            return

        # 一次性将整包 200ms (1600 字节) 的原始 PCM 编码为 Ogg/Opus
        full_ogg_opus = self._pcm_to_opus(user_audio)
        
        # 均匀切成 10 份（对应框架分 10 次调用 send_audio 发送）
        num_chunks = 10
        chunk_len = len(full_ogg_opus) // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_len
            # 最后一个分包带上余下的所有字节
            end_idx = (i + 1) * chunk_len if i < num_chunks - 1 else len(full_ogg_opus)
            chunks.append(full_ogg_opus[start_idx:end_idx])
            
        self._audio_chunks_queue = chunks

    async def send_audio(self, audio_data: bytes) -> None:
        """将测试框架生成的原始用户音频发送给本地 Moshi。"""
        if not self.is_connected:
            logger.warning("WebSocket is not connected. Skipping send_audio.")
            return

        try:
            # 🎯 优先从预先编码好的队列中直接弹出一个分片，时间复杂度 O(1)，无 CPU 计算！
            if self._audio_chunks_queue:
                opus_frame = self._audio_chunks_queue.pop(0)
            else:
                # 备用退路：如果队列空了，现场临时编码
                opus_frame = self._pcm_to_opus(audio_data)
                
            if opus_frame:
                payload = b"\x01" + opus_frame
                await self.ws.send(payload)
        except Exception as e:
            logger.error(f"Failed to send audio to Moshi: {e}")

    def _pcm_to_opus(self, pcm_data: bytes) -> bytes:
        """将 16kHz Mono 16-bit PCM 转换为符合 Moshi 要求的 24kHz Ogg/Opus 字节流。"""
        import numpy as np

        if not pcm_data:
            return b""

        # 1. 将原始 bytes 数据转换为 numpy 的 float32 数组 (以便重采样)
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. 核心重采样：16kHz 重采样到 24kHz
        # (因为 24000 / 16000 = 1.5 倍)
        num_samples = int(len(audio_array) * 1.5)
        resampled_audio = scipy.signal.resample(audio_array, num_samples)

        # 3. 核心压缩：使用 soundfile 将 24kHz 的 float32 音频数据压制进标准的 Ogg 容器中
        buffer = io.BytesIO()
        # 'OGG' 是容器格式，'OPUS' 是内部编解码器
        with sf.SoundFile(buffer, mode='w', format='OGG', subtype='OPUS', samplerate=24000, channels=1) as file:
            file.write(resampled_audio)
        
        # 4. 拿到标准的、以 "OggS" 开头的 Ogg/Opus 二进制数据
        ogg_opus_bytes = buffer.getvalue()
        
        return ogg_opus_bytes

    async def receive_events_for_duration(self, duration_seconds: float) -> List[BaseMoshiEvent]:
        """接收本地 Moshi 在特定 Tick 持续时间内返回的所有事件。"""
        if not self.is_connected:
            return []

        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        # 定义拦截你未来微调后 Moshi 工具调用的正则表达式
        # 匹配格式如：[TOOL_CALL: get_reservation_details, call_id: call_123, args: {"reservation_id": "EHGLP3"}]
        tool_pattern = re.compile(
            r"\[TOOL_CALL:\s*(\w+),\s*call_id:\s*(\w+),\s*args:\s*(\{.*\})\]"
        )

        while True:
            # 1. 计算本 Tick 剩余的可用监听时间
            remaining_time = end_time - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                break

            try:
                # 2. 异步等待接收 WebSocket 数据包
                db_message = await asyncio.wait_for(
                    self.ws.recv(), 
                    timeout=remaining_time
                )
                
                # 确保数据是二进制格式
                if not isinstance(db_message, bytes) or len(db_message) == 0:
                    continue

                # 3. 解析二进制数据的前缀 Tag
                tag = db_message[0]
                payload = db_message[1:]

                if tag == 1:
                    # ---- 【分支 1：音频数据 (Tag 0x01)】 ----
                    # 这里我们需要将 Moshi 的 24kHz Opus 音频解码，
                    # 并重采样回 16kHz PCM，以符合评测框架的要求。
                    # (此处你可以对接你本地的 Opus 解码模块 _opus_to_pcm)
                    pcm_audio = self._opus_to_pcm(payload)
                    events.append(MoshiAudioEvent(audio=pcm_audio))

                elif tag == 2:
                    # ---- 【分支 2：大脑独白文字数据 (Tag 0x02)】 ----
                    text_content = payload.decode("utf-8", errors="ignore")
                    logger.debug(f"Moshi Monologue Stream: {text_content}")

                    # 🎯 核心魔法：用正则表达式匹配你未来微调的 Special Token
                    match = tool_pattern.search(text_content)
                    if match:
                        tool_name = match.group(1)
                        call_id = match.group(2)
                        args_json_str = match.group(3)

                        try:
                            args_dict = json.loads(args_json_str)
                            # 虚构出一个评测框架支持的工具调用事件！
                            events.append(
                                MoshiToolCallEvent(
                                    call_id=call_id,
                                    name=tool_name,
                                    arguments=args_dict
                                )
                            )
                            logger.info(f"🎯 Intercepted Moshi Tool Call: {tool_name}({args_dict}) with ID {call_id}")
                        except Exception as parse_err:
                            logger.error(f"Failed to parse tool call arguments JSON: {parse_err}")
                    else:
                        # 只是普通说话文字，正常扔给框架
                        events.append(MoshiTextEvent(text=text_content))

            except asyncio.TimeoutError:
                # 正常的超时，说明在这个 Tick (0.2秒) 期间没有更多数据了，优雅退出循环
                break
            except Exception as e:
                logger.error(f"Error receiving from Moshi WebSocket: {e}")
                break

        return events

    def _opus_to_pcm(self, opus_data: bytes) -> bytes:
        """(辅助函数) 将 Moshi 发回的 24kHz Opus 音频解码并重采样回 16kHz PCM。"""
        # --- 调试阶段占位：如果你还在配置解码，可以先直接返回 opus_data ---
        # 实际运行评测前，请使用 pyogg / opuslib 对数据进行解码，以保证评测能正确读懂声音。
        return opus_data

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """(待实现) 将环境执行后的工具结果反馈给 Moshi。"""
        pass