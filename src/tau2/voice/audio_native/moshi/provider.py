import asyncio
import json
import logging
import urllib.parse
from typing import Optional, List, Dict
import websockets
import re
import io
import scipy.signal
import soundfile as sf
from .events import MoshiAudioEvent, MoshiTextEvent, MoshiToolCallEvent, BaseMoshiEvent
from .backend_agent import BackendAgent
import sphn

logger = logging.getLogger(__name__)

SHARED_USER_TRANSCRIPTS: List[str] = []

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

        # 以下为维护对话历史新增的状态机变量
        self.conversation_history: List[Dict[str, str]] = []  # 存放最终干净的对话历史记录
        self._current_agent_turn_text: str = ""  # 当前轮次moshi的文本字符

        # 🎯 新增这行：记录我们已经处理并写进历史的用户消息数量（初始化为 -1 代表一个都没处理过）
        self._last_user_index: int = -1
        self.opus_reader = sphn.OpusStreamReader(24000)

        # 🎯 实例化新解耦的后端认知代理类
        self.backend_agent = BackendAgent()

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
        # 🎯 修复状态污染：每次会话建立或重试时，彻底清空上一轮残留的历史与信箱
        # self.conversation_history.clear()
        # self._current_agent_turn_text = ""
        
        # # 🎯 修复时序误杀：只保留信箱中最新的一条（即当前轮次刚刚进来的用户发言）
        # if len(SHARED_USER_TRANSCRIPTS) > 1:
        #     latest_msg = SHARED_USER_TRANSCRIPTS[-1]
        #     SHARED_USER_TRANSCRIPTS.clear()
        #     SHARED_USER_TRANSCRIPTS.append(latest_msg)
        # self._last_user_index = -1

        # 1. 将工具和守则在本地存一份备用
        self.tools = tools
        self.system_prompt = system_prompt

        logger.info("Configuring Moshi session by reconnecting with new system prompt...")
        
        # 2. 断开初始的默认物理连接
        await self.disconnect()
        
        # 3. 带上真实的 system_prompt 重新连上本地 Moshi
        await self._connect_with_prompt(system_prompt)
        logger.info("Moshi session configured successfully with true system prompt.")
    
    async def send_text_to_moshi(self, text: str) -> None:
        """像用户打字一样，通过 WebSocket 将文本数据送入 Moshi 服务端。
        
        该文本数据将被包装成 0x02 (Text) 类型包，直接灌入 Moshi 服务端的接收循环 (recv_loop) 中。
        """
        if self.is_connected and self.ws:
            try:
                # 🎯 使用 0x02（二进制 Tag）作为前缀，拼接文本的字节码
                payload = b"\x02" + text.encode("utf-8")
                await self.ws.send(payload)
                logger.info(f"Successfully sent text payload back to Moshi server.")
            except Exception as e:
                logger.error(f"Failed to send text payload to Moshi: {e}")
        else:
            logger.warning("Skipped sending text payload because WebSocket is not connected.")

    def pre_encode_tick_audio(self, user_audio: bytes) -> None:
        """🎯 性能优化核心：预先对整包 200ms 的用户音频进行一次性 Ogg/Opus 压缩，避免分包重复计算。"""
        if not user_audio:
            self._audio_chunks_queue = []
            return

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
        """接收 24kHz Mono 16-bit PCM, 直接转换为符合 Moshi 要求的 24kHz Ogg/Opus 字节流。"""
        import numpy as np

        if not pcm_data:
            return b""

        resampled_audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        buffer = io.BytesIO()
        with sf.SoundFile(buffer, mode='w', format='OGG', subtype='OPUS', samplerate=24000, channels=1) as file:
            file.write(resampled_audio)
        
        ogg_opus_bytes = buffer.getvalue()
        
        return ogg_opus_bytes

    def _commit_current_turn(self) -> None:
        """只负责将 Moshi 临时积攒的文本打包存入最终的对话历史中。"""
        content = self._current_agent_turn_text.strip()
        if content:
            clean_content = re.sub(r"<tool_call>.*?</tool_call>|<tool_call>", "", content)
            clean_content = re.sub(r"<tool_response>.*?</tool_response>|<tool_response>.*", "", clean_content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            if clean_content:
                self.conversation_history.append({"role": "assistant", "content": clean_content})
                logger.info(f"📝 [History Commit] assistant: {clean_content}")
                print(f"\n\n📂 [CURRENT HISTORY STATE]\n{json.dumps(self.conversation_history, indent=2, ensure_ascii=False)}\n\n", flush=True)
        # 清空重置
        self._current_agent_turn_text = ""

    async def receive_events_for_duration(self, duration_seconds: float) -> List[BaseMoshiEvent]:
            """接收本地 Moshi 返回的所有事件，并在每个 Tick 开局自动维护多轮交替对话历史。"""
            if not self.is_connected:
                return []

            while len(SHARED_USER_TRANSCRIPTS) > self._last_user_index + 1:
                # 1) 用户开始新一轮发言，意味着 Moshi 上一轮的发言已经结束，先将其打包存入历史
                self._commit_current_turn()

                # 2) 提取用户的新发言并追加到最终历史中
                self._last_user_index += 1
                new_user_text = SHARED_USER_TRANSCRIPTS[self._last_user_index].strip()
                if new_user_text:
                    self.conversation_history.append({"role": "user", "content": new_user_text})
                    logger.info(f"📝 [History Commit] user: {new_user_text}")

            events = []
            end_time = asyncio.get_event_loop().time() + duration_seconds

            while True:
                # 计算本 Tick 剩余的可用监听时间
                remaining_time = end_time - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break

                try:
                    # 异步等待接收 WebSocket 数据包
                    db_message = await asyncio.wait_for(
                        self.ws.recv(), 
                        timeout=remaining_time
                    )
                    
                    if not isinstance(db_message, bytes) or len(db_message) == 0:
                        continue

                    tag = db_message[0]
                    payload = db_message[1:]

                    if tag == 1:
                        # ---- 【分支 1：音频数据 (Tag 0x01)】 ----
                        pcm_audio = self._opus_to_pcm(payload)
                        events.append(MoshiAudioEvent(audio=pcm_audio))

                    elif tag == 2:
                        # ---- 【分支 2：大脑独白文字数据 (Tag 0x02)】 ----
                        text_content = payload.decode("utf-8", errors="ignore")
                        # print(f"DEBUG MOSHI TEXT: {repr(text_content)}", flush=True)
                        logger.debug(f"Moshi Monologue Stream: {text_content}")

                        # A. 持续在本地累加 Moshi 这一轮说的字符
                        self._current_agent_turn_text += text_content

                        # B. 🎯 拦截调用指令：一旦发现累加的文字里出现了你训练的 "<tool_call>" 标记
                        if "<tool_call>" in self._current_agent_turn_text:
                            print(f"\n🚨🚨🚨 [INTERCEPTED <tool_call>] Accum text: {repr(self._current_agent_turn_text)} 🚨🚨🚨\n", flush=True)
                            logger.info("🎯 Intercepted '<tool_call>' token from Moshi's stream!")

                            # 1) 提交助理当前历史（过滤掉 '<tool_call>' 并打印完整历史状态）
                            self._commit_current_turn()

                            # 2) 🚀 调用解耦后的后端模型进行链式工具调用和结果事实压缩
                            logger.info("🚀 [GPT-4o Loop] Triggering real-time tool loop...")
                            compact_fact_summary, executed_tool_calls = await self.backend_agent.run_tool_loop(
                                system_prompt=self.system_prompt,
                                conversation_history=self.conversation_history,
                                tools=self.tools
                            )
                            
                            # 3) 🚀 核心改进：将压缩好的事实通过 WebSocket（Tag 0x02）强行灌回给 custom_server 
                            logger.info(f"📤 Feeding compact fact summary back to Moshi: {compact_fact_summary}")
                            await self.send_text_to_moshi(compact_fact_summary)
                            
                            # 4) 🚀 核心改进：将实际在后台运行过、并成功修改了数据库状态的工具转换为真实的 MoshiToolCallEvent
                            # 这一步能让评测框架（tau-voice）正确记录评测轨迹（Trajectory Logs），保障评测合规性
                            if executed_tool_calls:
                                for tool_call in executed_tool_calls:
                                    events.append(
                                        MoshiToolCallEvent(
                                            call_id=tool_call["call_id"],
                                            name=tool_call["name"],
                                            arguments=tool_call["arguments"]
                                        )
                                    )
                                    logger.info(f"Dispatched MoshiToolCallEvent to evaluation framework for: {tool_call['name']}({tool_call['call_id']})")
                            else:
                                print("No tool calls executed by GPT-4o; skipping tool event dispatch.")
                        else:
                            # 普通说话文字（没有触发工具），正常扔给框架，保持控制台转写同步显示
                            events.append(MoshiTextEvent(text=text_content))

                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    logger.error(f"Error receiving from Moshi WebSocket: {e}")
                    break

            return events

    def _opus_to_pcm(self, opus_data: bytes) -> bytes:
        """将 Moshi 发回的 24kHz Ogg/Opus 音频解码回 24kHz Mono 16-bit PCM。"""
        import soundfile as sf
        import io
        import numpy as np

        if not opus_data:
            return b""

        try:
            # ==============================================================================
            # 🎯 适配 sphn >= 0.2 的极简 API：
            # append_bytes 现在会直接返回解码好的 float32 numpy 数组！
            # ==============================================================================
            pcm_float = self.opus_reader.append_bytes(opus_data)
            
            if pcm_float is None or len(pcm_float) == 0:
                return b""
                
            # 将 float32 数组（范围 [-1.0, 1.0]）无损映射回 16-bit 有符号整数
            # 🎯 引入 np.clip 是流式音频的标准安全写法，可以防止信号振幅溢出产生的爆音和数据越界异常
            pcm_int16 = np.clip(pcm_float * 32768.0, -32768, 32767).astype(np.int16)
            
            # 返回原始 PCM16 二进制字节流
            return pcm_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Error decoding Moshi Ogg/Opus back to PCM: {e}")
            return b""

    async def send_tool_result(self, *args, **kwargs) -> None:
        """框架冲洗工具结果的钩子。
        
        因为工具调用已经在 Tick 内由 GPT-4o 实时执行完毕并修改了状态，
        使用 *args, **kwargs 接收并安全 pass，兼容框架传入的任何参数组合。
        """
        logger.debug(f"send_tool_result bypassed because execution was completed in-tick.")
        pass
    
    def reset_state(self) -> None:
        """任务结束时彻底清空 Provider 的对话历史、临时缓存与全局共享信箱。"""
        global SHARED_USER_TRANSCRIPTS
        SHARED_USER_TRANSCRIPTS.clear()
        self.conversation_history.clear()
        self._current_agent_turn_text = ""
        self._last_user_index = -1
        self._audio_chunks_queue.clear()
        logger.info("🧹 [State Reset] Provider and shared mailbox fully cleared for next task.")