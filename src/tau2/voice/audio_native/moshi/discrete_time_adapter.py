import asyncio
import logging
from typing import List, Optional, Any, Tuple
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter, TickResult
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.tick_result import UtteranceTranscript  # 🎯 刚刚锁定的导入路径
from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
)
from .provider import MoshiRealtimeProvider
from .events import MoshiAudioEvent, MoshiTextEvent, MoshiToolCallEvent, BaseMoshiEvent

logger = logging.getLogger(__name__)

class MoshiDiscreteTimeAdapter(DiscreteTimeAdapter):
    """Moshi Realtime API adapter bridging Moshi to the tick-based simulation."""

    def __init__(self, *args, **kwargs):
        # 1. 初始化基类
        super().__init__(*args, **kwargs)
        # 2. 实例化物理 Provider 和后台异步线程循环
        self._provider = MoshiRealtimeProvider()
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._tick_count = 0
        # 用于保存和聚合在不同 Tick 内音频和文本转写的字典
        self._utterance_transcripts = {}

        # 🎯 3. 新增这行：根据当前采样率自动计算每次发送的分块字节数（如 160 字节）
        self._chunk_size = int(
            self.bytes_per_tick * DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS / self.tick_duration_ms
        )

    @property
    def provider(self) -> MoshiRealtimeProvider:
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Optional[Any] = None,
        modality: str = "text",
    ) -> None:
        """生命周期钩子：启动后台 asyncio 循环并建立连接 (同步方法)。"""
        if self._connected:
            return

        # 🎯 性能优化核心：在连接最开始，对音频重采样和压缩库进行一次“冷启动预热”
        # 强行迫使操作系统提前将 scipy/soundfile 的外部 C 库装载进内存中，
        # 从而彻底免除 Tick 1 的 CPU 冷启动延迟！
        try:
            logger.info("Warming up Moshi audio encoder to prevent Tick 1 latency...")
            # 制造一个极微小的 20ms 哑白噪音，进行一次模拟预编码
            dummy_pcm = b"\x00" * 320  # 160 samples * 2 bytes (20ms)
            self.provider.pre_encode_tick_audio(dummy_pcm)
            logger.info("Moshi audio encoder warmed up successfully.")
        except Exception as warm_err:
            logger.warning(f"Audio encoder warm-up skipped: {warm_err}")

        # 启动后台异步线程
        self._bg_loop.start()

        try:
            # 投递到后台异步线程，建立连接并更新 Session 会话
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools),
                timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
            )
            self._connected = True
            logger.info("DiscreteTimeMoshiAdapter connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect Moshi Adapter: {e}")
            self._bg_loop.stop()
            raise RuntimeError(f"Failed to connect Moshi: {e}") from e

    async def _async_connect(self, system_prompt: str, tools: List[Tool]) -> None:
        """异步辅助：物理连接与 Session 初始化。"""
        await self.provider.connect()
        await self.provider.configure_session(system_prompt, tools)

    def disconnect(self) -> None:
        """生命周期钩子：断开物理连接并停止后台线程 (同步方法)。"""
        if not self._connected:
            return

        if self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(
                    self._async_disconnect(),
                    timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._bg_loop.stop()
        self._connected = False
        self._tick_count = 0
        self.clear_buffers()
        logger.info("MoshiDiscreteTimeAdapter disconnected")

    async def _async_disconnect(self) -> None:
        """异步辅助：切断 WebSocket。"""
        await self.provider.disconnect()

    def run_tick(
        self, user_audio: bytes, tick_number: Optional[int] = None
    ) -> TickResult:
        """同步入口：评测框架的编排器在每个 Tick 对它的直接调用。"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Moshi. Call connect() first.")

        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1

        try:
            # 将基类底层的异步运行协程，投递到后台异步事件循环中运行
            return self._bg_loop.run_coroutine(
                self._async_run_tick(user_audio, tick_number),
                timeout=self.tick_duration_ms / 1000
                + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
            )
        except Exception as e:
            logger.error(f"Error in run_tick (tick={tick_number}): {e}")
            raise

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """🎯 官方抽象方法：在单个 Tick (时间片) 内，收发音频并处理事件。
        
        新增本地 VAD 与打断检测：在本地实时监测用户说话状态并干预打断。
        """
                
        # 🎯 性能优化：在一开局，先对整包 200ms 的用户音频进行一次性超高效率预编码！
        if user_audio:
            self.provider.pre_encode_tick_audio(user_audio)
        
        async def receive_events():
            # 计算当前 Tick 还剩多少可用执行时间
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        # 1. 检查当前 Tick 用户是否在说话 (是否不为纯静音)
        is_user_speaking = False
        if user_audio:
            # 拿到系统默认的静音字节 (例如电话线的 b'\x7f')
            silence_byte = getattr(self, "silence_byte", b"\x7f")
            # 如果不全是静音字节，代表用户在发出声音
            is_user_speaking = not all(b == silence_byte[0] for b in user_audio)

        # 2. 并发地执行“分块发送音频”与“接收事件”
        _, events = await asyncio.gather(
            self._send_audio_chunked(
                user_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        # 3. 依次解析并注入事件数据
        for event in events:
            self._process_event(result, event)

        # 4. 🎯 本地 VAD 核心魔法：如果检测到用户在发出声音
        if is_user_speaking:
            # A. 标记用户开始说话
            if "speech_started" not in result.vad_events:
                result.vad_events.append("speech_started")
                logger.debug("Local VAD: User speech started detected.")

            # B. 判定打断：如果用户在说话的同时，模型也正在说话 (或者缓冲区里有待播放的声音)
            has_agent_audio = bool(result.agent_audio_chunks or getattr(self, "_buffered_agent_audio", None))
            if has_agent_audio:
                if "interrupted" not in result.vad_events:
                    result.vad_events.append("interrupted")
                result.was_truncated = True
                logger.info("Local VAD: Interruption detected! Truncating agent speech.")

                # 物理打断：立刻清空当前正在输出的模型音频，让模型在模拟器里瞬间“闭嘴”
                if result.agent_audio_chunks:
                    result.agent_audio_chunks.clear()
                if hasattr(self, "_buffered_agent_audio") and self._buffered_agent_audio:
                    self._buffered_agent_audio.clear()

        # 5. 冲洗工具结果
        await self._flush_pending_tool_results()

    def _process_event(self, result: TickResult, event: Any) -> None:
        """将捕获到的自定义 Moshi 事件，转换为评测框架的标准格式。"""
        result.events.append(event)

        # 我们对 Moshi 吐出的三种音频、文本、工具事件进行个性化路由
        if isinstance(event, MoshiAudioEvent):
            item_id = "moshi_utterance"  # 虚构一个统一的 utterance_id
            
            # A. 塞入音频缓冲区，供模拟器播放
            result.agent_audio_chunks.append((event.audio, item_id))

            # B. 登记在 self._utterance_transcripts 中，用于统计说话时间
            if item_id not in self._utterance_transcripts:
                self._utterance_transcripts[item_id] = UtteranceTranscript(
                    item_id=item_id
                )
            self._utterance_transcripts[item_id].add_audio(len(event.audio))

        elif isinstance(event, MoshiTextEvent):
            item_id = "moshi_utterance"
            
            # C. 将大模型的独白/文本输出，累加到当前音频片对应的文本转写缓存中
            if event.text:
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_transcript(event.text)

        elif isinstance(event, MoshiToolCallEvent):
            # D. 🎯 拦截到了大模型吐出的特殊 Token，将其组装成框架标准的 ToolCall
            tool_call = ToolCall(
                id=event.call_id,
                name=event.name,
                arguments=event.arguments,
            )
            result.tool_calls.append(tool_call)
            logger.info(f"Moshi Realtime Tool Call dispatched: {event.name}({event.call_id})")

    async def _flush_pending_tool_results(self) -> None:
        """🎯 官方抽象方法：将运行完的工具反馈冲洗给 Moshi。"""
        for (
            call_id,
            result_str,
            request_response,
            _is_error,
        ) in self._pending_tool_results:
            logger.info(f"Flushing tool result back to Moshi: ID {call_id}")
            # 完美透传给我们的 Moshi provider 
            await self.provider.send_tool_result(call_id, result_str, request_response)
        
        # 冲洗完后，清空待处理的工具结果缓存列表
        self._pending_tool_results.clear()