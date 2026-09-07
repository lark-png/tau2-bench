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
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter


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

        self._converter = StreamingTelephonyConverter(
            input_sample_rate=24000,
            output_sample_rate=24000
        )

        # 基于moshi期望的音频格式计算可知每次发送960字节
        self._chunk_size = int(
            24000 * 2 * (DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS / 1000)
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
        #print(f"\n[TICK HEARTBEAT] Tick {tick_number} started ----------------", flush=True)
        import json
        from tau2.voice.audio_native.moshi.provider import SHARED_USER_TRANSCRIPTS

        # 🎯 =================【核心魔法：每个 Tick 开局，监视信箱长度】=================
        # 如果全局共享信箱里出现了我们还没处理的新用户消息（说明开启了新一轮对话）
        #print(f"DEBUG: SHARED_USER_TRANSCRIPTS length = {len(SHARED_USER_TRANSCRIPTS)}, last_user_index = {self.provider._last_user_index}", flush=True)
        while len(SHARED_USER_TRANSCRIPTS) > self.provider._last_user_index + 1:
            # 1. 提交上一轮助理说的话（如果上一次助理说了话，自动打包归档为 assistant）
            self.provider._commit_current_turn()
            # 2. 读取并推进我们处理过的用户消息索引
            self.provider._last_user_index += 1
            new_user_text = SHARED_USER_TRANSCRIPTS[self.provider._last_user_index]
            
            # 3. 将这一轮新的用户台词，作为 user 写入最终的对话历史中（100% 对齐 SFT 格式）
            self.provider.conversation_history.append({"role": "user", "content": new_user_text})
            print(f"\n\n📂 [CURRENT HISTORY STATE]\n{json.dumps(self.provider.conversation_history, indent=2, ensure_ascii=False)}\n\n", flush=True)
        # ==============================================================================
        
        model_ready_audio = b""
        if user_audio:
            model_ready_audio = self._converter.convert_input(user_audio)

        if model_ready_audio:
            self.provider.pre_encode_tick_audio(model_ready_audio)
        
        async def receive_events():
            # 计算当前 Tick 还剩多少可用执行时间
            elapsed_so_far = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed_so_far)
            return await self.provider.receive_events_for_duration(remaining)

        # 1. 检查当前 Tick 用户是否在说话 (是否不为纯静音)
        is_user_speaking = False
        if user_audio:
            import audioop
            # 1. 模拟器灌过来的 user_audio 是 8kHz mu-law，我们无损转为 PCM16 以便精确计算音量能量
            user_pcm = audioop.ulaw2lin(user_audio, 2)
            
            # 2. 计算这一帧的 RMS（均方根振幅）能量。值范围为 0 ~ 32767
            user_rms = audioop.rms(user_pcm, 2)
            
            # 3. 设定一个合理的过滤阀值（通常 300 到 500 可以完美过滤掉电话背景沙沙声）
            # 只有大于 300 时，才判定用户真正开始说台词了
            is_user_speaking = user_rms > 300

            # logger.debug(f"Local VAD: User RMS = {user_rms}, is_user_speaking = {is_user_speaking}")

        # 2. 并发地执行“分块发送音频”与“接收事件”
        _, events = await asyncio.gather(
            self._send_audio_chunked(
                model_ready_audio, self.provider.send_audio, self._chunk_size
            ),
            receive_events(),
        )

        # 3. 依次解析并注入事件数据
        #print(f"DEBUG: Tick {tick_number} received {len(events)} events from Moshi.", flush=True)
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
                
                # 🎯 当发生打断（Interruption）时，立刻重置重采样状态机，防止历史缓存干扰下一轮对话
                self._converter.reset()

        # 5. 冲洗工具结果
        await self._flush_pending_tool_results()

    def _process_event(self, result: TickResult, event: Any) -> None:
        """将捕获到的自定义 Moshi 事件，转换为评测框架的标准格式。"""
        result.events.append(event)

        # 我们对 Moshi 吐出的三种音频、文本、工具事件进行个性化路由
        # if isinstance(event, MoshiAudioEvent):
        #     item_id = "moshi_utterance"  # 虚构一个统一的 utterance_id

        #     telephony_ready_audio = b""
        #     if event.audio:
        #         telephony_ready_audio = self._converter.convert_output(event.audio) 
            
        #     import audioop
        #     # 计算 16-bit PCM 的均方根音量 (0 代表绝对静音，32767 代表最大破音音量)
        #     rms_value = audioop.rms(event.audio, 2) if event.audio else 0  
        #     print(f"DEBUG: Audio Event - Volume RMS: {rms_value}", flush=True)
        #     print(f"DEBUG: MoshiAudioEvent - Raw 24kHz PCM size: {len(event.audio)} bytes, Converted Telephony size: {len(telephony_ready_audio)} bytes", flush=True)

        #     if telephony_ready_audio:
        #         # A. 塞入音频缓冲区，供模拟器播放
        #         result.agent_audio_chunks.append((telephony_ready_audio, item_id))

        #         # B. 登记在 self._utterance_transcripts 中，用于统计说话时间
        #         if item_id not in self._utterance_transcripts:
        #             self._utterance_transcripts[item_id] = UtteranceTranscript(
        #                 item_id=item_id
        #             )
        #         self._utterance_transcripts[item_id].add_audio(len(event.audio))

        if isinstance(event, MoshiAudioEvent):
            item_id = "moshi_utterance"  # 虚构一个统一的 utterance_id
            
            telephony_ready_audio = b""
            if event.audio:
                telephony_ready_audio = self._converter.convert_output(event.audio)

            # ==============================================================================
            # 🎯 第一步：计算音量能量 (RMS)
            # ==============================================================================
            import audioop
            # 测算 24kHz PCM16 的 RMS 音量 (范围 0 ~ 32767)
            rms_value = audioop.rms(event.audio, 2) if event.audio else 0

            # ==============================================================================
            # 🎯 第二步：物理阻断（核心改动）
            # 只有音量大于 300 (代表真正有人声在发出) 时，我们才塞给模拟器播放。
            # 如果是底噪 (RMS 几十或为 0)，我们直接放行，不往 result.agent_audio_chunks 里塞数据。
            # ==============================================================================
            if rms_value > 300:
                #print(f"DEBUG: Moshi is speaking (RMS={rms_value}). Sending audio to simulator.", flush=True)
                result.agent_audio_chunks.append((telephony_ready_audio, item_id))
                
                # 只有真正说话时，才记录在 transcripts 中用于说话时间统计
                if item_id not in self._utterance_transcripts:
                    self._utterance_transcripts[item_id] = UtteranceTranscript(
                        item_id=item_id
                    )
                self._utterance_transcripts[item_id].add_audio(len(telephony_ready_audio))
            else:
                # 当音量微弱时，我们保持 result.agent_audio_chunks 为空，给环境营造“绝对静音”
                logger.debug(f"DEBUG: Filtered silent/low-volume frame (RMS={rms_value}).")

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