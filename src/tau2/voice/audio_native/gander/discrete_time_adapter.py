"""τ-Voice adapter for a local Gander ``/ws/duplex`` service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT,
    DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
    DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS,
)
from tau2.data_model.audio import AudioFormat
from tau2.data_model.message import ToolCall
from tau2.environment.tool import Tool
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.audio_converter import StreamingTelephonyConverter
from tau2.voice.audio_native.tick_result import TickResult, UtteranceTranscript

from .events import (
    GanderAudioEvent,
    GanderErrorEvent,
    GanderPlaybackCancelEvent,
    GanderTextEvent,
    GanderToolCallEvent,
)
from .provider import GanderRealtimeProvider

logger = logging.getLogger(__name__)


class GanderDiscreteTimeAdapter(DiscreteTimeAdapter):
    """Adapt τ-Voice telephony ticks to Gander's PCM16 duplex WebSocket."""

    def __init__(self, *args: Any, ws_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._provider = GanderRealtimeProvider(ws_url=ws_url)
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._converter = StreamingTelephonyConverter(
            input_sample_rate=16000,
            output_sample_rate=24000,
        )
        self._chunk_size = int(16000 * 2 * DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS / 1000)
        self._user_text_fragments: list[str] = []
        self._pending_final_user_turn: str | None = None

    def add_user_text_fragment(self, text: str | None) -> None:
        """Collect known user transcript text for the current spoken turn."""
        text = (text or "").strip()
        if text and (
            not self._user_text_fragments
            or self._user_text_fragments[-1] != text
        ):
            self._user_text_fragments.append(text)

    def finalize_user_turn(self) -> None:
        """Queue the completed user turn for Gander's turn.final control event."""
        text = " ".join(self._user_text_fragments).strip()
        self._user_text_fragments.clear()

        if not text:
            #print("Gander turn.final skipped: accumulated text is empty")
            return

        self._pending_final_user_turn = text
        #print("Gander queued turn.final: %r", text)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._bg_loop.is_running

    def connect(self, system_prompt: str, tools: Sequence[Tool], vad_config: Any, modality: str = "audio") -> None:
        if self._connected:
            return
        self._bg_loop = BackgroundAsyncLoop()
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(
                self._async_connect(system_prompt, tools), timeout=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT
            )
            if self._provider.input_sample_rate != 16000 or self._provider.output_sample_rate != 24000:
                raise RuntimeError(
                    "This initial adapter expects Gander input=16kHz/output=24kHz; "
                    f"server reported {self._provider.input_sample_rate}/{self._provider.output_sample_rate}."
                )
            if self.tick_duration_ms != self._provider.chunk_ms:
                logger.warning(
                    "tau tick_duration_ms=%s differs from Gander chunk_ms=%s. Use 1000 ms for the first run.",
                    self.tick_duration_ms, self._provider.chunk_ms,
                )
            self._connected = True
        except Exception:
            self._bg_loop.stop()
            raise

    async def _async_connect(self, system_prompt: str, tools: Sequence[Tool]) -> None:
        await self._provider.connect(system_prompt, tools)

    def disconnect(self) -> None:
        if self._connected and self._bg_loop.is_running:
            try:
                self._bg_loop.run_coroutine(self._provider.disconnect(), timeout=DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT)
            finally:
                self._bg_loop.stop()
        self._connected = False
        self.clear_buffers()
        self._converter.reset()

    def run_tick(self, user_audio: bytes, tick_number: Optional[int] = None) -> TickResult:
        if not self.is_connected:
            raise RuntimeError("Gander adapter is not connected")
        number = self._tick_count if tick_number is None else tick_number
        self._tick_count = number + 1
        return self._bg_loop.run_coroutine(
            self._async_run_tick(user_audio, number),
            timeout=self.tick_duration_ms / 1000 + DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER,
        )

    async def _execute_tick(self, user_audio: bytes, tick_number: int, result: TickResult, tick_start: float) -> None:
        pcm16 = self._converter.convert_input(user_audio) if user_audio else b""
        if self._pending_final_user_turn:
            text = self._pending_final_user_turn
            self._pending_final_user_turn = None

            #print("Gander flushing turn.final: %r", text)
            await self._provider.send_turn_final(text)

        remaining = max(
            0.01,
            self.tick_duration_ms / 1000
            - (asyncio.get_running_loop().time() - tick_start),
        )
        _, events = await asyncio.gather(
            self._send_audio_chunked(pcm16, self._provider.send_audio, self._chunk_size),
            self._provider.receive_events_for_duration(remaining),
        )
        for event in events:
            self._process_event(result, event)

    async def _flush_pending_tool_results(self) -> None:
        for call_id, content, request_response, is_error in self._pending_tool_results:
            print(
                f"[GANDER TOOL] tool.result "
                f"id={call_id} error={is_error} content={content}",
                flush=True,
            )
            await self._provider.send_tool_result(
                call_id=call_id,
                result=content,
                request_response=request_response,
                is_error=is_error,
            )

        self._pending_tool_results.clear()

    def _process_event(self, result: TickResult, event: Any) -> None:
        result.events.append(event)
        if isinstance(event, GanderAudioEvent):
            audio = self._converter.convert_output(event.audio)
            if audio:
                result.agent_audio_chunks.append((audio, event.item_id))
                transcript = self._utterance_transcripts.setdefault(event.item_id, UtteranceTranscript(item_id=event.item_id))
                transcript.add_audio(len(audio))
        elif isinstance(event, GanderTextEvent):
            transcript = self._utterance_transcripts.setdefault(event.item_id, UtteranceTranscript(item_id=event.item_id))
            transcript.add_transcript(event.text)
        elif isinstance(event, GanderToolCallEvent):
            print(
                f"[GANDER TOOL] tool.call "
                f"id={event.call_id} name={event.name} arguments={event.arguments}",
                flush=True,
            )
            result.tool_calls.append(
                ToolCall(
                    id=event.call_id,
                    name=event.name,
                    arguments=event.arguments,
                )
            )
        elif isinstance(event, GanderPlaybackCancelEvent):
            result.was_truncated = True
            result.vad_events.append("interrupted")
            result.agent_audio_chunks.clear()
            self._buffered_agent_audio.clear()
        elif isinstance(event, GanderErrorEvent):
            logger.error("Gander server error: %s", event.message)
