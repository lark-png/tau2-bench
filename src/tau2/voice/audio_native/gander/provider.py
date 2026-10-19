"""WebSocket client for Gander's official ``gander_runtime`` duplex server.

The stock server serves ``/ws/duplex``.  It accepts little-endian PCM16 audio
as binary WebSocket messages and precedes every audio response with a JSON
metadata message.  This class deliberately does not use Gander's browser UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Sequence

import websockets

from .events import (
    GanderAudioEvent,
    GanderErrorEvent,
    GanderEvent,
    GanderPlaybackCancelEvent,
    GanderTextEvent,
    GanderToolCallEvent,
)

logger = logging.getLogger(__name__)


class GanderRealtimeProvider:
    """One Gander WebSocket session.

    Configure ``GANDER_WS_URL`` on the tau host, for example:
    ``ws://127.0.0.1:8000/ws/duplex``.

    The official Gander service configures the system prompt and tool schemas
    at server startup.  We retain the tau prompt/tools for diagnostics, but
    cannot dynamically change them until the Gander server exposes that API.
    """

    def __init__(self, ws_url: str | None = None) -> None:
        self.ws_url = ws_url or os.environ.get(
            "GANDER_WS_URL", "ws://127.0.0.1:8000/ws/duplex"
        )
        self.ws: Any | None = None
        self.session_id: str | None = None
        self.tools: Sequence[Any] = ()
        self.system_prompt = ""
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.chunk_ms = 1000

    @property
    def is_connected(self) -> bool:
        return self.ws is not None and not getattr(self.ws, "closed", False)

    async def connect(self) -> None:
        if self.is_connected:
            return
        session_id = f"tau_{uuid.uuid4().hex}"
        separator = "&" if "?" in self.ws_url else "?"
        url = f"{self.ws_url}{separator}session_id={session_id}"
        self.ws = await websockets.connect(url, max_size=None, ping_interval=20)
        ready = await asyncio.wait_for(self.ws.recv(), timeout=30)
        if not isinstance(ready, str):
            raise RuntimeError("Gander did not send a JSON ready event")
        payload = json.loads(ready)
        if payload.get("type") != "ready":
            raise RuntimeError(f"Gander session failed to become ready: {payload}")
        self.session_id = payload.get("session_id", session_id)
        self.input_sample_rate = int(payload.get("input_sample_rate", 16000))
        self.output_sample_rate = int(payload.get("output_sample_rate", 24000))
        self.chunk_ms = int(payload.get("chunk_ms", 1000))
        if payload.get("audio_input", {}).get("encoding") not in (None, "pcm_s16le"):
            raise RuntimeError(f"Unsupported Gander input encoding: {payload['audio_input']}")
        logger.info(
            "Connected to Gander session=%s input=%sHz output=%sHz chunk=%sms",
            self.session_id,
            self.input_sample_rate,
            self.output_sample_rate,
            self.chunk_ms,
        )

    async def configure_session(self, system_prompt: str, tools: Sequence[Any]) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        logger.warning(
            "Stock Gander /ws/duplex does not accept per-session prompt/tools. "
            "Ensure serve.local.yaml starts Gander with tau's policy/tool schema."
        )

    async def disconnect(self) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        try:
            await self.ws.close()
        finally:
            self.ws = None
            self.session_id = None

    async def send_audio(self, pcm16: bytes) -> None:
        if not self.is_connected:
            raise RuntimeError("Gander session is not connected")
        if len(pcm16) % 2:
            raise ValueError("Gander input must be PCM16 (even byte length)")
        await self.ws.send(pcm16)

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True, is_error: bool = False
    ) -> None:
        # The official endpoint's tool.response control event takes content;
        # correlation is maintained by its active tool coordinator.
        content: dict[str, Any] = {"call_id": call_id, "result": result, "is_error": is_error}
        await self.ws.send(json.dumps({"type": "tool.response", "content": content}))

    async def receive_events_for_duration(self, duration_seconds: float) -> list[GanderEvent]:
        if not self.is_connected or duration_seconds <= 0:
            return []
        events: list[GanderEvent] = []
        deadline = asyncio.get_running_loop().time() + duration_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return events
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                return events
            if isinstance(message, bytes):
                logger.warning("Ignoring unexpected unannounced Gander binary frame")
                continue
            payload = json.loads(message)
            kind = payload.get("type", "")
            if payload.get("audio"):
                audio = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
                if not isinstance(audio, bytes):
                    raise RuntimeError("Gander audio metadata was not followed by binary PCM16")
                events.append(
                    GanderAudioEvent(
                        audio=audio,
                        item_id=self._item_id(payload),
                        end_of_turn=bool(payload.get("end_of_turn", False)),
                        raw=payload,
                    )
                )
            if kind in {"chunk", "text.delta"} and payload.get("text"):
                events.append(GanderTextEvent(text=str(payload["text"]), item_id=self._item_id(payload), raw=payload))
            elif kind == "tool.call":
                events.extend(self._tool_events(payload))
            elif kind == "playback.cancel":
                events.append(GanderPlaybackCancelEvent(generation_id=payload.get("generation_id"), raw=payload))
            elif kind == "error":
                events.append(GanderErrorEvent(message=str(payload.get("message", "unknown Gander error")), raw=payload))

    @staticmethod
    def _item_id(payload: dict[str, Any]) -> str:
        return f"gander:{payload.get('generation_id', 0)}:{payload.get('unit_id', 0)}"

    @staticmethod
    def _tool_events(payload: dict[str, Any]) -> list[GanderToolCallEvent]:
        output: list[GanderToolCallEvent] = []
        for index, call in enumerate(payload.get("tool_calls") or []):
            function = call.get("function", call)
            name = function.get("name", call.get("name"))
            arguments = function.get("arguments", call.get("arguments", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not name:
                logger.warning("Ignoring Gander tool call without a name: %s", call)
                continue
            output.append(
                GanderToolCallEvent(
                    call_id=str(call.get("id") or call.get("call_id") or f"gander_call_{uuid.uuid4().hex}_{index}"),
                    name=str(name),
                    arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                    raw=payload,
                )
            )
        return output
