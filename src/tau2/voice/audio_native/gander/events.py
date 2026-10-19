"""Provider-neutral event models emitted by the local Gander WebSocket server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GanderEvent(BaseModel):
    type: str
    raw: dict[str, Any] = Field(default_factory=dict)


class GanderAudioEvent(GanderEvent):
    type: Literal["audio"] = "audio"
    audio: bytes
    item_id: str
    end_of_turn: bool = False


class GanderTextEvent(GanderEvent):
    type: Literal["text"] = "text"
    text: str
    item_id: str


class GanderToolCallEvent(GanderEvent):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class GanderPlaybackCancelEvent(GanderEvent):
    type: Literal["playback_cancel"] = "playback_cancel"
    generation_id: int | None = None


class GanderErrorEvent(GanderEvent):
    type: Literal["error"] = "error"
    message: str
