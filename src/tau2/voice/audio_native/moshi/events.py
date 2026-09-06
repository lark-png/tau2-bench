from pydantic import BaseModel
from typing import Literal, Dict, Any, Optional

class BaseMoshiEvent(BaseModel):
    """Moshi 事件基类"""
    type: str

class MoshiAudioEvent(BaseMoshiEvent):
    """Moshi 返回的音频事件"""
    type: Literal["audio"] = "audio"
    audio: bytes  # 这里的 audio 将会是解码为 16kHz PCM 的音频数据

class MoshiTextEvent(BaseMoshiEvent):
    """Moshi 返回的大脑独白或文字事件"""
    type: Literal["text"] = "text"
    text: str

class MoshiToolCallEvent(BaseMoshiEvent):
    """(🎯 重中之重) 我们拦截并虚构出来的工具调用事件"""
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: Dict[str, Any]