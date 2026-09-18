"""Provider-independent data exchanged by the assistant components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias

Primitive: TypeAlias = str | int | bool
ToolArguments: TypeAlias = Mapping[str, Primitive]


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: ToolArguments


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_call: ToolCall | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class FinalResponse:
    content: str


@dataclass(frozen=True, slots=True)
class ToolCallResponse:
    tool_call: ToolCall


ModelResponse: TypeAlias = FinalResponse | ToolCallResponse
