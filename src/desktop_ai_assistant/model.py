"""Interfaces and deterministic model implementations."""

from __future__ import annotations

from collections import deque
from typing import Protocol, Sequence

from .registry import ToolSchema
from .types import FinalResponse, Message, ModelResponse


class ModelBackend(Protocol):
    """A provider adapter that can produce one event for the current conversation."""

    @property
    def identifier(self) -> str: ...

    def next_response(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> ModelResponse: ...


class ScriptedModelBackend:
    """A deterministic backend for tests and local architecture demonstrations."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses: deque[ModelResponse] = deque(responses)

    @property
    def identifier(self) -> str:
        return "scripted"

    def next_response(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema]
    ) -> ModelResponse:
        if not self._responses:
            return FinalResponse("No scripted response is available.")
        return self._responses.popleft()
