"""Bounded sequential conversation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .logging import log_event
from .model import ModelBackend
from .registry import ToolRegistry
from .types import FinalResponse, Message, MessageRole


@dataclass(slots=True)
class TurnOutcome:
    response: str
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None


class AssistantOrchestrator:
    def __init__(
        self,
        model: ModelBackend,
        tools: ToolRegistry,
        logger: logging.Logger,
        maximum_steps: int = 5,
    ) -> None:
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be at least one")
        self._model = model
        self._tools = tools
        self._logger = logger
        self._maximum_steps = maximum_steps
        self._messages: list[Message] = []

    def _log_event(self, event: str, **details: object) -> None:
        log_event(self._logger, event, **details)

    def handle(self, user_input: str) -> TurnOutcome:
        self._messages.append(Message(MessageRole.USER, user_input))
        self._log_event("user_input", content=user_input)
        calls: list[str] = []
        try:
            for _ in range(self._maximum_steps):
                response = self._model.next_response(self._messages, self._tools.schemas())
                self._log_event(
                    "model_response", provider=self._model.identifier, response=response
                )
                if isinstance(response, FinalResponse):
                    self._messages.append(Message(MessageRole.ASSISTANT, response.content))
                    self._log_event("assistant_response", content=response.content)
                    return TurnOutcome(response.content, calls)
                else:
                    call = response.tool_call
                    calls.append(call.name)
                    self._log_event("tool_requested", tool_call=call)
                    self._messages.append(
                        Message(MessageRole.ASSISTANT, "", tool_call=call)
                    )
                    result = self._tools.dispatch(call)
                    self._log_event("tool_result", tool_result=result)
                    self._messages.append(
                        Message(MessageRole.TOOL, result.content, tool_call_id=result.tool_call_id)
                    )
            message = f"Stopped after {self._maximum_steps} tool calls."
            self._log_event("step_limit_reached", limit=self._maximum_steps)
            return TurnOutcome(message, calls, error=message)
        except Exception as error:
            self._logger.exception(
                "orchestration_failure",
                extra={
                    "details": {
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
                },
            )
            return TurnOutcome(
                "The assistant encountered a recoverable error. Please try again.",
                calls,
                error="orchestration failure",
            )
