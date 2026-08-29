from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from desktop_ai_assistant.codex import CodexModelBackend
from desktop_ai_assistant.registry import ArgumentSpec, ToolSchema
from desktop_ai_assistant.types import FinalResponse, Message, MessageRole, ToolCallResponse


@dataclass
class FakeResult:
    final_response: str


@dataclass
class FakeThread:
    response: str
    prompts: list[str] = field(default_factory=list)

    def run(self, input: str, *, output_schema: object) -> FakeResult:
        self.prompts.append(input)
        return FakeResult(self.response)


@dataclass
class FakeClient:
    response: str
    thread_arguments: list[object] = field(default_factory=list)
    closed: bool = False

    def thread_start(self, **kwargs: object) -> FakeThread:
        self.thread_arguments.append(kwargs)
        return FakeThread(self.response)

    def close(self) -> None:
        self.closed = True


def test_maps_final_response_from_codex() -> None:
    client = FakeClient('{"kind":"final","content":"Done."}')
    backend = CodexModelBackend(lambda: client)

    assert backend.next_response([Message(MessageRole.USER, "hello")], []) == FinalResponse(
        "Done."
    )
    backend.close()
    assert client.closed


def test_maps_tool_call_and_exposes_tool_schema() -> None:
    client = FakeClient(
        '{"kind":"tool_call","name":"inventory_append","arguments":{"text":"tea"}}'
    )
    backend = CodexModelBackend(lambda: client)

    response = backend.next_response(
        [Message(MessageRole.USER, "remember tea")],
        [ToolSchema("inventory_append", "Append an item.", (ArgumentSpec("text", str, "Item."),))],
    )

    assert isinstance(response, ToolCallResponse)
    assert response.tool_call.name == "inventory_append"
    assert response.tool_call.arguments == {"text": "tea"}
    assert "inventory_append" in client.thread_arguments[0]["developer_instructions"]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"kind":"final"}',
        '{"kind":"tool_call","name":"inventory_read","arguments":{"bad":[]}}',
    ],
)
def test_rejects_invalid_codex_output(response: str) -> None:
    backend = CodexModelBackend(lambda: FakeClient(response))

    with pytest.raises(ValueError):
        backend.next_response([Message(MessageRole.USER, "hello")], [])
