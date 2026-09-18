from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from desktop_ai_assistant.openrouter import OpenRouterModelBackend, _OpenRouterRequest
from desktop_ai_assistant.registry import ArgumentSpec, ToolSchema
from desktop_ai_assistant.types import (
    FinalResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolCallResponse,
)


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[_OpenRouterRequest] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(cast(_OpenRouterRequest, kwargs))
        return self.response


class FakeChat:
    def __init__(self, response: object) -> None:
        self.completions = FakeCompletions(response)


class FakeClient:
    def __init__(self, response: object) -> None:
        self.chat = FakeChat(response)


def test_maps_final_response() -> None:
    client = FakeClient(FakeResponse([FakeChoice(FakeMessage("Done."))]))
    backend = OpenRouterModelBackend(lambda: client)

    assert backend.next_response([Message(MessageRole.USER, "hello")], []) == FinalResponse(
        "Done."
    )
    assert "tools" not in client.chat.completions.requests[0]


def test_rejects_empty_final_response() -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(FakeResponse([FakeChoice(FakeMessage(""))]))
    )

    with pytest.raises(ValueError, match="empty assistant response"):
        backend.next_response([Message(MessageRole.USER, "hello")], [])


def test_maps_native_tool_call() -> None:
    client = FakeClient(
        FakeResponse(
            [
                FakeChoice(
                    FakeMessage(
                        None,
                        [FakeCall("call-1", FakeFunction("inventory_append", '{"text":"tea"}'))],
                    )
                )
            ]
        )
    )
    backend = OpenRouterModelBackend(lambda: client)

    response = backend.next_response(
        [Message(MessageRole.USER, "remember tea")],
        [ToolSchema("inventory_append", "Append.", (ArgumentSpec("text", str, "Item."),))],
    )

    assert isinstance(response, ToolCallResponse)
    assert response.tool_call.id == "call-1"
    assert response.tool_call.arguments == {"text": "tea"}
    assert client.chat.completions.requests[0]["tool_choice"] == "auto"
    assert client.chat.completions.requests[0]["parallel_tool_calls"] is False
    tool = client.chat.completions.requests[0]["tools"][0]
    assert tool["function"]["parameters"]["properties"]["text"]["type"] == "string"


def test_rejects_tool_argument_type_mismatch() -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(
            FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None,
                            [FakeCall("call-1", FakeFunction("count", '{"value":true}'))],
                        )
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError, match="invalid type"):
        backend.next_response(
            [Message(MessageRole.USER, "count")],
            [ToolSchema("count", "Count.", (ArgumentSpec("value", int, "Value."),))],
        )


def test_normalizes_integral_tool_argument_number() -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(
            FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None,
                            [FakeCall("call-1", FakeFunction("count", '{"value":1.0}'))],
                        )
                    )
                ]
            )
        )
    )

    response = backend.next_response(
        [Message(MessageRole.USER, "count")],
        [ToolSchema("count", "Count.", (ArgumentSpec("value", int, "Value."),))],
    )

    assert isinstance(response, ToolCallResponse)
    assert response.tool_call.arguments == {"value": 1}


def test_rejects_unknown_tool_name() -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(
            FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None,
                            [FakeCall("call-1", FakeFunction("unknown", "{}"))],
                        )
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError, match="unknown tool"):
        backend.next_response(
            [Message(MessageRole.USER, "hello")],
            [ToolSchema("known", "Known tool.", ())],
        )


def test_preserves_tool_calls_in_follow_up_messages() -> None:
    response = OpenRouterModelBackend._messages(
        [
            Message(MessageRole.USER, "remember tea"),
            Message(
                MessageRole.ASSISTANT,
                "",
                tool_call=ToolCall("call-1", "inventory_append", {"text": "tea"}),
            ),
            Message(MessageRole.TOOL, "Inventory entry appended.", "call-1"),
        ]
    )

    assert response[1]["role"] == "user"
    assert response[2].get("tool_calls") == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "inventory_append", "arguments": '{"text": "tea"}'},
        }
    ]
    assert response[3].get("tool_call_id") == "call-1"


@pytest.mark.parametrize("arguments", ["not json", "[]", '{"bad":[]}'])
def test_rejects_invalid_tool_arguments(arguments: str) -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(
            FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None, [FakeCall("call-1", FakeFunction("tool", arguments))]
                        )
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError):
        backend.next_response(
            [Message(MessageRole.USER, "hello")],
            [ToolSchema("tool", "Test tool.", (ArgumentSpec("bad", str, "Value."),))],
        )


def test_rejects_multiple_tool_calls() -> None:
    backend = OpenRouterModelBackend(
        lambda: FakeClient(
            FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None,
                            [
                                FakeCall("call-1", FakeFunction("first", "{}")),
                                FakeCall("call-2", FakeFunction("second", "{}")),
                            ],
                        )
                    )
                ]
            )
        )
    )

    with pytest.raises(ValueError, match="multiple tool calls"):
        backend.next_response([Message(MessageRole.USER, "hello")], [])


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterModelBackend()


def test_uses_documented_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    backend = OpenRouterModelBackend(
        lambda: FakeClient(FakeResponse([FakeChoice(FakeMessage("Done."))]))
    )

    assert backend.identifier == "openrouter:google/gemma-4-31b-it:free"
