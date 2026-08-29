from desktop_ai_assistant.registry import ArgumentSpec, ToolRegistry
from desktop_ai_assistant.types import ToolArguments, ToolCall


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        "repeat", "Return supplied text.", (ArgumentSpec("text", str, "Text"),)
    )
    def repeat(arguments: ToolArguments) -> str:
        return str(arguments["text"])

    return registry


def test_exposes_registered_schema() -> None:
    schema = make_registry().schemas()[0]
    assert schema.name == "repeat"
    assert schema.arguments[0].name == "text"


def test_dispatches_valid_call() -> None:
    result = make_registry().dispatch(ToolCall("1", "repeat", {"text": "hello"}))
    assert result.content == "hello"
    assert not result.is_error


def test_rejects_invalid_arguments() -> None:
    result = make_registry().dispatch(ToolCall("1", "repeat", {"text": 1}))
    assert result.is_error
    assert result.content == "Argument text must be str."


def test_rejects_unregistered_tool() -> None:
    result = make_registry().dispatch(ToolCall("1", "shell", {}))
    assert result.is_error
