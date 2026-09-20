import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_ai_assistant.integrations.tools.inventory import (
    InventoryStore,
    register_inventory_tools,
)
from desktop_ai_assistant.model import ScriptedModelBackend
from desktop_ai_assistant.orchestrator import AssistantOrchestrator
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import FinalResponse, ToolArguments, ToolCall, ToolCallResponse


def make_assistant(
    responses: list[FinalResponse | ToolCallResponse], maximum_steps: int = 5
) -> AssistantOrchestrator:
    registry = ToolRegistry()

    @registry.register("ping", "Return pong.")
    def ping(arguments: ToolArguments) -> str:
        assert arguments == {}
        return "pong"

    return AssistantOrchestrator(
        ScriptedModelBackend(responses), registry, logging.getLogger("test"), maximum_steps
    )


def test_runs_sequential_tool_calls_before_final_response() -> None:
    assistant = make_assistant(
        [
            ToolCallResponse(ToolCall("one", "ping", {})),
            ToolCallResponse(ToolCall("two", "ping", {})),
            FinalResponse("Done."),
        ]
    )
    outcome = assistant.handle("Do the task")
    assert outcome.response == "Done."
    assert outcome.tool_calls == ["ping", "ping"]


def test_stops_at_limit() -> None:
    assistant = make_assistant(
        [ToolCallResponse(ToolCall(str(index), "ping", {})) for index in range(3)],
        maximum_steps=2,
    )
    outcome = assistant.handle("loop")
    assert outcome.error is not None
    assert outcome.tool_calls == ["ping", "ping"]


def test_tool_error_is_returned_to_model() -> None:
    assistant = make_assistant(
        [
            ToolCallResponse(ToolCall("bad", "missing", {})),
            FinalResponse("Recovered."),
        ]
    )
    outcome = assistant.handle("try")
    assert outcome.response == "Recovered."


def test_scripted_model_exercises_inventory_flow() -> None:
    with TemporaryDirectory() as directory:
        registry = ToolRegistry()
        inventory_path = Path(directory) / "inventory.txt"
        register_inventory_tools(
            registry, InventoryStore(inventory_path, logging.getLogger("test"))
        )
        assistant = AssistantOrchestrator(
            ScriptedModelBackend(
                [
                    ToolCallResponse(ToolCall("append", "inventory_append", {"text": "tea"})),
                    ToolCallResponse(ToolCall("read", "inventory_read", {})),
                    FinalResponse("Tea is in the inventory."),
                ]
            ),
            registry,
            logging.getLogger("test"),
        )
        outcome = assistant.handle("Remember tea")
        assert outcome.response == "Tea is in the inventory."
        assert outcome.tool_calls == ["inventory_append", "inventory_read"]
        assert inventory_path.read_text(encoding="utf-8") == "tea\n"
