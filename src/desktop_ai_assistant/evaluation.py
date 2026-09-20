"""Small, repeatable behavioral evaluations for the assistant."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Sequence

from .integrations.inventory import InventoryStore, register_inventory_tools
from .integrations.sway import SwayAdapter, register_sway_tools
from .model import ModelBackend, ScriptedModelBackend
from .orchestrator import AssistantOrchestrator, TurnOutcome
from .registry import ToolRegistry
from .types import FinalResponse, Message, ToolCall, ToolCallResponse


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    scenario: str
    prompt: str
    model: str
    response: str
    tool_calls: list[str]
    error: str | None
    objective_checks: dict[str, bool]
    messages: list[Message]
    qualitative_review: bool = False

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.objective_checks.values())

    def as_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    prompt: str
    scripted_responses: Sequence[FinalResponse | ToolCallResponse]
    expected_tool_calls: list[str] | None
    checks: Callable[[TurnOutcome], dict[str, bool]]
    qualitative_review: bool = False


def _inventory_scenario() -> _Scenario:
    return _Scenario(
        "inventory_remember_and_confirm",
        "Remember that tea is in the kitchen, then confirm it.",
        (
            ToolCallResponse(ToolCall("append", "inventory_append", {"text": "tea: kitchen"})),
            ToolCallResponse(ToolCall("read", "inventory_read", {})),
            FinalResponse("Tea is in the kitchen."),
        ),
        ["inventory_append", "inventory_read"],
        lambda outcome: {
            "final_response_mentions_tea": "tea" in outcome.response.lower(),
            "tool_sequence_is_bounded": len(outcome.tool_calls) <= 5,
        },
    )


def _sway_scenario() -> _Scenario:
    return _Scenario(
        "sway_move_and_focus",
        "Find Firefox and VS Code, move VS Code to Firefox's workspace, and focus it.",
        (
            ToolCallResponse(ToolCall("windows", "list_windows", {})),
            ToolCallResponse(
                ToolCall(
                    "move",
                    "move_window_to_workspace",
                    {"window_id": 22, "workspace": "2:web"},
                )
            ),
            ToolCallResponse(ToolCall("workspace", "focus_workspace", {"workspace": "2:web"})),
            ToolCallResponse(ToolCall("focus", "focus_window", {"window_id": 22})),
            FinalResponse("VS Code was moved to Firefox's workspace and focused."),
        ),
        ["list_windows", "move_window_to_workspace", "focus_workspace", "focus_window"],
        lambda outcome: {
            "final_response_confirms_action": "focused" in outcome.response.lower(),
            "tool_sequence_is_bounded": len(outcome.tool_calls) <= 5,
        },
    )


def _qualitative_scenario() -> _Scenario:
    return _Scenario(
        "capability_boundary_explanation",
        "What kinds of desktop actions can you safely help with?",
        (FinalResponse("I can use the listed inventory and constrained Sway tools."),),
        None,
        lambda outcome: {"final_response_is_nonempty": bool(outcome.response.strip())},
        qualitative_review=True,
    )


class _MockSway:
    def __init__(self) -> None:
        self.workspace_by_window = {11: "2:web", 22: "1:code"}
        self.focused_window = 11

    def run(
        self, command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        if command[1:] == ["-t", "get_tree"]:
            windows: list[dict[str, object]] = [
                {
                    "type": "con",
                    "id": window_id,
                    "name": "Firefox" if window_id == 11 else "VS Code",
                    "app_id": "firefox" if window_id == 11 else "code",
                    "focused": window_id == self.focused_window,
                    "nodes": [],
                    "floating_nodes": [],
                }
                for window_id in self.workspace_by_window
            ]
            tree = {
                "type": "root",
                "nodes": [
                    {"type": "workspace", "name": workspace, "nodes": [window], "floating_nodes": []}
                    for workspace in {"2:web", "1:code"}
                    for window in windows
                    if isinstance(window["id"], int)
                    and self.workspace_by_window[window["id"]] == workspace
                ],
                "floating_nodes": [],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(tree), "")
        if command[1:] == ["-t", "get_workspaces"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "num": int(name.split(":", 1)[0]),
                            "name": name,
                            "focused": name == self.workspace_by_window[self.focused_window],
                            "visible": True,
                            "urgent": False,
                        }
                        for name in ("1:code", "2:web")
                    ]
                ),
                "",
            )
        statement = command[1]
        if statement.startswith("[con_id=22] move container to workspace"):
            self.workspace_by_window[22] = "2:web"
        elif statement.startswith("[con_id=22] focus"):
            self.focused_window = 22
        return subprocess.CompletedProcess(command, 0, "", "")


def _run_scenario(
    scenario: _Scenario,
    model: ModelBackend,
    registry: ToolRegistry,
) -> EvaluationTrace:
    outcome = AssistantOrchestrator(model, registry, logging.getLogger("evaluation")).handle(
        scenario.prompt
    )
    checks = scenario.checks(outcome)
    if scenario.expected_tool_calls is not None:
        checks["expected_tool_calls"] = outcome.tool_calls == scenario.expected_tool_calls
    return EvaluationTrace(
        scenario.name,
        scenario.prompt,
        model.identifier,
        outcome.response,
        outcome.tool_calls,
        outcome.error,
        checks,
        outcome.messages,
        scenario.qualitative_review,
    )


def run_scripted_evaluations() -> list[EvaluationTrace]:
    """Run deterministic checks without credentials or a Sway session."""
    traces: list[EvaluationTrace] = []
    with TemporaryDirectory() as directory:
        inventory_registry = ToolRegistry()
        register_inventory_tools(
            inventory_registry,
            InventoryStore(Path(directory) / "inventory.txt", logging.getLogger("evaluation")),
        )
        inventory = _inventory_scenario()
        traces.append(
            _run_scenario(
                inventory,
                ScriptedModelBackend(inventory.scripted_responses),
                inventory_registry,
            )
        )

    sway_state = _MockSway()
    sway_registry = ToolRegistry()
    register_sway_tools(
        sway_registry,
        SwayAdapter(logging.getLogger("evaluation"), runner=sway_state.run),
    )
    sway = _sway_scenario()
    traces.append(
        _run_scenario(sway, ScriptedModelBackend(sway.scripted_responses), sway_registry)
    )
    traces[-1].objective_checks["mock_state_updated"] = (
        sway_state.workspace_by_window[22] == "2:web" and sway_state.focused_window == 22
    )
    qualitative = _qualitative_scenario()
    traces.append(
        _run_scenario(
            qualitative,
            ScriptedModelBackend(qualitative.scripted_responses),
            sway_registry,
        )
    )
    return traces


def run_model_evaluations(model: ModelBackend) -> list[EvaluationTrace]:
    """Run the fixed scenarios against a supplied model and mocked state."""
    traces: list[EvaluationTrace] = []
    with TemporaryDirectory() as directory:
        registry = ToolRegistry()
        register_inventory_tools(
            registry,
            InventoryStore(Path(directory) / "inventory.txt", logging.getLogger("evaluation")),
        )
        traces.append(_run_scenario(_inventory_scenario(), model, registry))
    sway_registry = ToolRegistry()
    sway_state = _MockSway()
    register_sway_tools(
        sway_registry,
        SwayAdapter(logging.getLogger("evaluation"), runner=sway_state.run),
    )
    traces.append(_run_scenario(_sway_scenario(), model, sway_registry))
    traces.append(_run_scenario(_qualitative_scenario(), model, sway_registry))
    return traces


def _write_traces(traces: Sequence[EvaluationTrace], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(trace) for trace in traces], default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted", action="store_true", help="avoid credentials and network access")
    parser.add_argument("--output", type=Path, default=Path("evaluation-traces.json"))
    arguments = parser.parse_args(argv)
    if arguments.scripted:
        traces = run_scripted_evaluations()
    else:
        from .openrouter import OpenRouterModelBackend

        traces = run_model_evaluations(OpenRouterModelBackend())
    _write_traces(traces, arguments.output)
    for trace in traces:
        print(f"{trace.scenario}: {'PASS' if trace.passed else 'REVIEW'}")
    return 0 if all(trace.passed for trace in traces) else 1


if __name__ == "__main__":
    raise SystemExit(main())
