"""Concrete inventory, Sway, and qualitative evaluation scenarios."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from desktop_ai_assistant.integrations.inventory import (
    InventoryStore,
    InventoryIntegration,
)
from desktop_ai_assistant.integrations.sway import SwayAdapter, SwayIntegration
from desktop_ai_assistant.model import ModelBackend, ScriptedModelBackend
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import FinalResponse, ToolCall, ToolCallResponse

from .framework import EvaluationTrace, Scenario, ScenarioFixture, run_scenario


def _inventory_fixture(directory: Path) -> ScenarioFixture:
    registry = ToolRegistry()
    InventoryIntegration(
        InventoryStore(directory / "inventory.txt", logging.getLogger("evaluation"))
    ).register(registry)
    return ScenarioFixture(registry=registry)


def _inventory_scenario() -> Scenario:
    return Scenario(
        name="inventory_remember_and_confirm",
        prompt="Remember that tea is in the kitchen, then confirm it.",
        scripted_responses=(
            ToolCallResponse(
                ToolCall("append", "inventory_append", {"text": "tea: kitchen"})
            ),
            ToolCallResponse(ToolCall("read", "inventory_read", {})),
            FinalResponse("Tea is in the kitchen."),
        ),
        expected_tool_calls=["inventory_append", "inventory_read"],
        fixture_factory=_inventory_fixture,
        checks=lambda outcome: {
            "final_response_mentions_tea": "tea" in outcome.response.lower(),
            "tool_sequence_is_bounded": len(outcome.tool_calls) <= 5,
        },
    )


def _sway_fixture(directory: Path) -> ScenarioFixture:
    sway_state = _MockSway()
    registry = ToolRegistry()
    SwayIntegration(
        SwayAdapter(logging.getLogger("evaluation"), runner=sway_state.run)
    ).register(registry)
    return ScenarioFixture(
        registry=registry,
        fixture_checks=lambda outcome: {
            "mock_state_updated": (
                sway_state.workspace_by_window[22] == "2:web"
                and sway_state.focused_window == 22
            )
        },
    )


def _sway_scenario() -> Scenario:
    return Scenario(
        name="sway_move_and_focus",
        prompt="Find Firefox and VS Code, move VS Code to Firefox's workspace, and focus it.",
        scripted_responses=(
            ToolCallResponse(ToolCall("windows", "list_windows", {})),
            ToolCallResponse(
                ToolCall(
                    "move",
                    "move_window_to_workspace",
                    {"window_id": 22, "workspace": "2:web"},
                )
            ),
            ToolCallResponse(ToolCall("focus", "focus_window", {"window_id": 22})),
            FinalResponse("VS Code was moved to Firefox's workspace and focused."),
        ),
        expected_tool_calls=[
            "list_windows",
            "move_window_to_workspace",
            "focus_window",
        ],
        fixture_factory=_sway_fixture,
        checks=lambda outcome: {
            "final_response_confirms_action": "focused" in outcome.response.lower(),
            "tool_sequence_is_bounded": len(outcome.tool_calls) <= 5,
        },
    )


def _qualitative_scenario() -> Scenario:
    return Scenario(
        name="capability_boundary_explanation",
        prompt="What kinds of desktop actions can you safely help with?",
        scripted_responses=(
            FinalResponse("I can use the listed inventory and constrained Sway tools."),
        ),
        expected_tool_calls=None,
        fixture_factory=lambda directory: ScenarioFixture(registry=ToolRegistry()),
        checks=lambda outcome: {
            "final_response_is_nonempty": bool(outcome.response.strip())
        },
        qualitative_review=True,
        qualitative_review_hint=(
            "The response should accurately describe the constrained inventory and "
            "Sway capabilities, explain relevant limitations, and avoid claiming "
            "unavailable desktop actions."
        ),
    )


class _MockSway:
    def __init__(self) -> None:
        self.workspace_by_window = {11: "2:web", 22: "1:code"}
        self.focused_window = 11

    def run(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        match command[1:]:
            case ["-t", "get_tree"]:
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
                        {
                            "type": "workspace",
                            "name": workspace,
                            "nodes": [window],
                            "floating_nodes": [],
                        }
                        for workspace in {"2:web", "1:code"}
                        for window in windows
                        if isinstance(window["id"], int)
                        and self.workspace_by_window[window["id"]] == workspace
                    ],
                    "floating_nodes": [],
                }
                return subprocess.CompletedProcess(command, 0, json.dumps(tree), "")
            case ["-t", "get_workspaces"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        [
                            {
                                "num": int(name.split(":", 1)[0]),
                                "name": name,
                                "focused": name
                                == self.workspace_by_window[self.focused_window],
                                "visible": True,
                                "urgent": False,
                            }
                            for name in ("1:code", "2:web")
                        ]
                    ),
                    "",
                )
            case [statement] if statement.startswith(
                "[con_id=22] move container to workspace"
            ):
                self.workspace_by_window[22] = "2:web"
            case [statement] if statement.startswith("[con_id=22] focus"):
                self.focused_window = 22
            case _:
                return subprocess.CompletedProcess(
                    command, 1, "", "Unsupported mocked Sway command."
                )

        return subprocess.CompletedProcess(command, 0, "", "")


def _run_evaluations(
    model_factory: Callable[[Scenario], ModelBackend],
) -> list[EvaluationTrace]:
    traces: list[EvaluationTrace] = []
    with TemporaryDirectory() as directory:
        for scenario in (
            _inventory_scenario(),
            _sway_scenario(),
            _qualitative_scenario(),
        ):
            traces.append(
                run_scenario(
                    scenario,
                    model_factory(scenario),
                    scenario.fixture_factory(Path(directory)),
                )
            )
    return traces


def run_scripted_evaluations() -> list[EvaluationTrace]:
    """Run deterministic checks without credentials or a Sway session."""
    return _run_evaluations(
        lambda scenario: ScriptedModelBackend(scenario.scripted_responses)
    )


def run_model_evaluations(model: ModelBackend) -> list[EvaluationTrace]:
    """Run the fixed scenarios against a supplied model and mocked state."""
    return _run_evaluations(lambda _: model)
