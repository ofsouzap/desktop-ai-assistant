import json
import logging
import subprocess

import pytest

from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.sway import SwayAdapter, SwayError, register_sway_tools
from desktop_ai_assistant.types import ToolCall


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["swaymsg"], returncode, stdout, stderr)


def test_normalizes_tree_windows_and_workspaces() -> None:
    responses = [
        completed(
            json.dumps(
                {
                    "type": "root",
                    "nodes": [
                        {
                            "type": "workspace",
                            "name": "2:web",
                            "nodes": [
                                {
                                    "type": "con",
                                    "id": 42,
                                    "name": "Firefox",
                                    "app_id": "firefox",
                                    "focused": True,
                                    "nodes": [],
                                    "floating_nodes": [],
                                }
                            ],
                        }
                    ],
                    "floating_nodes": [],
                }
            )
        ),
        completed(
            json.dumps(
                [
                    {
                        "num": 2,
                        "name": "2:web",
                        "focused": True,
                        "visible": True,
                        "urgent": False,
                    }
                ]
            )
        ),
    ]
    commands: list[list[str]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return responses.pop(0)

    adapter = SwayAdapter(logging.getLogger("test"), runner=runner)
    assert adapter.list_windows()[0].workspace == "2:web"
    workspaces = adapter.list_workspaces()
    assert workspaces[0].name == "2:web"
    assert commands == [
        ["swaymsg", "-t", "get_tree"],
        ["swaymsg", "-t", "get_workspaces"],
    ]


def test_constructs_safe_write_commands() -> None:
    commands: list[list[str]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return completed()

    adapter = SwayAdapter(logging.getLogger("test"), runner=runner)
    adapter.focus_window(42)
    adapter.move_window_to_workspace(42, 'desk "A"')
    adapter.focus_workspace("desk")
    adapter.set_fullscreen(42, False)
    assert commands == [
        ["swaymsg", "[con_id=42] focus"],
        ["swaymsg", '[con_id=42] move container to workspace "desk \\"A\\""'],
        ["swaymsg", 'workspace "desk"'],
        ["swaymsg", "[con_id=42] fullscreen disable"],
    ]


@pytest.mark.parametrize("window_id", [0, -1, True])
def test_rejects_invalid_window_id(window_id: int) -> None:
    adapter = SwayAdapter(
        logging.getLogger("test"), runner=lambda *args, **kwargs: completed()
    )
    with pytest.raises(SwayError, match="positive integer"):
        adapter.focus_window(window_id)


def test_normalizes_subprocess_and_json_failures() -> None:
    failed = SwayAdapter(
        logging.getLogger("test"),
        runner=lambda *args, **kwargs: completed(returncode=1, stderr="not connected"),
    )
    with pytest.raises(SwayError, match="not connected"):
        failed.list_workspaces()

    invalid = SwayAdapter(
        logging.getLogger("test"),
        runner=lambda *args, **kwargs: completed("not json"),
    )
    with pytest.raises(SwayError, match="invalid JSON"):
        invalid.list_workspaces()


def test_registers_all_sway_tools_and_dispatches_validation() -> None:
    commands: list[list[str]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:] == ["-t", "get_workspaces"]:
            return completed("[]")
        return completed()

    registry = ToolRegistry()
    register_sway_tools(registry, SwayAdapter(logging.getLogger("test"), runner=runner))
    names = {schema.name for schema in registry.schemas()}
    assert names == {
        "list_windows",
        "list_workspaces",
        "get_focused_window",
        "focus_window",
        "move_window_to_workspace",
        "focus_workspace",
        "set_fullscreen",
    }
    result = registry.dispatch(ToolCall("1", "focus_window", {"window_id": "42"}))
    assert result.is_error
    assert not commands
