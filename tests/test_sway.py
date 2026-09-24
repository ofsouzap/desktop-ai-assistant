import json
import logging
import subprocess
from collections.abc import Callable

import pytest

from desktop_ai_assistant.integrations.sway import (
    SwayAdapter,
    SwayError,
    SwayIntegration,
    SwayWindow,
    SwayWorkspace,
)
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import ToolCall


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["swaymsg"], returncode, stdout, stderr)


def scripted_runner(
    responses: list[tuple[list[str], subprocess.CompletedProcess[str]]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        expected_command, response = responses.pop(0)
        assert command == expected_command
        return response

    return runner


def test_normalizes_tree_windows_and_workspaces() -> None:
    responses = [
        (
            ["swaymsg", "-t", "get_tree"],
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
        ),
        (
            ["swaymsg", "-t", "get_workspaces"],
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
        ),
    ]

    adapter = SwayAdapter(logging.getLogger("test"), runner=scripted_runner(responses))
    assert adapter.list_windows() == [
        SwayWindow(
            id=42,
            app_id="firefox",
            class_name=None,
            title="Firefox",
            workspace="2:web",
            focused=True,
        )
    ]
    assert adapter.list_workspaces() == [
        SwayWorkspace(
            num=2,
            name="2:web",
            focused=True,
            visible=True,
            urgent=False,
        )
    ]
    assert not responses


def test_constructs_safe_write_commands() -> None:
    responses = [
        (["swaymsg", "[con_id=42] focus"], completed()),
        (
            ["swaymsg", '[con_id=42] move container to workspace "desk \\"A\\""'],
            completed(),
        ),
        (["swaymsg", 'workspace "desk"'], completed()),
        (["swaymsg", "[con_id=42] fullscreen disable"], completed()),
    ]

    adapter = SwayAdapter(logging.getLogger("test"), runner=scripted_runner(responses))
    adapter.focus_window(42)
    adapter.move_window_to_workspace(42, 'desk "A"')
    adapter.focus_workspace("desk")
    adapter.set_fullscreen(42, False)
    assert not responses


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

    invalid_workspace = SwayAdapter(
        logging.getLogger("test"),
        runner=lambda *args, **kwargs: completed(
            '[{"num": 1, "name": "1", "focused": "false", '
            '"visible": true, "urgent": false}]'
        ),
    )
    with pytest.raises(SwayError, match="invalid workspace"):
        invalid_workspace.list_workspaces()


def test_registers_all_sway_tools_and_dispatches_validation() -> None:
    responses = [
        (["swaymsg", "-t", "get_workspaces"], completed("[]")),
    ]

    registry = ToolRegistry()
    SwayIntegration(
        SwayAdapter(logging.getLogger("test"), runner=scripted_runner(responses))
    ).register(registry)

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

    # window_id should be an int value, not a string
    result = registry.dispatch(ToolCall("1", "focus_window", {"window_id": "42"}))
    assert result.is_error

    workspaces = registry.dispatch(ToolCall("2", "list_workspaces", {}))
    assert not workspaces.is_error
    assert workspaces.content == "[]"

    assert not responses
