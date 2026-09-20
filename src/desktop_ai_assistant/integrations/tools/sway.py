"""Constrained Sway IPC access and model-facing desktop tools."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .logging import log_event
from .registry import ArgumentSpec, ToolExecutionError, ToolRegistry
from .types import ToolArguments


class SwayError(ToolExecutionError):
    """A normalized failure while querying or controlling Sway."""


@dataclass(frozen=True, slots=True)
class SwayWindow:
    id: int
    app_id: str | None
    class_name: str | None
    title: str | None
    workspace: str | None
    focused: bool


@dataclass(frozen=True, slots=True)
class SwayWorkspace:
    num: int
    name: str
    focused: bool
    visible: bool
    urgent: bool


Runner = Callable[..., subprocess.CompletedProcess[str]]


class SwayAdapter:
    """The only subprocess authority used by the Sway tools."""

    def __init__(
        self,
        logger: logging.Logger,
        runner: Runner,
        command: str = "swaymsg",
    ) -> None:
        self._logger = logger
        self._runner = runner
        self._command = command

    def list_windows(self) -> list[SwayWindow]:
        tree = self._query_json(["-t", "get_tree"])
        windows: list[SwayWindow] = []
        self._collect_windows(tree, None, windows)
        return windows

    def list_workspaces(self) -> list[SwayWorkspace]:
        result = self._query_json(["-t", "get_workspaces"])
        if not isinstance(result, list):
            raise SwayError("Sway returned an invalid workspace list.")
        try:
            return [
                SwayWorkspace(
                    num=int(item["num"]),
                    name=str(item["name"]),
                    focused=bool(item["focused"]),
                    visible=bool(item["visible"]),
                    urgent=bool(item["urgent"]),
                )
                for item in result
                if isinstance(item, Mapping)
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise SwayError("Sway returned an invalid workspace.") from error

    def get_focused_window(self) -> SwayWindow | None:
        return next((window for window in self.list_windows() if window.focused), None)

    def focus_window(self, window_id: int) -> None:
        self._run_command(f"[con_id={self._window_id(window_id)}] focus")

    def move_window_to_workspace(self, window_id: int, workspace: str) -> None:
        self._validate_workspace(workspace)
        self._run_command(
            f"[con_id={self._window_id(window_id)}] move container to workspace "
            f"{self._quote(workspace)}"
        )

    def focus_workspace(self, workspace: str) -> None:
        self._validate_workspace(workspace)
        self._run_command(f"workspace {self._quote(workspace)}")

    def set_fullscreen(self, window_id: int, enabled: bool) -> None:
        self._run_command(
            f"[con_id={self._window_id(window_id)}] fullscreen "
            f"{'enable' if enabled else 'disable'}"
        )

    def _query_json(self, arguments: Sequence[str]) -> Any:
        completed = self._run(
            ["-t", *arguments[1:]] if arguments[:1] == ("-t",) else arguments
        )
        if completed.returncode != 0:
            raise SwayError(self._failure_message(completed))
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise SwayError("Sway returned invalid JSON.") from error

    def _run_command(self, command: str) -> None:
        self._run([command])

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = [self._command, *arguments]
        log_event(self._logger, "sway_command_requested", command=command)

        try:
            completed = self._runner(
                command, capture_output=True, text=True, check=False, timeout=3
            )
            log_event(
                self._logger,
                "sway_command_completed",
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except FileNotFoundError as error:
            raise SwayError("The swaymsg executable was not found.") from error
        except subprocess.TimeoutExpired as error:
            raise SwayError("Sway did not respond before the timeout.") from error
        except OSError as error:
            raise SwayError("Unable to communicate with Sway.") from error

        return completed

    @staticmethod
    def _failure_message(completed: subprocess.CompletedProcess[str]) -> str:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return f"Sway command failed{': ' + detail if detail else '.'}"

    @staticmethod
    def _window_id(window_id: int) -> str:
        if type(window_id) is not int or window_id <= 0:
            raise SwayError("Window ID must be a positive integer.")
        return str(window_id)

    @staticmethod
    def _validate_workspace(workspace: str) -> None:
        if (
            not workspace
            or "\x00" in workspace
            or "\n" in workspace
            or "\r" in workspace
        ):
            raise SwayError("Workspace must be a non-empty single-line name.")

    @staticmethod
    def _quote(value: str) -> str:
        return json.dumps(value)

    @classmethod
    def _collect_windows(
        cls, node: object, workspace: str | None, windows: list[SwayWindow]
    ) -> None:
        if not isinstance(node, Mapping):
            return
        node_type = node.get("type")
        current_workspace = workspace
        if node_type == "workspace":
            name = node.get("name")
            current_workspace = str(name) if name is not None else None

        children = node.get("nodes", [])
        floating = node.get("floating_nodes", [])
        if isinstance(children, list):
            for child in children:
                cls._collect_windows(child, current_workspace, windows)
        if isinstance(floating, list):
            for child in floating:
                cls._collect_windows(child, current_workspace, windows)

        if node_type not in {"con", "floating_con"} or children or floating:
            return

        try:
            window_id = int(node["id"])
        except (KeyError, TypeError, ValueError):
            return

        properties = node.get("window_properties")
        window_properties = properties if isinstance(properties, Mapping) else {}
        app_id = node.get("app_id")
        class_name = window_properties.get("class")
        title = node.get("name")

        windows.append(
            SwayWindow(
                id=window_id,
                app_id=str(app_id) if app_id is not None else None,
                class_name=str(class_name) if class_name is not None else None,
                title=str(title) if title is not None else None,
                workspace=current_workspace,
                focused=bool(node.get("focused", False)),
            )
        )


def register_sway_tools(registry: ToolRegistry, adapter: SwayAdapter) -> None:
    """Register the fixed, structured Sway capability set."""

    @registry.register(
        "list_windows",
        "List open windows with IDs, applications, titles, workspaces, and focus.",
    )
    def list_windows(arguments: ToolArguments) -> str:
        return json.dumps([asdict(window) for window in adapter.list_windows()])

    @registry.register(
        "list_workspaces",
        "List Sway workspaces and their focus, visibility, and urgency.",
    )
    def list_workspaces(arguments: ToolArguments) -> str:
        return json.dumps(
            [asdict(workspace) for workspace in adapter.list_workspaces()]
        )

    @registry.register(
        "get_focused_window",
        "Return the currently focused window, or indicate that none is focused.",
    )
    def get_focused_window(arguments: ToolArguments) -> str:
        window = adapter.get_focused_window()
        return json.dumps(asdict(window) if window is not None else None)

    @registry.register(
        "focus_window",
        "Focus a window by its concrete Sway container ID from list_windows.",
        (ArgumentSpec("window_id", int, "Positive Sway container ID."),),
    )
    def focus_window(arguments: ToolArguments) -> str:
        adapter.focus_window(int(arguments["window_id"]))
        return "Window focused."

    @registry.register(
        "move_window_to_workspace",
        "Move a window by ID to a named workspace.",
        (
            ArgumentSpec("window_id", int, "Positive Sway container ID."),
            ArgumentSpec("workspace", str, "Workspace name."),
        ),
    )
    def move_window_to_workspace(arguments: ToolArguments) -> str:
        adapter.move_window_to_workspace(
            int(arguments["window_id"]), str(arguments["workspace"])
        )
        return "Window moved."

    @registry.register(
        "focus_workspace",
        "Focus a named workspace.",
        (ArgumentSpec("workspace", str, "Workspace name."),),
    )
    def focus_workspace(arguments: ToolArguments) -> str:
        adapter.focus_workspace(str(arguments["workspace"]))
        return "Workspace focused."

    @registry.register(
        "set_fullscreen",
        "Enable or disable fullscreen for a window by its concrete Sway container ID.",
        (
            ArgumentSpec("window_id", int, "Positive Sway container ID."),
            ArgumentSpec("enabled", bool, "Whether fullscreen should be enabled."),
        ),
    )
    def set_fullscreen(arguments: ToolArguments) -> str:
        adapter.set_fullscreen(int(arguments["window_id"]), bool(arguments["enabled"]))
        return "Fullscreen updated."
