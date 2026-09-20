"""Safe, assistant-owned text inventory persistence and tool registration."""

from __future__ import annotations

import logging
from pathlib import Path

from .logging import log_event
from .registry import ArgumentSpec, ToolExecutionError, ToolRegistry
from .types import ToolArguments

INVENTORY_FILENAME = "inventory.txt"

_READ_DESCRIPTION = "Read the complete inventory text."

_APPEND_DESCRIPTION = """Append one inventory entry.
The inventory is UTF-8 plain text with one item or note per line. Each line is
free-form, such as "coffee filters" or "Laptop charger: office desk". Supply
exactly one non-empty line; do not add a newline. You can append directly
without reading first."""

_OVERWRITE_DESCRIPTION = """Replace the complete inventory.
The inventory is UTF-8 plain text with one free-form item or note per line.
Preserve its established line-oriented format. For example:
"coffee filters\nLaptop charger: office desk\n". Each line is conventional,
but its text is free-form. Read the current inventory first, then overwrite only
to make a small change while preserving its contents."""


class InventoryStore:
    """A UTF-8 inventory at one application-controlled path."""

    def __init__(self, path: Path, logger: logging.Logger) -> None:
        self._path = path
        self._logger = logger

    def read(self) -> str:
        if not self._path.exists():
            return ""
        try:
            return self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ToolExecutionError("Unable to read inventory.") from error

    def append(self, text: str) -> None:
        self._validate_entry(text)
        previous = self.read()
        separator = "" if not previous or previous.endswith("\n") else "\n"
        updated = f"{previous}{separator}{text}\n"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as inventory:
                inventory.write(separator + text + "\n")
        except OSError as error:
            raise ToolExecutionError("Unable to append to inventory.") from error
        log_event(
            self._logger,
            "inventory_appended",
            path=str(self._path),
            previous_content=previous,
            new_content=updated,
        )

    def overwrite(self, text: str) -> None:
        # Future hardening: require the current content for a compare-and-swap overwrite.
        previous = self.read()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(text, encoding="utf-8")
        except OSError as error:
            raise ToolExecutionError("Unable to overwrite inventory.") from error
        log_event(
            self._logger,
            "inventory_overwritten",
            path=str(self._path),
            previous_content=previous,
            new_content=text,
        )

    @staticmethod
    def _validate_entry(text: str) -> None:
        if not text or "\n" in text or "\r" in text:
            raise ToolExecutionError("Inventory entries must be one non-empty line.")


def register_inventory_tools(registry: ToolRegistry, store: InventoryStore) -> None:
    """Register the complete model-facing inventory capability."""

    @registry.register("inventory_read", _READ_DESCRIPTION)
    def inventory_read(arguments: ToolArguments) -> str:
        return store.read()

    @registry.register(
        "inventory_append",
        _APPEND_DESCRIPTION,
        (ArgumentSpec("text", str, "One non-empty inventory line."),),
    )
    def inventory_append(arguments: ToolArguments) -> str:
        store.append(str(arguments["text"]))
        return "Inventory entry appended."

    @registry.register(
        "inventory_overwrite",
        _OVERWRITE_DESCRIPTION,
        (ArgumentSpec("text", str, "Complete replacement inventory text."),),
    )
    def inventory_overwrite(arguments: ToolArguments) -> str:
        store.overwrite(str(arguments["text"]))
        return "Inventory replaced."
