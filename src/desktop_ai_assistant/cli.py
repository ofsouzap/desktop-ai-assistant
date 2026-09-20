"""Interactive REPL entry point."""

from __future__ import annotations

import os
import subprocess

from .openrouter import OpenRouterModelBackend
from .integrations.inventory import (
    INVENTORY_FILENAME,
    InventoryStore,
    register_inventory_tools,
)
from .logging import configure_logging
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry
from .integrations.sway import SwayAdapter, register_sway_tools


def main() -> None:
    paths = application_paths()
    paths.ensure_directories()

    console_logging = os.environ.get("DESKTOP_AI_ASSISTANT_CONSOLE_LOGS") == "1"
    logger = configure_logging(paths.state / "logs", console=console_logging)

    registry = ToolRegistry()
    register_inventory_tools(
        registry, InventoryStore(paths.data / INVENTORY_FILENAME, logger)
    )
    register_sway_tools(registry, SwayAdapter(logger, runner=subprocess.run))

    try:
        model = OpenRouterModelBackend()
    except ValueError as error:
        print(f"Error: {error}")
        return
    assistant = AssistantOrchestrator(model, registry, logger)

    print("Desktop AI Assistant (OpenRouter). Type 'quit' to exit.")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if text.lower().strip() in {"quit", "exit"}:
            return
        if not text:
            continue
        outcome = assistant.handle(text)

        for tool_name in outcome.tool_calls:
            print(f"[tool] {tool_name}")

        if outcome.error is not None:
            print(f"Error: {outcome.error}")

        print(outcome.response)


if __name__ == "__main__":
    main()
