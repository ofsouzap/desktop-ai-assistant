"""Interactive REPL entry point."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from .config import load_config
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


def main(argv: Sequence[str] | None = None) -> None:
    paths = application_paths()
    paths.ensure_directories()

    try:
        config = load_config(paths.config)
    except ValueError as error:
        print(f"Configuration error: {error}")
        return
    logger = configure_logging(paths.state / "logs", console=config.console_logs)
    del argv  # Reserved for future CLI flags without changing the entry point.

    registry = ToolRegistry()
    register_inventory_tools(
        registry, InventoryStore(paths.data / INVENTORY_FILENAME, logger)
    )
    register_sway_tools(registry, SwayAdapter(logger, runner=subprocess.run))

    try:
        model = OpenRouterModelBackend(model=config.model)
    except ValueError as error:
        print(f"Error: {error}")
        return
    assistant = AssistantOrchestrator(
        model, registry, logger, maximum_steps=config.maximum_steps
    )

    print("Desktop AI Assistant. Type 'quit' or 'exit' to leave.")

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

        if outcome.tool_calls:
            print(f"[tools: {', '.join(outcome.tool_calls)}]")

        if outcome.error is not None:
            print(f"[error] {outcome.error}")

        if outcome.response:
            print(outcome.response)


if __name__ == "__main__":
    main()
