"""Interactive REPL entry point."""

from __future__ import annotations

import os
import subprocess

from .openrouter import OpenRouterModelBackend
from .integrations import Integration
from .integrations.inventory import (
    INVENTORY_FILENAME,
    InventoryStore,
    InventoryIntegration,
)
from .logging import configure_logging
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry
from .integrations.sway import SwayAdapter, SwayIntegration


def main() -> None:
    paths = application_paths()
    paths.ensure_directories()

    console_logging = os.environ.get("DESKTOP_AI_ASSISTANT_CONSOLE_LOGS") == "1"
    logger = configure_logging(paths.state / "logs", console=console_logging)

    registry = ToolRegistry()
    integrations: list[Integration] = [
        InventoryIntegration(InventoryStore(paths.data / INVENTORY_FILENAME, logger)),
        SwayIntegration(SwayAdapter(logger, runner=subprocess.run)),
    ]
    for integration in integrations:
        integration.register(registry)
    integration_prompts = [
        integration.integration_prompt
        for integration in integrations
        if integration.integration_prompt
    ]

    try:
        model = OpenRouterModelBackend(
            extra_system_prompt="\n\n".join(integration_prompts)
        )
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
