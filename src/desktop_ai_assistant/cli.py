"""Interactive REPL entry point."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from .config import load_config
from .openrouter import OpenRouterModelBackend
from .integrations import Integration
from .integrations.inventory import (
    INVENTORY_FILENAME,
    InventoryStore,
    InventoryIntegration,
)
from .logging import configure_logging
from .logging import log_event
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry
from .integrations.sway import SwayAdapter, SwayIntegration


def main(argv: Sequence[str] | None = None) -> None:
    paths = application_paths()
    paths.ensure_directories()

    try:
        config = load_config(paths.config, environment=dict(os.environ))
    except ValueError as error:
        print(f"Configuration error: {error}")
        return
    logger = configure_logging(paths.state / "logs", console=config.console_logs)
    del argv  # Reserved for future CLI flags without changing the entry point.
    log_event(
        logger,
        "startup_complete",
        model=config.model,
        maximum_steps=config.maximum_steps,
    )

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
            model=config.model,
            extra_system_prompt="\n\n".join(integration_prompts),
        )
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
