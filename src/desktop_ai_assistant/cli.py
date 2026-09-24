"""Interactive REPL entry point."""

from __future__ import annotations

import logging
import os
import subprocess

import click

from .config import AssistantConfig, load_config
from .integrations import Integration
from .integrations.inventory import (
    INVENTORY_FILENAME,
    InventoryIntegration,
    InventoryStore,
)
from .integrations.sway import SwayAdapter, SwayIntegration
from .logging import configure_logging, log_event
from .openrouter import OpenRouterModelBackend
from .orchestrator import AssistantOrchestrator
from .paths import ApplicationPaths, application_paths
from .registry import ToolRegistry


def _load_assistant_config(paths: ApplicationPaths) -> AssistantConfig | None:
    try:
        return load_config(paths.config, environment=dict(os.environ))
    except (ValueError, TypeError) as error:
        print(f"Configuration error: {error}")
        return None


def _build_assistant(
    paths: ApplicationPaths, config: AssistantConfig, logger: logging.Logger
) -> AssistantOrchestrator | None:
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
        return None
    return AssistantOrchestrator(
        model, registry, logger, maximum_steps=config.maximum_steps
    )


def _run_repl(assistant: AssistantOrchestrator) -> int:
    print("Desktop AI Assistant. Type 'quit' or 'exit' to leave.")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if text.lower().strip() in {"quit", "exit"}:
            return 0
        elif not text:
            continue
        else:
            outcome = assistant.handle(text)
            if outcome.tool_calls:
                print(f"[tools: {', '.join(outcome.tool_calls)}]")
            if outcome.error is not None:
                print(f"[error] {outcome.error}")
            if outcome.response:
                print(outcome.response)


@click.command()
def main() -> None:
    """Run the constrained Linux/Sway desktop assistant REPL.

    The assistant uses OpenRouter for inference and various integrations. It does not provide shell,
    filesystem, or arbitrary Sway command execution.
    """
    paths = application_paths()
    paths.ensure_directories()

    config = _load_assistant_config(paths)
    if config is None:
        raise click.exceptions.Exit(1)
    logger = configure_logging(paths.state / "logs", console=config.console_logs)

    log_event(
        logger,
        "startup_complete",
        model=config.model,
        maximum_steps=config.maximum_steps,
    )

    assistant = _build_assistant(paths, config, logger)
    if assistant is None:
        raise click.exceptions.Exit(1)

    raise click.exceptions.Exit(_run_repl(assistant))


if __name__ == "__main__":
    main()
