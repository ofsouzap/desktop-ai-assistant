"""Interactive REPL entry point."""

from __future__ import annotations

from .openrouter import OpenRouterModelBackend
from .inventory import INVENTORY_FILENAME, InventoryStore, register_inventory_tools
from .logging import configure_logging
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry


def main() -> None:
    paths = application_paths()
    paths.ensure_directories()

    logger = configure_logging(paths.state / "logs")

    registry = ToolRegistry()
    register_inventory_tools(
        registry, InventoryStore(paths.data / INVENTORY_FILENAME, logger)
    )

    model = OpenRouterModelBackend()
    try:
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
            else:
                outcome = assistant.handle(text)

                for tool_name in outcome.tool_calls:
                    print(f"[tool] {tool_name}")

                if outcome.error is not None:
                    print(f"Error: {outcome.error}")

                print(outcome.response)
    finally:
        del model


if __name__ == "__main__":
    main()
