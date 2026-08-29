"""Interactive REPL entry point."""

from __future__ import annotations

from .inventory import INVENTORY_FILENAME, InventoryStore, register_inventory_tools
from .logging import configure_logging
from .model import ScriptedModelBackend
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry


def main() -> None:
    paths = application_paths()
    paths.ensure_directories()
    logger = configure_logging(paths.state / "logs")
    registry = ToolRegistry()
    register_inventory_tools(registry, InventoryStore(paths.data / INVENTORY_FILENAME, logger))
    assistant = AssistantOrchestrator(
        ScriptedModelBackend(()), registry, logger
    )
    print("Desktop AI Assistant (mock backend). Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"quit", "exit"}:
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
