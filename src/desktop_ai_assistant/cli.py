"""Interactive REPL entry point."""

from __future__ import annotations

from .logging import configure_logging
from .model import ScriptedModelBackend
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry


def main() -> None:
    paths = application_paths()
    paths.ensure_directories()
    assistant = AssistantOrchestrator(
        ScriptedModelBackend(()), ToolRegistry(), configure_logging(paths.state / "logs")
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

