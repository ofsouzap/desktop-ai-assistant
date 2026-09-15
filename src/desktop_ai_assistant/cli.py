"""Interactive REPL entry point."""

from __future__ import annotations

import argparse

from openai_codex import Codex

from .codex import CodexModelBackend
from .inventory import INVENTORY_FILENAME, InventoryStore, register_inventory_tools
from .logging import configure_logging
from .orchestrator import AssistantOrchestrator
from .paths import application_paths
from .registry import ToolRegistry


def command_login() -> None:
    with Codex() as codex:
        if codex.account().account is not None:
            print("A ChatGPT login is already available.")
            return
        login = codex.login_chatgpt_device_code()
        print(f"Open {login.verification_url} and enter code: {login.user_code}")
        result = login.wait()
    if not result.success:
        raise RuntimeError("ChatGPT login did not complete.")
    print("ChatGPT login completed.")


def command_chat() -> None:
    paths = application_paths()
    paths.ensure_directories()

    logger = configure_logging(paths.state / "logs")

    registry = ToolRegistry()
    register_inventory_tools(
        registry, InventoryStore(paths.data / INVENTORY_FILENAME, logger)
    )

    with CodexModelBackend() as model:
        assistant = AssistantOrchestrator(model, registry, logger)

        print("Desktop AI Assistant (OpenAI Codex). Type 'quit' to exit.")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("chat", "login"), default="chat", nargs="?")
    arguments = parser.parse_args()

    if arguments.command == "login":
        command_login()
    elif arguments.command == "chat":
        command_chat()


if __name__ == "__main__":
    main()
