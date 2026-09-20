import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_ai_assistant.integrations.tools.inventory import (
    INVENTORY_FILENAME,
    InventoryStore,
    register_inventory_tools,
)
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import ToolCall


def make_store(path: Path) -> InventoryStore:
    return InventoryStore(path, logging.getLogger("test.inventory"))


def test_reads_missing_inventory_as_empty() -> None:
    with TemporaryDirectory() as directory:
        assert make_store(Path(directory) / INVENTORY_FILENAME).read() == ""


def test_appends_one_line_entries() -> None:
    with TemporaryDirectory() as directory:
        store = make_store(Path(directory) / INVENTORY_FILENAME)
        store.append("coffee filters")
        store.append("Laptop charger: office desk")
        assert store.read() == "coffee filters\nLaptop charger: office desk\n"


def test_overwrite_replaces_inventory() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / INVENTORY_FILENAME
        store = make_store(path)
        store.append("old entry")
        store.overwrite("new entry\nanother note\n")
        assert store.read() == "new entry\nanother note\n"


def test_accepts_large_inventory_entries_and_content() -> None:
    with TemporaryDirectory() as directory:
        store = make_store(Path(directory) / INVENTORY_FILENAME)
        large_text = "a" * (64 * 1024)
        store.append(large_text)
        assert store.read() == large_text + "\n"
        store.overwrite(large_text)
        assert store.read() == large_text


def test_rejects_multiline_entry() -> None:
    with TemporaryDirectory() as directory:
        store = make_store(Path(directory) / INVENTORY_FILENAME)
        registry = ToolRegistry()
        register_inventory_tools(registry, store)
        multiline = registry.dispatch(
            ToolCall("1", "inventory_append", {"text": "one\ntwo"})
        )
        assert multiline.is_error
        assert store.read() == ""


def test_registered_tools_expose_format_instructions_and_persist() -> None:
    with TemporaryDirectory() as directory:
        store = make_store(Path(directory) / INVENTORY_FILENAME)
        registry = ToolRegistry()
        register_inventory_tools(registry, store)
        schemas = {schema.name: schema for schema in registry.schemas()}
        append = registry.dispatch(ToolCall("1", "inventory_append", {"text": "tea"}))
        read = registry.dispatch(ToolCall("2", "inventory_read", {}))
        assert "one item or note per line" in schemas["inventory_append"].description
        assert "For example" in schemas["inventory_overwrite"].description
        assert not append.is_error
        assert read.content == "tea\n"
