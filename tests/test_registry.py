import unittest

from desktop_ai_assistant.registry import ArgumentSpec, ToolRegistry
from desktop_ai_assistant.types import ToolArguments, ToolCall


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()

        @self.registry.register(
            "repeat", "Return supplied text.", (ArgumentSpec("text", str, "Text"),)
        )
        def repeat(arguments: ToolArguments) -> str:
            return str(arguments["text"])

    def test_exposes_registered_schema(self) -> None:
        schema = self.registry.schemas()[0]
        self.assertEqual(schema.name, "repeat")
        self.assertEqual(schema.arguments[0].name, "text")

    def test_dispatches_valid_call(self) -> None:
        result = self.registry.dispatch(ToolCall("1", "repeat", {"text": "hello"}))
        self.assertEqual(result.content, "hello")
        self.assertFalse(result.is_error)

    def test_rejects_invalid_arguments(self) -> None:
        result = self.registry.dispatch(ToolCall("1", "repeat", {"text": 1}))
        self.assertTrue(result.is_error)
        self.assertIn("must be str", result.content)

    def test_rejects_unregistered_tool(self) -> None:
        result = self.registry.dispatch(ToolCall("1", "shell", {}))
        self.assertTrue(result.is_error)
