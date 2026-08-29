import logging
import unittest

from desktop_ai_assistant.model import ScriptedModelBackend
from desktop_ai_assistant.orchestrator import AssistantOrchestrator
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import FinalResponse, ToolArguments, ToolCall, ToolCallResponse


class OrchestratorTests(unittest.TestCase):
    def _assistant(
        self, responses: list[FinalResponse | ToolCallResponse], maximum_steps: int = 5
    ) -> AssistantOrchestrator:
        registry = ToolRegistry()

        @registry.register("ping", "Return pong.")
        def ping(arguments: ToolArguments) -> str:
            self.assertEqual(arguments, {})
            return "pong"

        return AssistantOrchestrator(
            ScriptedModelBackend(responses), registry, logging.getLogger("test"), maximum_steps
        )

    def test_runs_sequential_tool_calls_before_final_response(self) -> None:
        assistant = self._assistant(
            [
                ToolCallResponse(ToolCall("one", "ping", {})),
                ToolCallResponse(ToolCall("two", "ping", {})),
                FinalResponse("Done."),
            ]
        )
        outcome = assistant.handle("Do the task")
        self.assertEqual(outcome.response, "Done.")
        self.assertEqual(outcome.tool_calls, ["ping", "ping"])

    def test_stops_at_limit(self) -> None:
        assistant = self._assistant(
            [ToolCallResponse(ToolCall(str(index), "ping", {})) for index in range(3)],
            maximum_steps=2,
        )
        outcome = assistant.handle("loop")
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.tool_calls, ["ping", "ping"])

    def test_tool_error_is_returned_to_model(self) -> None:
        assistant = self._assistant(
            [
                ToolCallResponse(ToolCall("bad", "missing", {})),
                FinalResponse("Recovered."),
            ]
        )
        outcome = assistant.handle("try")
        self.assertEqual(outcome.response, "Recovered.")
