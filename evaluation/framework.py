"""Reusable evaluation execution and trace infrastructure."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from desktop_ai_assistant.model import ModelBackend
from desktop_ai_assistant.orchestrator import AssistantOrchestrator, TurnOutcome
from desktop_ai_assistant.registry import ToolRegistry
from desktop_ai_assistant.types import FinalResponse, Message, ToolCallResponse


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationTrace:
    """Recorded result of running one evaluation scenario.

    ``objective_checks`` contains named, machine-evaluated assertions for the
    outcome. ``qualitative_review`` marks scenarios that also require human
    assessment and therefore cannot be judged by those assertions alone.
    """

    scenario: str
    prompt: str
    model: str
    response: str
    tool_calls: list[str]
    error: str | None
    objective_checks: dict[str, bool]
    messages: list[Message]
    qualitative_review: bool = False

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.objective_checks.values())

    def as_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class Scenario:
    """Inputs and objective criteria for one deterministic evaluation.

    ``scripted_responses`` is the sequence returned by the scripted model as
    the orchestrator advances through the scenario. ``checks`` receives the
    completed :class:`TurnOutcome` and returns named boolean assertions that
    are copied into the trace's ``objective_checks`` field.
    """

    name: str
    prompt: str
    scripted_responses: Sequence[FinalResponse | ToolCallResponse]
    expected_tool_calls: list[str] | None
    checks: Callable[[TurnOutcome], dict[str, bool]]
    qualitative_review: bool = False


def run_scenario(
    scenario: Scenario,
    model: ModelBackend,
    registry: ToolRegistry,
) -> EvaluationTrace:
    outcome = AssistantOrchestrator(model, registry, logging.getLogger("evaluation")).handle(
        scenario.prompt
    )
    checks = scenario.checks(outcome)
    if scenario.expected_tool_calls is not None:
        checks["expected_tool_calls"] = outcome.tool_calls == scenario.expected_tool_calls
    return EvaluationTrace(
        scenario=scenario.name,
        prompt=scenario.prompt,
        model=model.identifier,
        response=outcome.response,
        tool_calls=outcome.tool_calls,
        error=outcome.error,
        objective_checks=checks,
        messages=outcome.messages,
        qualitative_review=scenario.qualitative_review,
    )


def write_traces(traces: Sequence[EvaluationTrace], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(trace) for trace in traces], default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )
