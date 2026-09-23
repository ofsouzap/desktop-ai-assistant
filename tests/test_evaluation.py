import json
import re

import pytest

from evaluation import run_scripted_evaluations
from evaluation.__main__ import parse_arguments


def test_evaluation_backend_is_required_and_limited() -> None:
    assert parse_arguments(["--backend", "scripted"]).backend == "scripted"
    assert parse_arguments(["--backend", "openrouter"]).backend == "openrouter"

    with pytest.raises(SystemExit):
        parse_arguments([])

    with pytest.raises(SystemExit):
        parse_arguments(["--backend", "unknown"])


def test_evaluation_help_describes_exit_codes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_arguments(["--help"])

    output = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "Exit codes:" in output
    assert "0 for all checks passing" in output
    assert "1 for objective failures or errors" in output
    assert "2 when qualitative review is required" in output


def test_scripted_evaluations_cover_inventory_and_sway() -> None:
    traces = run_scripted_evaluations()

    assert [trace.scenario for trace in traces] == [
        "inventory_remember_and_confirm",
        "sway_move_and_focus",
        "capability_boundary_explanation",
    ]
    assert all(
        (trace.passed_state in ("full_pass", "passing_but_requires_qualitative_review"))
        for trace in traces
    )
    assert traces[0].tool_calls == ["inventory_append", "inventory_read"]
    assert traces[1].tool_calls == [
        "list_windows",
        "move_window_to_workspace",
        "focus_window",
    ]
    assert traces[2].qualitative_review


def test_evaluation_trace_is_complete_json() -> None:
    trace = run_scripted_evaluations()[0]

    payload = json.loads(trace.as_json())
    assert payload["scenario"] == "inventory_remember_and_confirm"
    assert payload["prompt"]
    assert payload["model"] == "scripted"
    assert payload["messages"]
    assert payload["objective_checks"]["expected_tool_calls"] is True
