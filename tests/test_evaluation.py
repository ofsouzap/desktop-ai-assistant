import json

from desktop_ai_assistant.evaluation import run_scripted_evaluations


def test_scripted_evaluations_cover_inventory_and_sway() -> None:
    traces = run_scripted_evaluations()

    assert [trace.scenario for trace in traces] == [
        "inventory_remember_and_confirm",
        "sway_move_and_focus",
        "capability_boundary_explanation",
    ]
    assert all(trace.passed for trace in traces)
    assert traces[0].tool_calls == ["inventory_append", "inventory_read"]
    assert traces[1].tool_calls == [
        "list_windows",
        "move_window_to_workspace",
        "focus_workspace",
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
