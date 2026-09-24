"""Command-line entry point for behavioral evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .cases import run_model_evaluations, run_scripted_evaluations
from .framework import write_traces


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run desktop assistant behavioral evaluations. "
            "Exit codes: 0 for all checks passing, 1 for objective failures or "
            "errors, and 2 when qualitative review is required."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("scripted", "openrouter"),
        required=True,
        help=(
            "model backend: scripted runs deterministic checks without network access; "
            "openrouter evaluates the configured OpenRouter model"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-traces.json"),
        help="path for the JSON trace output (default: evaluation-traces.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    if arguments.backend == "scripted":
        traces = run_scripted_evaluations()
    elif arguments.backend == "openrouter":
        from desktop_ai_assistant.openrouter import OpenRouterModelBackend

        traces = run_model_evaluations(OpenRouterModelBackend())
    else:
        raise ValueError(f"Unsupported backend: {arguments.backend}")

    write_traces(traces, arguments.output)

    pass_count, fail_count, review_count = 0, 0, 0
    for trace in traces:
        match trace.passed_state:
            case "full_pass":
                result = "PASS"
                pass_count += 1
            case "failed_objective_checks" | "has_error":
                result = "FAIL"
                fail_count += 1
            case "passing_but_requires_qualitative_review":
                result = "REVIEW"
                review_count += 1

        print(f"{trace.scenario}: {result}")

    if fail_count > 0:
        return 1
    elif review_count > 0:
        assert fail_count == 0
        return 2
    else:
        assert fail_count == review_count == 0
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
