"""Command-line entry point for behavioral evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cases import run_model_evaluations, run_scripted_evaluations
from .framework import write_traces


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run desktop assistant behavioral evaluations."
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
    for trace in traces:
        print(f"{trace.scenario}: {'PASS' if trace.passed else 'REVIEW'}")

    return 0 if all(trace.passed for trace in traces) else 1


if __name__ == "__main__":
    raise SystemExit(main())
