"""Command-line entry point for behavioral evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cases import run_model_evaluations, run_scripted_evaluations
from .framework import write_traces


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run desktop assistant behavioral evaluations.")
    parser.add_argument("--scripted", action="store_true", help="avoid credentials and network access")
    parser.add_argument("--output", type=Path, default=Path("evaluation-traces.json"))
    arguments = parser.parse_args(argv)
    if arguments.scripted:
        traces = run_scripted_evaluations()
    else:
        from desktop_ai_assistant.openrouter import OpenRouterModelBackend

        traces = run_model_evaluations(OpenRouterModelBackend())
    write_traces(traces, arguments.output)
    for trace in traces:
        print(f"{trace.scenario}: {'PASS' if trace.passed else 'REVIEW'}")
    return 0 if all(trace.passed for trace in traces) else 1


if __name__ == "__main__":
    raise SystemExit(main())
