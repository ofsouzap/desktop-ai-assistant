"""Behavioral evaluation suite for the desktop assistant."""

from .cases import run_model_evaluations, run_scripted_evaluations
from .framework import EvaluationTrace

__all__ = ["EvaluationTrace", "run_model_evaluations", "run_scripted_evaluations"]
