"""Runtime configuration with XDG and environment-backed defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONFIG_FILENAME = "config.toml"
DEFAULT_MAXIMUM_STEPS = 5


@dataclass(frozen=True, slots=True)
class AssistantConfig:
    model: str
    maximum_steps: int
    console_logs: bool


def load_config(
    config_directory: Path,
    environment: Mapping[str, str] | None = None,
) -> AssistantConfig:
    """Load config file values, overridden by environment variables."""
    values = os.environ if environment is None else environment
    file_values: dict[str, object] = {}
    config_path = config_directory / CONFIG_FILENAME
    if config_path.exists():
        try:
            with config_path.open("rb") as config_file:
                parsed = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"Unable to read configuration: {config_path}") from error
        raw_assistant = parsed.get("assistant", {})
        if not isinstance(raw_assistant, dict):
            raise ValueError("The [assistant] configuration must be a table.")
        file_values = raw_assistant

    model = _string_setting(
        values.get("OPENROUTER_MODEL"),
        file_values.get("model", "google/gemma-4-31b-it:free"),
        "model",
    )
    maximum_steps = _integer_setting(
        values.get("DESKTOP_AI_ASSISTANT_MAXIMUM_STEPS"),
        file_values.get("maximum_steps", DEFAULT_MAXIMUM_STEPS),
        "maximum_steps",
    )
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be at least one.")

    console_logs = _boolean_setting(
        values.get("DESKTOP_AI_ASSISTANT_CONSOLE_LOGS"),
        file_values.get("console_logs", False),
        "console_logs",
    )
    return AssistantConfig(model, maximum_steps, console_logs)


def _string_setting(
    environment_value: str | None, file_value: object, name: str
) -> str:
    value: object = environment_value if environment_value is not None else file_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _integer_setting(
    environment_value: str | None, file_value: object, name: str
) -> int:
    value: object = environment_value if environment_value is not None else file_value
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer.") from error
    raise ValueError(f"{name} must be an integer.")


def _boolean_setting(
    environment_value: str | None, file_value: object, name: str
) -> bool:
    value: object = environment_value if environment_value is not None else file_value
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"0", "false", "no", "off"}:
        return False
    if isinstance(value, str) and value.lower() in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean.")
