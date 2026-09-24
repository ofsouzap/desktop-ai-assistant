"""Runtime configuration with XDG and environment-backed defaults."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILENAME = "config.toml"
DEFAULT_MAXIMUM_STEPS = 5


@dataclass(frozen=True, slots=True)
class AssistantConfig:
    model: str
    maximum_steps: int
    console_logs: bool

    def validate(self) -> None:
        if self.maximum_steps < 1:
            raise ValueError("maximum_steps must be at least one.")


def load_config(
    config_directory: Path, environment: Mapping[str, str] | None = None
) -> AssistantConfig:
    """Load config file values, overridden by environment variables."""

    if environment is None:
        environment = {}

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

    def string(
        *,
        environment_key: str,
        file_key: str,
        default_value: str,
        name_for_error_message: str | None = None,
    ) -> str:
        return _string_setting(
            environment.get(environment_key),
            file_values.get(file_key, default_value),
            name_for_error_message,
        )

    def integer(
        *,
        environment_key: str,
        file_key: str,
        default_value: int,
        name_for_error_message: str | None = None,
    ) -> int:
        return _integer_setting(
            environment.get(environment_key),
            file_values.get(file_key, default_value),
            name_for_error_message,
        )

    def boolean(
        *,
        environment_key: str,
        file_key: str,
        default_value: bool,
        name_for_error_message: str | None = None,
    ) -> bool:
        return _boolean_setting(
            environment.get(environment_key),
            file_values.get(file_key, default_value),
            name_for_error_message,
        )

    config = AssistantConfig(
        model=string(
            environment_key="OPENROUTER_MODEL",
            file_key="model",
            default_value="openrouter/free",
            name_for_error_message="model",
        ),
        maximum_steps=integer(
            environment_key="DESKTOP_AI_ASSISTANT_MAXIMUM_STEPS",
            file_key="maximum_steps",
            default_value=DEFAULT_MAXIMUM_STEPS,
            name_for_error_message="maximum_steps",
        ),
        console_logs=boolean(
            environment_key="DESKTOP_AI_ASSISTANT_CONSOLE_LOGS",
            file_key="console_logs",
            default_value=False,
            name_for_error_message="console_logs",
        ),
    )

    config.validate()
    return config


def _string_setting(
    environment_value: str | None,
    file_value: object,
    name_for_error_message: str | None = None,
) -> str:
    name_for_error_message = name_for_error_message or "Value"

    value: object = environment_value if environment_value is not None else file_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name_for_error_message} must be a non-empty string.")
    else:
        return value


def _integer_setting(
    environment_value: str | None,
    file_value: object,
    name_for_error_message: str | None = None,
) -> int:
    name_for_error_message = name_for_error_message or "Value"

    value: object = environment_value if environment_value is not None else file_value
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise TypeError(f"{name_for_error_message} must be an integer.") from error
    else:
        raise TypeError(f"{name_for_error_message} must be an integer.")


def _boolean_setting(
    environment_value: str | None,
    file_value: object,
    name_for_error_message: str | None = None,
) -> bool:
    name_for_error_message = name_for_error_message or "Value"

    value: object = environment_value if environment_value is not None else file_value
    if isinstance(value, bool):
        return value
    elif isinstance(value, str) and value.lower() in {"0", "false", "no", "off"}:
        return False
    elif isinstance(value, str) and value.lower() in {"1", "true", "yes", "on"}:
        return True
    else:
        raise ValueError(f"{name_for_error_message} must be a boolean.")
