from pathlib import Path

import pytest

from desktop_ai_assistant.config import AssistantConfig, load_config


def test_loads_toml_and_environment_overrides(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        """\
[assistant]
model = "file-model"
maximum_steps = 3
console_logs = true
""",
        encoding="utf-8",
    )

    environment = {
        "OPENROUTER_MODEL": "environment-model",
        "DESKTOP_AI_ASSISTANT_MAXIMUM_STEPS": "7",
        "DESKTOP_AI_ASSISTANT_CONSOLE_LOGS": "false",
    }

    config = load_config(
        tmp_path,
        environment,
    )

    assert config.model == "environment-model"
    assert config.maximum_steps == 7
    assert not config.console_logs


def test_rejects_invalid_step_limit(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        """\
[assistant]
maximum_steps = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one"):
        load_config(tmp_path, {})


def test_validates_step_limit() -> None:
    config = AssistantConfig(model="model", maximum_steps=0, console_logs=False)

    with pytest.raises(ValueError, match="at least one"):
        config.validate()


@pytest.mark.parametrize(
    ("setting", "value", "error_message"),
    [
        ("model", "123", "model must be a non-empty string"),
        ("maximum_steps", "true", "maximum_steps must be an integer"),
        ("console_logs", "1", "console_logs must be a boolean"),
    ],
)
def test_rejects_wrong_type_config_values(
    tmp_path: Path, setting: str, value: str, error_message: str
) -> None:
    (tmp_path / "config.toml").write_text(
        f"[assistant]\n{setting} = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=error_message):
        load_config(tmp_path, {})


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "false",
        "FALSE",
        "no",
        "No",
        "off",
        "OFF",
    ],
)
def test_parses_false_boolean_values(tmp_path: Path, value: str) -> None:
    config = load_config(tmp_path, {"DESKTOP_AI_ASSISTANT_CONSOLE_LOGS": value})

    assert config.console_logs is False


@pytest.mark.parametrize(
    "value",
    [
        "1",
        "true",
        "TRUE",
        "yes",
        "Yes",
        "on",
        "ON",
    ],
)
def test_parses_true_boolean_values(tmp_path: Path, value: str) -> None:
    config = load_config(tmp_path, {"DESKTOP_AI_ASSISTANT_CONSOLE_LOGS": value})

    assert config.console_logs is True


@pytest.mark.parametrize("value", ["", "2", "maybe", "enabled"])
def test_rejects_invalid_boolean_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="console_logs must be a boolean"):
        load_config(tmp_path, {"DESKTOP_AI_ASSISTANT_CONSOLE_LOGS": value})
