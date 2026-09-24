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
