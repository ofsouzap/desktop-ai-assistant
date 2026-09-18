from pathlib import Path

import pytest

from desktop_ai_assistant.logging import configure_logging


def test_console_logging_writes_errors_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = configure_logging(tmp_path, console=True)
    logger.error("development failure")

    assert "development failure" in capsys.readouterr().err


def test_console_logging_omits_info_events(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = configure_logging(tmp_path, console=True)
    logger.info("internal event")

    assert "internal event" not in capsys.readouterr().err


def test_file_logging_remains_available(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path, console=True)
    logger.info("persisted event")

    assert "persisted event" in (tmp_path / "assistant.jsonl").read_text()