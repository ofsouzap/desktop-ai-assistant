"""Structured, full-fidelity local logging."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        details = getattr(record, "details", None)
        if details is not None:
            payload["details"] = details
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default, sort_keys=True)


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def configure_logging(log_directory: Path) -> logging.Logger:
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("desktop_ai_assistant")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(log_directory / "assistant.jsonl", encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **details: Any) -> None:
    logger.info(event, extra={"details": details})

