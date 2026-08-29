"""Central XDG path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

APPLICATION_NAME = "desktop-ai-assistant"


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    config: Path
    data: Path
    state: Path

    def ensure_directories(self) -> None:
        for directory in (self.config, self.data, self.state):
            directory.mkdir(parents=True, exist_ok=True)


def application_paths(
    environment: Mapping[str, str] | None = None, home: Path | None = None
) -> ApplicationPaths:
    values = os.environ if environment is None else environment
    home_directory = Path.home() if home is None else home
    return ApplicationPaths(
        config=Path(values.get("XDG_CONFIG_HOME", home_directory / ".config"))
        / APPLICATION_NAME,
        data=Path(values.get("XDG_DATA_HOME", home_directory / ".local/share"))
        / APPLICATION_NAME,
        state=Path(values.get("XDG_STATE_HOME", home_directory / ".local/state"))
        / APPLICATION_NAME,
    )
