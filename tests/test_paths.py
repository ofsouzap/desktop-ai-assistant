from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_ai_assistant.paths import APPLICATION_NAME, application_paths


def test_honors_xdg_environment() -> None:
    paths = application_paths(
        {
            "XDG_CONFIG_HOME": "/config",
            "XDG_DATA_HOME": "/data",
            "XDG_STATE_HOME": "/state",
        }
    )
    assert paths.config == Path("/config") / APPLICATION_NAME
    assert paths.data == Path("/data") / APPLICATION_NAME
    assert paths.state == Path("/state") / APPLICATION_NAME


def test_uses_fallback_for_missing_xdg_value() -> None:
    home = Path("/home/tester")
    paths = application_paths({"XDG_CONFIG_HOME": "/config"}, home)
    assert paths.config == Path("/config") / APPLICATION_NAME
    assert paths.data == home / ".local/share" / APPLICATION_NAME
    assert paths.state == home / ".local/state" / APPLICATION_NAME


def test_uses_fallbacks_when_no_xdg_values_are_set() -> None:
    home = Path("/home/tester")
    paths = application_paths({}, home)
    assert paths.config == home / ".config" / APPLICATION_NAME
    assert paths.data == home / ".local/share" / APPLICATION_NAME
    assert paths.state == home / ".local/state" / APPLICATION_NAME


def test_creates_directories() -> None:
    with TemporaryDirectory() as temporary_directory:
        paths = application_paths({}, Path(temporary_directory))
        paths.ensure_directories()
        assert paths.config.is_dir()
        assert paths.data.is_dir()
        assert paths.state.is_dir()
