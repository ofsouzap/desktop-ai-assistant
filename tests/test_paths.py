from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from desktop_ai_assistant.paths import APPLICATION_NAME, application_paths


class ApplicationPathsTests(unittest.TestCase):
    def test_honors_xdg_environment(self) -> None:
        paths = application_paths(
            {
                "XDG_CONFIG_HOME": "/config",
                "XDG_DATA_HOME": "/data",
                "XDG_STATE_HOME": "/state",
            }
        )
        self.assertEqual(paths.config, Path("/config") / APPLICATION_NAME)
        self.assertEqual(paths.data, Path("/data") / APPLICATION_NAME)
        self.assertEqual(paths.state, Path("/state") / APPLICATION_NAME)

    def test_creates_directories(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = application_paths({}, Path(temporary_directory))
            paths.ensure_directories()
            self.assertTrue(paths.config.is_dir())
            self.assertTrue(paths.data.is_dir())
            self.assertTrue(paths.state.is_dir())

