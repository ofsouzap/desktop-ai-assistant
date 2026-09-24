import re

from click.testing import CliRunner

from desktop_ai_assistant import cli


def test_help_is_provided_by_click() -> None:
    result = CliRunner().invoke(cli.main, ["--help"])
    output = re.sub(r"\s+", " ", result.output)

    assert result.exit_code == 0
    assert "Usage:" in output
    assert "--help" in output
    assert "constrained Linux/Sway desktop assistant REPL" in output
    assert "does not provide shell" in output
