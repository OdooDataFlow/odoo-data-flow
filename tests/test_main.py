"""Test cases for the __main__ module."""

import pytest
from click.testing import CliRunner

# CORRECTED: Use an underscore for the package name in the import.
from odoo_data_flow import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds_without_command(runner: CliRunner) -> None:
    """Test main Succeeds.

    It exits with a status code of 0 when no command is provided
    and should show the main help message.
    """
    # CORRECTED: The entry point function from our __main__.py is now 'cli'.
    result = runner.invoke(__main__.cli)
    assert result.exit_code == 0
    # A good basic test is to ensure the main commands are listed in the help output.
    assert "import" in result.output
    assert "export" in result.output
    assert "path-to-image" in result.output
    assert "url-to-image" in result.output


def test_main_shows_version(runner: CliRunner) -> None:
    """It shows the version of the package when --version is used."""
    result = runner.invoke(__main__.cli, ["--version"])
    assert result.exit_code == 0
    # This checks that the command runs and that the word 'version'
    # appears in the output, which is a robust check for the --version flag.
    assert "version" in result.output


# You can also add more specific tests for each command.
# For example, testing that the 'import' command fails without required options:
def test_import_fails_without_options(runner: CliRunner) -> None:
    """The import command should fail if required options are missing."""
    # We invoke the 'import' sub-command directly.
    result = runner.invoke(__main__.cli, ["import"])
    # It should exit with a non-zero status code because options are missing.
    assert result.exit_code != 0
    # Click's error message should mention the missing options.
    assert "Missing option" in result.output
    assert "--file" in result.output
