"""Test cases for the __main__ module."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

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
    result = runner.invoke(__main__.cli)
    assert result.exit_code == 0
    assert "import" in result.output
    assert "export" in result.output
    assert "path-to-image" in result.output
    assert "url-to-image" in result.output


def test_main_shows_version(runner: CliRunner) -> None:
    """It shows the version of the package when --version is used."""
    result = runner.invoke(__main__.cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_import_fails_without_options(runner: CliRunner) -> None:
    """The import command should fail if required options are missing."""
    result = runner.invoke(__main__.cli, ["import"])
    assert result.exit_code != 0
    assert "Missing option" in result.output
    assert "--file" in result.output


@patch("odoo_data_flow.__main__.run_import")
def test_import_command_calls_runner(
    mock_run_import: MagicMock, runner: CliRunner
) -> None:
    """Tests that the import command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli,
        [
            "import",
            "--config",
            "my.conf",
            "--file",
            "my.csv",
            "--model",
            "res.partner",
        ],
    )
    assert result.exit_code == 0
    mock_run_import.assert_called_once()
    call_kwargs = mock_run_import.call_args.kwargs
    assert call_kwargs["config"] == "my.conf"
    assert call_kwargs["filename"] == "my.csv"
    assert call_kwargs["model"] == "res.partner"


@patch("odoo_data_flow.__main__.run_export")
def test_export_command_calls_runner(
    mock_run_export: MagicMock, runner: CliRunner
) -> None:
    """Tests that the export command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli,
        [
            "export",
            "--config",
            "my.conf",
            "--output",
            "my.csv",
            "--model",
            "res.partner",
            "--fields",
            "id,name",
        ],
    )
    assert result.exit_code == 0
    mock_run_export.assert_called_once()


MOCK_TARGET = "odoo_data_flow.__main__.run_export"


@patch(MOCK_TARGET)
def test_export_cmd_default_mode(mock_run_export: MagicMock, runner: CliRunner) -> None:
    """Verifies that --streaming is False by default."""
    result = runner.invoke(
        __main__.cli,
        [
            "export",
            "--config",
            "dummy.conf",
            "--model",
            "res.partner",
            "--output",
            "out.csv",
            "--fields",
            "id,name",
            "--domain",
            "[]",
        ],
    )

    assert result.exit_code == 0
    mock_run_export.assert_called_once()
    call_kwargs = mock_run_export.call_args.kwargs
    assert "streaming" in call_kwargs
    assert call_kwargs["streaming"] is False


@patch(MOCK_TARGET)
def test_export_cmd_streaming_mode(
    mock_run_export: MagicMock, runner: CliRunner
) -> None:
    """Verifies that --streaming flag sets the streaming argument to True."""
    result = runner.invoke(
        __main__.cli,
        [
            "export",
            "--config",
            "dummy.conf",
            "--model",
            "res.partner",
            "--output",
            "out.csv",
            "--fields",
            "id,name",
            "--domain",
            "[]",
            "--streaming",
        ],
    )

    assert result.exit_code == 0
    mock_run_export.assert_called_once()
    call_kwargs = mock_run_export.call_args.kwargs
    assert "streaming" in call_kwargs
    assert call_kwargs["streaming"] is True


@patch("odoo_data_flow.__main__.run_path_to_image")
def test_path_to_image_command_calls_runner(
    mock_run_path_to_image: MagicMock, runner: CliRunner
) -> None:
    """Tests that the path-to-image command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli, ["path-to-image", "my.csv", "--fields", "image"]
    )
    assert result.exit_code == 0
    mock_run_path_to_image.assert_called_once()


@patch("odoo_data_flow.__main__.run_url_to_image")
def test_url_to_image_command_calls_runner(
    mock_run_url_to_image: MagicMock, runner: CliRunner
) -> None:
    """Tests that the url-to-image command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli, ["url-to-image", "my.csv", "--fields", "image_url"]
    )
    assert result.exit_code == 0
    mock_run_url_to_image.assert_called_once()


@patch("odoo_data_flow.__main__.run_migration")
def test_migrate_command_calls_runner(
    mock_run_migration: MagicMock, runner: CliRunner
) -> None:
    """Tests that the migrate command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli,
        [
            "migrate",
            "--config-export",
            "src.conf",
            "--config-import",
            "dest.conf",
            "--model",
            "res.partner",
            "--fields",
            "id,name",
            "--mapping",
            "{'name': ('val', 'name')}",
        ],
    )
    assert result.exit_code == 0
    mock_run_migration.assert_called_once()
    call_kwargs = mock_run_migration.call_args.kwargs
    assert isinstance(call_kwargs["mapping"], dict)


@patch("odoo_data_flow.__main__.run_migration")
def test_migrate_command_bad_mapping_syntax(
    mock_run_migration: MagicMock, runner: CliRunner
) -> None:
    """Tests that the migrate command handles a bad mapping string."""
    result = runner.invoke(
        __main__.cli,
        [
            "migrate",
            "--config-export",
            "src.conf",
            "--config-import",
            "dest.conf",
            "--model",
            "res.partner",
            "--fields",
            "id,name",
            "--mapping",
            "this-is-not-a-dict",
        ],
    )
    assert result.exit_code == 0
    assert "Invalid mapping provided" in result.output
    mock_run_migration.assert_not_called()


@patch("odoo_data_flow.__main__.run_migration")
def test_migrate_command_mapping_not_a_dict(
    mock_run_migration: MagicMock, runner: CliRunner
) -> None:
    """Tests that migrate command handles a valid literal that is not a dict."""
    result = runner.invoke(
        __main__.cli,
        [
            "migrate",
            "--config-export",
            "src.conf",
            "--config-import",
            "dest.conf",
            "--model",
            "res.partner",
            "--fields",
            "id,name",
            "--mapping",
            "['this', 'is', 'a', 'list']",  # Valid literal, but not a dict
        ],
    )
    assert result.exit_code == 0
    assert "Mapping must be a dictionary" in result.output
    mock_run_migration.assert_not_called()


@patch("odoo_data_flow.__main__.run_invoice_v9_workflow")
def test_workflow_command_calls_runner(
    mock_run_workflow: MagicMock, runner: CliRunner
) -> None:
    """Tests that the workflow command calls the correct runner function."""
    result = runner.invoke(
        __main__.cli,
        [
            "workflow",
            "invoice-v9",
            "--config",
            "my.conf",
            "--field",
            "x_status",
            "--status-map",
            "{}",
            "--paid-date-field",
            "x_date",
            "--payment-journal",
            "1",
        ],
    )
    assert result.exit_code == 0
    mock_run_workflow.assert_called_once()


@patch("odoo_data_flow.__main__.run_update_module_list")
def test_module_update_list_command(
    mock_run_update: MagicMock, runner: CliRunner
) -> None:
    """Tests that the 'module update-list' command calls the correct function."""
    result = runner.invoke(
        __main__.cli, ["module", "update-list", "--config", "c.conf"]
    )
    assert result.exit_code == 0
    mock_run_update.assert_called_once_with(config="c.conf")


@patch("odoo_data_flow.__main__.run_module_installation")
def test_module_install_command(mock_run_install: MagicMock, runner: CliRunner) -> None:
    """Tests that the 'module install' command calls the correct function."""
    result = runner.invoke(__main__.cli, ["module", "install", "--modules", "sale,mrp"])
    assert result.exit_code == 0
    mock_run_install.assert_called_once_with(
        config="conf/connection.conf", modules=["sale", "mrp"]
    )


@patch("odoo_data_flow.__main__.run_module_uninstallation")
def test_module_uninstall_command(
    mock_run_uninstall: MagicMock, runner: CliRunner
) -> None:
    """Tests that the 'module uninstall' command calls the correct function."""
    result = runner.invoke(
        __main__.cli, ["module", "uninstall", "--modules", "sale,purchase"]
    )
    assert result.exit_code == 0
    mock_run_uninstall.assert_called_once_with(
        config="conf/connection.conf", modules=["sale", "purchase"]
    )


@patch("odoo_data_flow.__main__.run_language_installation")
def test_module_install_languages_command(
    mock_run_install: MagicMock, runner: CliRunner
) -> None:
    """Tests that the 'module install-languages' command calls the correct function."""
    result = runner.invoke(
        __main__.cli,
        ["module", "install-languages", "--languages", "en_US,fr_FR"],
    )
    assert result.exit_code == 0
    mock_run_install.assert_called_once_with(
        config="conf/connection.conf", languages=["en_US", "fr_FR"]
    )
