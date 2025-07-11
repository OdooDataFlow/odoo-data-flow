"""Test The Exporter Orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from odoo_data_flow.exporter import (
    _show_error_panel,
    run_export,
    run_export_for_migration,
    run_export_from_file,
)


@patch(
    "odoo_data_flow.exporter.export_threaded.export_data_to_file",
    return_value=(True, "OK"),
)
def test_run_export(mock_export_data: MagicMock) -> None:
    """Tests the main `run_export` function.

    Verifies that it correctly parses string arguments and calls the underlying
    `export_threaded.export_data` function with the correct parameters.
    """
    # 1. Setup
    config_file = "conf/test.conf"
    filename = "output.csv"
    model = "res.partner"
    fields_str = "id,name,email"
    domain_str = "[('is_company', '=', True)]"
    context_str = "{'lang': 'fr_FR'}"

    # 2. Action: Call the function we want to test
    run_export(
        config=config_file,
        filename=filename,
        model=model,
        fields=fields_str,
        domain=domain_str,
        context=context_str,
        worker=2,
        batch_size=50,
        separator="|",
        encoding="latin1",
    )

    # 3. Assertions: Check that the mocked function was called correctly
    mock_export_data.assert_called_once()

    # Correctly inspect positional and keyword arguments
    pos_args, kw_args = mock_export_data.call_args

    assert pos_args[0] == config_file
    assert pos_args[1] == model
    assert pos_args[2] == [("is_company", "=", True)]  # parsed domain
    assert pos_args[3] == ["id", "name", "email"]  # parsed fields

    assert kw_args.get("context") == {"lang": "fr_FR"}
    assert kw_args.get("output") == filename
    assert kw_args.get("max_connection") == 2
    assert kw_args.get("batch_size") == 50
    assert kw_args.get("separator") == "|"
    assert kw_args.get("encoding") == "latin1"


@patch("odoo_data_flow.exporter.export_threaded.export_data_for_migration")
def test_run_export_for_migration(mock_export_data: MagicMock) -> None:
    """Tests the `run_export_for_migration` function.

    Verifies that it correctly prepares arguments for an in-memory data export.
    """
    # 1. Setup
    # Simulate the return value from the mocked function
    mock_export_data.return_value = (["id", "name"], [["1", "Test Partner"]])
    fields_list = ["id", "name"]

    # 2. Action
    header, data = run_export_for_migration(
        config="conf/test.conf",
        model="res.partner",
        fields=fields_list,
    )

    # 3. Assertions
    mock_export_data.assert_called_once()

    pos_args, kw_args = mock_export_data.call_args

    assert pos_args[0] == "conf/test.conf"
    assert pos_args[1] == "res.partner"
    assert pos_args[3] == fields_list  # Correctly check positional argument

    assert kw_args.get("output") is None, "Output should be None for in-memory return"

    assert header == ["id", "name"]
    assert data == [["1", "Test Partner"]]


@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_invalid_domain(mock_show_error_panel: MagicMock) -> None:
    """Tests that `run_export` logs an error for a malformed domain string."""
    # 1. Action
    run_export(
        config="dummy.conf",
        filename="dummy.csv",
        model="dummy.model",
        fields="id",
        domain="this-is-not-a-list",
    )

    # 2. Assertions
    mock_show_error_panel.assert_called_once()
    assert "Invalid Domain" in mock_show_error_panel.call_args[0][0]


@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_invalid_context(mock_show_error_panel: MagicMock) -> None:
    """Tests that `run_export` logs an error for a malformed context string."""
    # 1. Action
    run_export(
        config="dummy.conf",
        filename="dummy.csv",
        model="dummy.model",
        fields="id",
        context="this-is-not-a-dict",
    )

    # 2. Assertions
    mock_show_error_panel.assert_called_once()
    assert "Invalid Context" in mock_show_error_panel.call_args[0][0]


@patch("odoo_data_flow.exporter.export_threaded.export_data_for_migration")
def test_run_export_for_migration_bad_domain(
    mock_export_data: MagicMock,
) -> None:
    """Tests that `run_export_for_migration` handles a bad domain string."""
    mock_export_data.return_value = ([], [])
    run_export_for_migration(
        config="dummy.conf",
        model="res.partner",
        fields=["id"],
        domain="bad-domain",
    )
    # Assert that the domain passed to the core function is an empty list
    assert mock_export_data.call_args.args[2] == []


@patch("odoo_data_flow.exporter.Console")
def test_show_error_panel(mock_console: MagicMock) -> None:
    """Tests the `_show_error_panel` function."""
    _show_error_panel("Test Title", "Test Message")
    mock_console.assert_called_once_with(stderr=True, style="bold red")
    mock_console.return_value.print.assert_called_once()


@patch(
    "odoo_data_flow.exporter.export_threaded.export_data_to_file",
    return_value=(False, "Error"),
)
@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_fail(
    mock_show_error_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests that `run_export` calls `_show_error_panel` when the export fails."""
    run_export(
        config="dummy.conf",
        filename="dummy.csv",
        model="dummy.model",
        fields="id",
    )
    mock_show_error_panel.assert_called_once_with("Export Aborted", "Error")


@patch("odoo_data_flow.exporter.export_threaded.export_data_for_migration")
def test_run_export_for_migration_bad_context(mock_export_data: MagicMock) -> None:
    """Tests `run_export_for_migration` with a bad context string."""
    mock_export_data.return_value = ([], [])
    run_export_for_migration(
        config="dummy.conf",
        model="res.partner",
        fields=["id"],
        context="bad-context",
    )
    # Assert that the context passed is an empty dict
    assert mock_export_data.call_args.kwargs["context"] == {}


def test_run_export_from_file_not_implemented() -> None:
    """Tests that `run_export_from_file` raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        run_export_from_file(
            config="dummy.conf",
            filename="dummy.csv",
        )


@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_domain_not_list(mock_show_error_panel: MagicMock) -> None:
    """Tests that `run_export` logs an error for a domain that is not a list."""
    run_export(
        config="dummy.conf",
        filename="dummy.csv",
        model="dummy.model",
        fields="id",
        domain="{'is_company': True}",  # Not a list
    )
    mock_show_error_panel.assert_called_once()
    assert "Domain must be a list" in mock_show_error_panel.call_args[0][1]


@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_context_not_dict(mock_show_error_panel: MagicMock) -> None:
    """Tests that `run_export` logs an error for a context that is not a dict."""
    run_export(
        config="dummy.conf",
        filename="dummy.csv",
        model="dummy.model",
        fields="id",
        context="['lang', 'fr_FR']",  # Not a dict
    )
    mock_show_error_panel.assert_called_once()
    assert "Context must be a dictionary" in mock_show_error_panel.call_args[0][1]
