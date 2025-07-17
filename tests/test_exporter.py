"""Test the high-level export orchestrator."""

from unittest.mock import MagicMock, patch

import polars as pl

from odoo_data_flow.exporter import (
    _show_error_panel,
    _show_success_panel,
    run_export,
    run_export_for_migration,
)


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_success_panel")
def test_run_export_success(
    mock_show_success: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests the main `run_export` function in a success scenario."""
    # 1. Setup
    mock_export_data.return_value = pl.DataFrame({"id": [1, 2]})

    # 2. Action
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id,name",
        output="partners.csv",
        domain="[('is_company', '=', True)]",
        context="{'lang': 'en_US'}",
    )

    # 3. Assertions
    mock_export_data.assert_called_once()
    call_kwargs = mock_export_data.call_args.kwargs
    assert call_kwargs["config_file"] == "dummy.conf"
    assert call_kwargs["model"] == "res.partner"
    assert call_kwargs["header"] == ["id", "name"]
    assert call_kwargs["domain"] == [("is_company", "=", True)]
    assert call_kwargs["output"] == "partners.csv"
    assert call_kwargs["context"] == {"lang": "en_US"}
    mock_show_success.assert_called_once()


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_bad_domain(
    mock_show_error_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests that `run_export` handles a bad domain string."""
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id",
        output="dummy.csv",
        domain="this-is-not-a-list",
    )
    mock_show_error_panel.assert_called_once()
    assert "Invalid Domain" in mock_show_error_panel.call_args.args[0]
    mock_export_data.assert_not_called()


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_bad_context(
    mock_show_error_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests that `run_export` handles a bad context string."""
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id",
        output="dummy.csv",
        context="this-is-not-a-dict",
    )
    mock_show_error_panel.assert_called_once()
    assert "Invalid Context" in mock_show_error_panel.call_args.args[0]
    mock_export_data.assert_not_called()


@patch("odoo_data_flow.exporter.export_threaded.export_data")
def test_run_export_for_migration(mock_export_data: MagicMock) -> None:
    """Tests the `run_export_for_migration` function."""
    # 1. Setup
    mock_export_data.return_value = pl.DataFrame({"id": [1], "name": ["Test Partner"]})
    fields_list = ["id", "name"]

    # 2. Action
    header, data = run_export_for_migration(
        config="conf/test.conf",
        model="res.partner",
        fields=fields_list,
    )

    # 3. Assertions
    mock_export_data.assert_called_once()
    call_kwargs = mock_export_data.call_args.kwargs
    assert call_kwargs["config_file"] == "conf/test.conf"
    assert call_kwargs["model"] == "res.partner"
    assert call_kwargs["header"] == fields_list
    assert call_kwargs["output"] is None  # Ensures in-memory operation

    assert header == ["id", "name"]
    assert data == [[1, "Test Partner"]]


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter.log.warning")
def test_run_export_for_migration_bad_domain(
    mock_log_warning: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests that `run_export_for_migration` handles a bad domain string."""
    mock_export_data.return_value = pl.DataFrame()
    run_export_for_migration(
        config="dummy.conf",
        model="res.partner",
        fields=["id"],
        domain="bad-domain",
    )
    mock_log_warning.assert_called_once()
    assert "Invalid domain string" in mock_log_warning.call_args[0][0]
    assert mock_export_data.call_args.kwargs["domain"] == []


@patch("odoo_data_flow.exporter.export_threaded.export_data")
def test_run_export_for_migration_no_data(mock_export_data: MagicMock) -> None:
    """Tests `run_export_for_migration` when no data is returned."""
    mock_export_data.return_value = pl.DataFrame({"id": [], "name": []})
    header, data = run_export_for_migration(
        config="dummy.conf", model="res.partner", fields=["id", "name"]
    )
    assert header == ["id", "name"]
    assert data == []


@patch("odoo_data_flow.exporter.Console")
def test_show_error_panel(mock_console: MagicMock) -> None:
    """Test that the error panel is shown correctly."""
    mock_print = MagicMock()
    mock_console.return_value = MagicMock(print=mock_print)
    _show_error_panel("Test Title", "Test Message")
    mock_print.assert_called_once()


@patch("odoo_data_flow.exporter.Console")
def test_show_success_panel(mock_console: MagicMock) -> None:
    """Test that the success panel is shown correctly."""
    mock_print = MagicMock()
    mock_console.return_value = MagicMock(print=mock_print)
    _show_success_panel("Test Message")
    mock_print.assert_called_once()


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_error_panel")
def test_run_export_failure(
    mock_show_error_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests the main `run_export` function in a failure scenario."""
    mock_export_data.return_value = None
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id,name",
        output="partners.csv",
    )
    mock_show_error_panel.assert_called_once()
    assert "Export Failed" in mock_show_error_panel.call_args.args[0]


@patch("odoo_data_flow.exporter.export_threaded.export_data")
def test_run_export_for_migration_bad_context(mock_export_data: MagicMock) -> None:
    """Tests `run_export_for_migration` with a bad context."""
    mock_export_data.return_value = pl.DataFrame()
    run_export_for_migration(
        config="dummy.conf",
        model="res.partner",
        fields=["id"],
        context="bad-context",
    )
    assert mock_export_data.call_args.kwargs["context"] == {}


@patch("odoo_data_flow.exporter.export_threaded.export_data")
def test_run_export_for_migration_none_df(mock_export_data: MagicMock) -> None:
    """Tests `run_export_for_migration` when the dataframe is None."""
    mock_export_data.return_value = None
    header, data = run_export_for_migration(
        config="dummy.conf",
        model="res.partner",
        fields=["id"],
    )
    assert header == ["id"]
    assert data is None


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_success_panel")
def test_run_export_success_with_dataframe(
    mock_show_success_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests the main `run_export` function in a success scenario with a dataframe."""
    mock_export_data.return_value = pl.DataFrame({"id": [1, 2]})
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id,name",
        output="partners.csv",
    )
    mock_show_success_panel.assert_called_once()


@patch("odoo_data_flow.exporter.export_threaded.export_data")
@patch("odoo_data_flow.exporter._show_success_panel")
def test_run_export_with_empty_dataframe(
    mock_show_success_panel: MagicMock, mock_export_data: MagicMock
) -> None:
    """Tests the main `run_export` function with an empty dataframe."""
    mock_export_data.return_value = pl.DataFrame()
    run_export(
        config="dummy.conf",
        model="res.partner",
        fields="id,name",
        output="partners.csv",
    )
    mock_show_success_panel.assert_called_once()
