"""Test the high-level data migration orchestrator."""

from unittest.mock import MagicMock, patch

from odoo_data_flow.lib import mapper
from odoo_data_flow.migrator import run_migration


@patch("odoo_data_flow.migrator.run_import_for_migration")
@patch("odoo_data_flow.migrator.run_export_for_migration")
@patch("odoo_data_flow.migrator.Processor")
def test_run_migration_success_with_mapping(
    mock_processor: MagicMock,
    mock_run_export: MagicMock,
    mock_run_import: MagicMock,
) -> None:
    """Tests the successful end-to-end migration workflow with a custom mapping."""
    # 1. Setup
    # Mock the return value of the export function
    mock_run_export.return_value = (["id", "name"], [["1", "Source Name"]])

    # Mock the processor and its process method
    mock_processor_instance = MagicMock()
    mock_processor_instance.process.return_value = (
        ["id", "name"],
        [["1", "Transformed Name"]],
    )
    mock_processor.return_value = mock_processor_instance

    # Define a valid custom mapping where the value is a callable mapper function
    custom_mapping = {
        "name": mapper.val("name", postprocess=lambda x, s: f"Transformed {x}")
    }

    # 2. Action
    run_migration(
        config_export="src.conf",
        config_import="dest.conf",
        model="res.partner",
        fields=["id", "name"],
        mapping=custom_mapping,
    )

    # 3. Assertions
    mock_run_export.assert_called_once_with(
        config="src.conf",
        model="res.partner",
        domain="[]",
        fields=["id", "name"],
        worker=1,
        batch_size=100,
        technical_names=True,
    )
    mock_processor.assert_called_once_with(
        header=["id", "name"], data=[["1", "Source Name"]]
    )
    mock_processor_instance.process.assert_called_once_with(
        custom_mapping, filename_out=""
    )
    mock_run_import.assert_called_once_with(
        config="dest.conf",
        model="res.partner",
        header=["id", "name"],
        data=[["1", "Transformed Name"]],
        worker=1,
        batch_size=10,
    )


@patch("odoo_data_flow.migrator.run_import_for_migration")
@patch("odoo_data_flow.migrator.run_export_for_migration")
@patch("odoo_data_flow.migrator.Processor")
def test_run_migration_success_no_mapping(
    mock_processor: MagicMock,
    mock_run_export: MagicMock,
    mock_run_import: MagicMock,
) -> None:
    """Tests that a 1-to-1 mapping is generated if none is provided."""
    # 1. Setup
    mock_run_export.return_value = (["id", "name"], [["1", "Source Name"]])

    # Mock the processor and its methods
    mock_processor_instance = MagicMock()
    # Simulate get_o2o_mapping returning a simple callable mapping
    mock_processor_instance.get_o2o_mapping.return_value = {
        "id": MagicMock(func=lambda line, state: line["id"]),
        "name": MagicMock(func=lambda line, state: line["name"]),
    }
    mock_processor_instance.process.return_value = (
        ["id", "name"],
        [["1", "Source Name"]],
    )
    mock_processor.return_value = mock_processor_instance

    # 2. Action
    run_migration(
        config_export="src.conf", config_import="dest.conf", model="res.partner"
    )

    # 3. Assertions
    mock_run_export.assert_called_once()
    assert mock_run_export.call_args.kwargs["technical_names"] is True
    mock_processor_instance.get_o2o_mapping.assert_called_once()
    mock_processor_instance.process.assert_called_once()
    mock_run_import.assert_called_once()


@patch("odoo_data_flow.migrator.run_import_for_migration")
@patch("odoo_data_flow.migrator.run_export_for_migration")
@patch("odoo_data_flow.migrator.log.warning")
def test_run_migration_no_data_exported(
    mock_log_warning: MagicMock,
    mock_run_export: MagicMock,
    mock_run_import: MagicMock,
) -> None:
    """Tests that the migration stops gracefully if no data is exported."""
    # 1. Setup: Simulate the export function returning no data
    mock_run_export.return_value = ([], [])

    # 2. Action
    run_migration(
        config_export="src.conf", config_import="dest.conf", model="res.partner"
    )

    # 3. Assertions
    mock_run_export.assert_called_once()
    assert mock_run_export.call_args.kwargs["technical_names"] is True
    mock_log_warning.assert_called_once_with("No data exported. Migration finished.")
    # The import function should never be called
    mock_run_import.assert_not_called()
