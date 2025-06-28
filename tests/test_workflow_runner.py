"""Test Logging functionality."""
# tests/test_workflow_runner.py

from unittest.mock import MagicMock, patch

from odoo_data_flow.workflow_runner import run_invoice_v9_workflow


@patch("odoo_data_flow.workflow_runner.InvoiceWorkflowV9")
@patch("odoo_data_flow.workflow_runner.get_connection_from_config")
def test_run_invoice_v9_workflow_all_actions(
    mock_get_connection: MagicMock, mock_invoice_workflow: MagicMock
) -> None:
    """Tests that when action is 'all', all workflow methods are called."""
    # 1. Setup: Create a mock instance of the workflow class
    mock_wf_instance = MagicMock()
    mock_invoice_workflow.return_value = mock_wf_instance

    # 2. Action
    run_invoice_v9_workflow(
        actions=["all"],
        config="dummy.conf",
        field="x_legacy_status",
        status_map_str="{'open': ['OP'], 'paid': ['PA']}",
        paid_date_field="x_paid_date",
        payment_journal=1,
        max_connection=4,
    )

    # 3. Assertions
    mock_get_connection.assert_called_once_with(config_file="dummy.conf")
    mock_invoice_workflow.assert_called_once()

    # Check that all methods were called
    mock_wf_instance.set_tax.assert_called_once()
    mock_wf_instance.validate_invoice.assert_called_once()
    mock_wf_instance.paid_invoice.assert_called_once()
    mock_wf_instance.proforma_invoice.assert_called_once()
    mock_wf_instance.rename.assert_called_once_with("x_legacy_number")


@patch("odoo_data_flow.workflow_runner.InvoiceWorkflowV9")
@patch("odoo_data_flow.workflow_runner.get_connection_from_config")
def test_run_invoice_v9_workflow_specific_action(
    mock_get_connection: MagicMock, mock_invoice_workflow: MagicMock
) -> None:
    """Tests that when a specific action is provided, only that method is called."""
    # 1. Setup
    mock_wf_instance = MagicMock()
    mock_invoice_workflow.return_value = mock_wf_instance

    # 2. Action
    run_invoice_v9_workflow(
        actions=["pay"],  # Only run the 'pay' action
        config="dummy.conf",
        field="x_legacy_status",
        status_map_str="{'paid': ['PA']}",
        paid_date_field="x_paid_date",
        payment_journal=1,
        max_connection=4,
    )

    # 3. Assertions
    # Check that only the paid_invoice method was called
    mock_wf_instance.paid_invoice.assert_called_once()

    # Check that other methods were NOT called
    mock_wf_instance.set_tax.assert_not_called()
    mock_wf_instance.validate_invoice.assert_not_called()
    mock_wf_instance.proforma_invoice.assert_not_called()
    mock_wf_instance.rename.assert_not_called()


@patch("odoo_data_flow.workflow_runner.get_connection_from_config")  # This was missing
@patch("odoo_data_flow.workflow_runner.log.error")
def test_run_invoice_v9_workflow_bad_status_map(
    mock_log_error: MagicMock, mock_get_connection: MagicMock
) -> None:
    """Tests that an error is logged if the status_map string is not a valid dict."""
    run_invoice_v9_workflow(
        actions=["all"],
        config="dummy.conf",
        field="x_legacy_status",
        status_map_str="this-is-not-a-dict",
        paid_date_field="x_paid_date",
        payment_journal=1,
        max_connection=4,
    )
    mock_log_error.assert_called_once()
    assert "Failed to initialize workflow" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.workflow_runner.get_connection_from_config")
@patch("odoo_data_flow.workflow_runner.log.error")
def test_run_invoice_v9_workflow_connection_fails(
    mock_log_error: MagicMock, mock_get_connection: MagicMock
) -> None:
    """Tests that an error is logged if the connection fails."""
    mock_get_connection.side_effect = Exception("Connection Refused")
    run_invoice_v9_workflow(
        actions=["all"],
        config="bad.conf",
        field="x_legacy_status",
        status_map_str="{}",
        paid_date_field="x_paid_date",
        payment_journal=1,
        max_connection=4,
    )
    mock_log_error.assert_called_once()
    assert "Failed to initialize workflow" in mock_log_error.call_args[0][0]
