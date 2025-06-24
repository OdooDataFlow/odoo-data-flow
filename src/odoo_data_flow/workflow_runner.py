"""Workflow Runner, invoke workflows.

This module acts as a dispatcher for running post-import workflows
from the command line.
"""

import ast

from .lib.conf_lib import get_connection_from_config
from .lib.workflow.invoice_v9 import InvoiceWorkflowV9
from .logging_config import log


def run_invoice_v9_workflow(
    actions,
    config,
    field,
    status_map_str,
    paid_date_field,
    payment_journal,
    max_connection,
):
    """Initializes and runs the requested actions for the InvoiceWorkflowV9."""
    log.info("--- Initializing Invoice Workflow for Odoo v9 ---")

    try:
        connection = get_connection_from_config(config_file=config)

        # Safely evaluate the status map string into a dictionary
        status_map = ast.literal_eval(status_map_str)

        if not isinstance(status_map, dict):
            raise TypeError("Status map must be a dictionary.")

    except Exception as e:
        log.error(f"Failed to initialize workflow: {e}")
        return

    # Instantiate the legacy workflow class
    wf = InvoiceWorkflowV9(
        connection,
        field=field,
        status_map=status_map,
        paid_date_field=paid_date_field,
        payment_journal=payment_journal,
        max_connection=max_connection,
    )

    # Run the requested actions in a specific order
    if not actions or "all" in actions:
        actions = ["tax", "validate", "pay", "proforma", "rename"]

    log.info(f"Executing workflow actions: {', '.join(actions)}")

    if "tax" in actions:
        wf.set_tax()
    if "validate" in actions:
        wf.validate_invoice()
    if "pay" in actions:
        wf.paid_invoice()
    if "proforma" in actions:
        wf.proforma_invoice()
    if "rename" in actions:
        rename_field = "x_legacy_number"
        log.info(f"Note: 'rename' action is using a placeholder field: {rename_field}")
        wf.rename(rename_field)

    log.info("--- Invoice Workflow Finished ---")


# We can add runners for other workflows here in the future
# def run_sale_order_workflow(...):
#     pass
