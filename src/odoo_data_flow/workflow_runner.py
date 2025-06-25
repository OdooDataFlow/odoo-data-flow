"""Workflow Runner, invoke workflows.

This module acts as a dispatcher for running post-import workflows
from the command line.
"""

import ast
from typing import Any

from .lib.conf_lib import get_connection_from_config
from .lib.workflow.invoice_v9 import InvoiceWorkflowV9
from .logging_config import log


def run_invoice_v9_workflow(
    actions: list[str],
    config: str,
    field: str,
    status_map_str: str,
    paid_date_field: str,
    payment_journal: int,
    max_connection: int,
) -> None:
    """Initializes and runs the requested actions for the InvoiceWorkflowV9.

    Args:
        actions: A list of workflow actions to perform (e.g., ['tax', 'validate']).
        config: The path to the connection configuration file.
        field: The source field containing the legacy invoice status.
        status_map_str: A string representation of the dictionary mapping Odoo
                        states to legacy states.
        paid_date_field: The source field containing the payment date.
        payment_journal: The database ID of the payment journal.
        max_connection: The number of parallel threads to use.
    """
    log.info("--- Initializing Invoice Workflow for Odoo v9 ---")

    try:
        connection: Any = get_connection_from_config(config_file=config)

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
    final_actions = actions
    if not final_actions or "all" in final_actions:
        final_actions = ["tax", "validate", "pay", "proforma", "rename"]

    log.info(f"Executing workflow actions: {', '.join(final_actions)}")

    if "tax" in final_actions:
        wf.set_tax()
    if "validate" in final_actions:
        wf.validate_invoice()
    if "pay" in final_actions:
        wf.paid_invoice()
    if "proforma" in final_actions:
        wf.proforma_invoice()
    if "rename" in final_actions:
        rename_field = "x_legacy_number"
        log.info(f"Note: 'rename' action is using a placeholder field: {rename_field}")
        wf.rename(rename_field)

    log.info("--- Invoice Workflow Finished ---")


# We can add runners for other workflows here in the future
# def run_sale_order_workflow(...):
#     pass
