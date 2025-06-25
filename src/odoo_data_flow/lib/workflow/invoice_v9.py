"""Invoice helper for odoo version 9.

This module contains a legacy workflow helper for processing imported
invoices in Odoo v9. It is preserved for reference but will need to be
updated to work with modern Odoo versions.
"""

from time import time
from typing import Any
from xmlrpc.client import Fault

from ...logging_config import log
from ..internal.rpc_thread import RpcThread


class InvoiceWorkflowV9:
    """Automate odoo 9 Invoice Workflow.

    A class to automate the lifecycle of imported invoices in Odoo v9,
    such as validating, paying, and setting taxes.
    """

    def __init__(
        self,
        connection: Any,
        field: str,
        status_map: dict[str, list[str]],
        paid_date_field: str,
        payment_journal: int,
        max_connection: int = 4,
    ) -> None:
        """Initializes the workflow processor.

        Args:
            connection: An active odoo-client-lib connection object.
            field: The field that contains the legacy status from source data
            status_map: A dict mapping Odoo states to lists of legacy states.
                        e.g., {'open': ['status1'], 'paid': ['status2']}
            paid_date_field: The field containing the payment date.
            payment_journal: The database ID of the payment journal to use.
            max_connection: The number of parallel threads to use.
        """
        self.connection = connection
        self.invoice_obj = connection.get_model("account.invoice")
        self.payment_obj = connection.get_model("account.payment")
        self.account_invoice_tax = self.connection.get_model("account.invoice.tax")
        self.field = field
        self.status_map = status_map
        self.paid_date = paid_date_field
        self.payment_journal = payment_journal
        self.max_connection = max_connection
        self.time = time()

    def _display_percent(self, i: int, percent_step: int, total: int) -> None:
        if i % percent_step == 0:
            percentage = round(i / float(total) * 100, 2)
            elapsed_time = time() - self.time
            log.info(f"{percentage}% : {i}/{total} time {elapsed_time:.2f} sec")

    def set_tax(self) -> None:
        """Finds draft invoices and computes their taxes."""

        def create_tax(invoice_id: int) -> None:
            taxes = self.invoice_obj.get_taxes_values(invoice_id)
            for tax in taxes.values():
                self.account_invoice_tax.create(tax)

        invoices: list[int] = self.invoice_obj.search(
            [
                ("state", "=", "draft"),
                ("type", "=", "out_invoice"),
                ("tax_line_ids", "=", False),
            ]
        )
        total = len(invoices)
        percent_step = int(total / 5000) or 1
        self.time = time()
        rpc_thread = RpcThread(self.max_connection)
        log.info(f"Computing tax for {total} invoices...")
        for i, invoice_id in enumerate(invoices):
            self._display_percent(i, percent_step, total)
            rpc_thread.spawn_thread(create_tax, [invoice_id])
        rpc_thread.wait()

    def validate_invoice(self) -> None:
        """Finds and validates invoices that should be open or paid."""
        statuses_to_validate = self.status_map.get("open", []) + self.status_map.get(
            "paid", []
        )
        invoice_to_validate: list[int] = self.invoice_obj.search(
            [
                (self.field, "in", statuses_to_validate),
                ("state", "=", "draft"),
                ("type", "=", "out_invoice"),
            ]
        )
        total = len(invoice_to_validate)
        percent_step = int(total / 5000) or 1
        rpc_thread = RpcThread(1)  # Validation should be single-threaded
        log.info(f"Validating {total} invoices...")
        self.time = time()
        for i, invoice_id in enumerate(invoice_to_validate):
            self._display_percent(i, percent_step, total)
            fun = self.connection.get_service("object").exec_workflow
            rpc_thread.spawn_thread(
                fun,
                [
                    self.connection.database,
                    self.connection.user_id,
                    self.connection.password,
                    "account.invoice",
                    "invoice_open",
                    invoice_id,
                ],
            )
        rpc_thread.wait()

    def proforma_invoice(self) -> None:
        """Finds and moves invoices to the pro-forma state."""
        invoice_to_proforma: list[int] = self.invoice_obj.search(
            [
                (self.field, "in", self.status_map.get("proforma", [])),
                ("state", "=", "draft"),
                ("type", "=", "out_invoice"),
            ]
        )
        total = len(invoice_to_proforma)
        percent_step = int(total / 100) or 1
        self.time = time()
        rpc_thread = RpcThread(self.max_connection)
        log.info(f"Setting {total} invoices to pro-forma...")
        for i, invoice_id in enumerate(invoice_to_proforma):
            self._display_percent(i, percent_step, total)
            fun = self.connection.get_service("object").exec_workflow
            rpc_thread.spawn_thread(
                fun,
                [
                    self.connection.database,
                    self.connection.user_id,
                    self.connection.password,
                    "account.invoice",
                    "invoice_proforma2",
                    invoice_id,
                ],
                {},
            )
        rpc_thread.wait()

    def paid_invoice(self) -> None:
        """Finds open invoices and registers payments for them."""

        def pay_single_invoice(
            data_update: dict[str, Any], wizard_context: dict[str, Any]
        ) -> None:
            fields_to_get = [
                "communication",
                "currency_id",
                "invoice_ids",
                "payment_difference",
                "partner_id",
                "payment_method_id",
                "payment_difference_handling",
                "journal_id",
                "state",
                "writeoff_account_id",
                "payment_date",
                "partner_type",
                "hide_payment_method",
                "payment_method_code",
                "partner_bank_account_id",
                "amount",
                "payment_type",
            ]
            data = self.payment_obj.default_get(fields_to_get, context=wizard_context)
            data.update(data_update)
            wizard_id = self.payment_obj.create(data, context=wizard_context)
            try:
                self.payment_obj.post([wizard_id], context=wizard_context)
            except Fault:
                # Odoo may raise a fault for various reasons
                # (e.g., already paid),
                # which can be ignored in a batch process.
                pass

        invoices_to_paid: list[dict[str, Any]] = self.invoice_obj.search_read(
            domain=[
                (self.field, "in", self.status_map.get("paid", [])),
                ("state", "=", "open"),
                ("type", "=", "out_invoice"),
            ],
            fields=[self.paid_date, "date_invoice"],
        )
        total = len(invoices_to_paid)
        percent_step = int(total / 1000) or 1
        self.time = time()
        rpc_thread = RpcThread(self.max_connection)
        log.info(f"Registering payment for {total} invoices...")
        for i, invoice in enumerate(invoices_to_paid):
            self._display_percent(i, percent_step, total)
            wizard_context = {
                "active_id": invoice["id"],
                "active_ids": [invoice["id"]],
                "active.model": "account.invoice",
                "default_invoice_ids": [(4, invoice["id"], 0)],
                "type": "out_invoice",
                "journal_type": "sale",
            }
            data_update = {
                "journal_id": self.payment_journal,
                "payment_date": invoice.get(self.paid_date)
                or invoice.get("date_invoice"),
                "payment_method_id": 1,  # Manual
            }
            rpc_thread.spawn_thread(
                pay_single_invoice, [data_update, wizard_context], {}
            )
        rpc_thread.wait()

    def rename(self, name_field: str) -> None:
        """Utility to move a value from a custom field to the invoice number."""
        invoices_to_rename: list[dict[str, Any]] = self.invoice_obj.search_read(
            domain=[
                (name_field, "!=", False),
                (name_field, "!=", "0.0"),
                ("state", "!=", "draft"),
                ("type", "=", "out_invoice"),
            ],
            fields=[name_field],
        )
        total = len(invoices_to_rename)
        percent_step = int(total / 1000) or 1
        self.time = time()
        rpc_thread = RpcThread(int(self.max_connection * 1.5))
        log.info(f"Renaming {total} invoices...")
        for i, invoice in enumerate(invoices_to_rename):
            self._display_percent(i, percent_step, total)
            update_vals = {"number": invoice[name_field], name_field: False}
            rpc_thread.spawn_thread(
                self.invoice_obj.write, [invoice["id"], update_vals], {}
            )
        rpc_thread.wait()
