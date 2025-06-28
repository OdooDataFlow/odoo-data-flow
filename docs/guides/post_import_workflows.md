# Guide: Post-Import Workflows

The `odoo-data-flow` library provides a powerful system for running automated actions on your data _after_ it has been imported into Odoo. This is handled by the `odoo-data-flow workflow` command.

This feature is designed for complex data migrations where simple importing is not enough. A common use case is in accounting, where imported draft invoices must be validated, reconciled, and paid. Instead of performing these actions manually in the Odoo UI for thousands of records, you can automate them with a workflow.

## The `invoice-v9` Workflow

The library currently includes a built-in workflow specifically for processing customer invoices (`account.invoice`) in Odoo version 9.

**Warning:** This workflow uses legacy Odoo v9 API calls and will **not** work on modern Odoo versions (10.0+). It is provided as a reference and an example of how a post-import process can be structured.

The workflow allows you to perform the following actions on your imported invoices:

- **`tax`**: Computes taxes for imported draft invoices.
- **`validate`**: Validates draft invoices, moving them to the 'Open' state.
- **`pay`**: Registers a payment against an open invoice, moving it to the 'Paid' state.
- **`proforma`**: Converts draft invoices to pro-forma invoices.
- **`rename`**: A utility to move a value from a custom field to the official `number` field.

### Usage

You run the workflow from the command line, specifying which action(s) you want to perform.

```bash
odoo-data-flow workflow invoice-v9 [OPTIONS]
```

### Command-Line Options

| Option              | Description                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-c`, `--config`    | **Required**. Path to your `connection.conf` file.                                                                                                                       |
| `--action`          | The workflow action to run (`tax`, `validate`, `pay`, `proforma`, `rename`). This option can be used multiple times. If omitted, all actions are run in a logical order. |
| `--field`           | **Required**. The name of the field in `account.invoice` that holds the legacy status from your source system. The workflow uses this to find the right invoices.        |
| `--status-map`      | **Required**. A dictionary string that maps Odoo states to your legacy statuses. For example: `"{'open': ['OP', 'Validated'], 'paid': ['PD']}"`                          |
| `--paid-date-field` | **Required**. The name of the field containing the payment date, used by the `pay` action.                                                                               |
| `--payment-journal` | **Required**. The database ID (integer) of the `account.journal` to be used for payments.                                                                                |
| `--max-connection`  | The number of parallel threads to use for processing. Defaults to `4`.                                                                                                   |

### Example Command

Imagine you have imported thousands of invoices. Now, you want to find all the invoices with a legacy status of "Validated" and move them to the "Open" state in Odoo.

You would run the following command:

```bash
odoo-data-flow workflow invoice-v9 \
    --config conf/connection.conf \
    --action validate \
    --field x_studio_legacy_status \
    --status-map "{'open': ['Validated']}" \
    --paid-date-field x_studio_payment_date \
    --payment-journal 5
```

This command will:

1. Connect to Odoo.
2. Search for all `account.invoice` records where `x_studio_legacy_status` is 'Validated'.
3. Run the `validate_invoice` function on those records, triggering the workflow to open them.
