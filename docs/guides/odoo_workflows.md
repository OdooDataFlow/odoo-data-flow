# Guide: Odoo Automation Workflows

The `odoo-data-flow` library includes a powerful system for running automated actions directly on your Odoo server. This is handled by the `odoo-data-flow workflow` command group.

These workflows are designed to automate common administrative tasks that are part of a larger data migration or deployment process, such as installing modules, validating invoices, or triggering other specific business logic.

---

## Managing Odoo Modules

You can automate the installation, upgrade, and uninstallation of modules directly from the command line. This is particularly useful for setting up a new database or ensuring your environments are consistent.

### Installing or Upgrading Modules

The `workflow install-modules` command will install new modules or upgrade them if they are already installed.

#### Usage
```bash
odoo-data-flow workflow install-modules --modules sale_management,mrp
```

#### Command-Line Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | **(Optional)** Path to your `connection.conf` file. Defaults to `conf/connection.conf`. |
| `-m`, `--modules`| **(Required)** A comma-separated string of technical module names to install or upgrade. |

### Uninstalling Modules

The `workflow uninstall-modules` command will uninstall modules that are currently installed.

#### Usage
```bash
odoo-data-flow workflow uninstall-modules --modules stock,account
```

#### Command-Line Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | **(Optional)** Path to your `connection.conf` file. Defaults to `conf/connection.conf`. |
| `-m`, `--modules`| **(Required)** A comma-separated string of technical module names to uninstall. |

---

## Legacy Workflows

### The `invoice-v9` Workflow

The library also includes a built-in workflow specifically for processing customer invoices (`account.invoice`) in **Odoo version 9**.

**Warning:** This workflow uses legacy Odoo v9 API calls and will **not** work on modern Odoo versions (10.0+). It is provided as a reference and an example of how a post-import process can be structured.

The workflow allows you to perform the following actions on your imported invoices:

- **`tax`**: Computes taxes for imported draft invoices.
- **`validate`**: Validates draft invoices, moving them to the 'Open' state.
- **`pay`**: Registers a payment against an open invoice, moving it to the 'Paid' state.
- **`proforma`**: Converts draft invoices to pro-forma invoices.
- **`rename`**: A utility to move a value from a custom field to the official `number` field.

#### Usage

You run the workflow from the command line, specifying which action(s) you want to perform.

```bash
odoo-data-flow workflow invoice-v9 [OPTIONS]
```

#### Command-Line Options

| Option              | Description                                                                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-c`, `--config`      | Path to your `connection.conf` file. Defaults to `conf/connection.conf`.                                                                                         |
| `--action`          | The workflow action to run (`tax`, `validate`, `pay`, `proforma`, `rename`). This option can be used multiple times. If omitted, all actions are run.              |
| `--field`           | **Required**. The name of the field in `account.invoice` that holds the legacy status from your source system. The workflow uses this to find the right invoices. |
| `--status-map`      | **Required**. A dictionary string that maps Odoo states to your legacy statuses. For example: `"{'open': ['OP', 'Validated'], 'paid': ['PD']}"`                   |
| `--paid-date-field` | **Required**. The name of the field containing the payment date, used by the `pay` action.                                                                         |
| `--payment-journal` | **Required**. The database ID (integer) of the `account.journal` to be used for payments.                                                                          |
| `--max-connection`  | The number of parallel threads to use for processing. Defaults to `4`.                                                                                           |

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
