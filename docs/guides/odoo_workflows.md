# Guide: Automation & Workflows

The `odoo-data-flow` library includes a powerful system for running automated actions directly on your Odoo server. These actions are split into two main categories:

* **Module Management (`module` command):** For administrative tasks like installing, uninstalling, or updating the list of available modules. These are typically run to prepare an Odoo environment.
* **Data Workflows (`workflow` command):** For running multi-step processes on data that already exists in the database, such as validating a batch of imported invoices.

---

## Module Management

You can automate the installation, upgrade, and uninstallation of modules directly from the command line. This is particularly useful for setting up a new database or ensuring your environments are consistent.

### Step 1: Updating the Apps List

Before you can install a new module, you must first ensure that Odoo is aware of it. The `module update-list` command scans your Odoo instance's addons path and updates the list of available modules.

This is a crucial first step to run after you have added new custom modules to your server. The command will wait for the server to complete the scan before finishing, so you can safely chain it with an installation command.

#### Usage
```bash
odoo-data-flow module update-list
```

#### Command-Line Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | **(Optional)** Path to your `connection.conf` file. Defaults to `conf/connection.conf`. |


### Step 2: Installing or Upgrading Modules


The `module install` command will install new modules or upgrade them if they are already installed.

#### Usage
```bash
odoo-data-flow module install --modules sale_management,mrp
```

#### Command-Line Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | **(Optional)** Path to your `connection.conf` file. Defaults to `conf/connection.conf`. |
| `-m`, `--modules`| **(Required)** A comma-separated string of technical module names to install or upgrade. |

### Uninstalling Modules

The `module uninstall` command will uninstall modules that are currently installed.

#### Usage
```bash
odoo-data-flow module uninstall --modules stock,account
```

#### Command-Line Options

| Option | Description |
| :--- | :--- |
| `-c`, `--config` | **(Optional)** Path to your `connection.conf` file. Defaults to `conf/connection.conf`. |
| `-m`, `--modules`| **(Required)** A comma-separated string of technical module names to uninstall. |

---

## Data Processing Workflows

This command group is for running multi-step processes on records that are already in the database.

### The `invoice-v9` Workflow (Legacy Example)

The library includes a built-in workflow specifically for processing customer invoices (`account.invoice`) in **Odoo version 9**.

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
