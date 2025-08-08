# Guide: A Deep Dive into Importing

This guide expands on the import workflow, providing a detailed look at the `Processor` class and, most importantly, the requirements for your input data files.

## Command-Line Usage

The primary way to import data is through the `import` command. If your configuration file is in the default location and your CSV file is named after the Odoo model, the command is very simple:

```bash
odoo-data-flow import --file path/to/res_partner.csv
```

## Pre-flight Checks

To save time and prevent common errors, `odoo-data-flow` automatically runs a series of pre-flight checks before starting the import process. These checks validate your environment and data to catch systemic issues that would otherwise cause the entire import to fail record by record.

Currently, the following checks are performed by default:

* **Field Existence Check**: Verifies that every column in your CSV header corresponds to an actual field on the target Odoo model. This immediately catches typos or field name changes between Odoo versions.
* **Language Check**: For imports into `res.partner` or `res.users`, this check scans the `lang` column in your CSV. It then verifies that all required languages are installed and active on the target Odoo database.

### Managing Pre-flight Checks

* **Disabling Checks**: If you need to bypass these validations for any reason, you can use the `--no-preflight-checks` flag.

    ```bash
    odoo-data-flow import --file ... --no-preflight-checks
    ```

* **Headless Mode**: The language check may prompt you to install missing languages. To run the import in a non-interactive environment (like a CI/CD pipeline), use the `--headless` flag. This will automatically approve the installation of any missing languages.

    ```bash
    odoo-data-flow import --file ... --headless
    ```

### Key Options for `import`

* `--config`: **(Optional)** Path to your connection configuration file. **Defaults to `conf/connection.conf`**.
* `--file`: **(Required)** Path to the source CSV file you want to import.
* `--model`: **(Optional)** The target Odoo model (e.g., `res.partner`). If you omit this option, the tool will automatically infer the model name from your CSV filename. For example, a file named `res_partner.csv` will be imported into the `res.partner` model.
* `--worker`: Number of parallel threads to use for the import.
* `--fail`: Runs the import in "fail mode," retrying only the records from the corresponding `_fail.csv` file.
* `--skip`: The number of initial lines to skip in the source file before reading the header.
* `--sep`: The character separating columns. Defaults to a semicolon (`;`).

## Automatic Field Verification

To prevent common errors, `odoo-data-flow` automatically verifies that every column in your CSV header exists as a field on the target Odoo model. This is a core part of the pre-flight checks that run by default before any data is imported.

This powerful check allows you to "fail fast" with a clear error message, rather than waiting for a large import to fail midway through due to a single typo in a column name.

!!! note "Now Automatic"
    This behavior is now the default and runs automatically. The old `--verify-fields` flag has been removed and is no longer necessary.

## The "Upsert" Strategy: How External IDs Work

A core feature of `odoo-data-flow` is its ability to safely handle both creating new records and updating existing ones in a single process. This is often called an "upsert" (update or insert) operation, and it is the default behavior of the tool.

This makes your data imports **idempotent**, meaning you can run the same import script multiple times without creating duplicate records.

### The Role of the External ID (`id` column)

This entire feature is powered by the mandatory `id` column in your CSV file. This column holds a unique **External ID** for every record.

When you run an import, Odoo's `load` method performs the following logic for each row:

1. **Check for External ID:** It looks at the value in the `id` column.

2. **If the ID Exists:** If a record with that same external ID is found in the database, Odoo **updates** that existing record with the new values from your file.

3. **If the ID Does Not Exist:** If no record with that external ID is found, Odoo **creates** a new record and assigns it that external ID.

This built-in upsert logic is essential for incremental data loads and for re-running scripts to correct or enrich data that has already been imported.

## Automatic Handling of Relational Data

The `import` command is now "smart." It automatically detects complex relationships in your data and uses the best strategy to import it.

When you import a file with self-referential fields (like `parent_id` on partners) or `many2many` fields, the importer will:
1.  **Automatically detect** these fields during a pre-flight check.
2.  **Automatically switch** to a robust, two-pass import strategy.
3.  **Automatically use** the `id` column as the unique identifier to map relationships.

This means for most standard cases, the import will work without any extra flags.

```bash
# If your file has an 'id' column, this is all you need.
odoo-data-flow import --file path/to/res_partner_with_parents.csv --model res.partner
```

If your unique identifier column is named something else (e.g., external_id), you must specify it using the --unique-id-field option.

```bash
# Use this if your unique ID column is not named 'id'.
odoo-data-flow import \
    --file path/to/my_data.csv \
    --unique-id-field "external_id"
    --model res.partner
```

You can also manually force the two-pass strategy by providing the --deferred-fields option.

!!! note
The two-pass strategy is not compatible with `--fail` mode.


## Input File Requirements

For a successful import into Odoo, the clean CSV file you generate (the `filename_out` in your script) must follow some important rules.

* **Encoding**: The file must be in `UTF-8` encoding.
* **One Model per File**: Each CSV file should only contain data for a single Odoo model (e.g., all `res.partner` records).
* **Header Row**: The first line of the file must be the header row. All column names must be the technical field names from the Odoo model (e.g., `name`, `parent_id`, `list_price`).
* **External ID**: All rows must have an `id` column containing a unique External ID (also known as an XML ID). This is essential for the "upsert" logic described above.
* **Field Separator**: The character separating columns can be defined with the `--separator` command-line option. The default is a semicolon (`;`). **Crucially, if a field's value contains the separator character, the entire field value must be enclosed in double quotes (`"`).**
* **Skipping Lines**: If your source file contains introductory lines before the header, you can use the `--skip` option to ignore them during the import process.

### Special Field Naming Conventions

To handle relational data and updates by database ID, the tool uses special column headers:

* **`/id` Suffix (for External IDs)**: When mapping to a `Many2one` or `Many2many` field, you must append `/id` to the field name (e.g., `partner_id/id`). This tells Odoo to look up the related record using the provided External ID.

* **`.id` Field Name (for Database IDs)**: To update a record using its existing database ID (an integer, not an external ID), use the special field name `.id`. When you use this, you should also provide an empty `id` column to tell Odoo you are updating, not creating a new record.

### Field Formatting Rules

Odoo's `load` method expects data for certain field types to be in a specific format.

* **Boolean**: Must be `1` for True and `0` for False. The `mapper.bool_val` can help with this.
* **Binary**: Must be a base64 encoded string. The `mapper.binary` and `mapper.binary_url_to_base64` functions handle this automatically.
* **Date & Datetime**: The format depends on the user's language settings in Odoo, but the standard, safe formats are `YYYY-MM-DD` for dates and `YYYY-MM-DD HH:MM:SS` for datetimes.
* **Float**: The decimal separator must be a dot (`.`). The `mapper.num` function handles converting comma separators automatically.
* **Selection**: Must contain the internal value for the selection, not the human-readable label (e.g., `'draft'` instead of `'Draft'`).
* **Many2one**: The column header must be suffixed with `/id`, and the value should be the external ID of the related record.
* **Many2many**: The column header must be suffixed with `/id`, and the value should be a comma-separated list of external IDs for the related records.

### Important: Do Not Import into Computed Fields

A common mistake is to try to import data directly into a field that is **computed** by Odoo (i.e., a field whose value is calculated automatically based on other fields). This will not work as expected.

**The Rule:** Always import the raw "ingredient" fields and let Odoo perform the calculation.

Even if an import into a computed field appears to succeed, the value will be overwritten the next time the record is saved or its source fields are changed. Most computed fields are also marked as `readonly`, which will cause the import to fail outright.

#### Example: `price_subtotal` on a Sales Order Line

The `price_subtotal` on a sales order line is calculated automatically from the quantity, unit price, and discount.

**Incorrect Mapping (This will fail or be overwritten):**
```python
order_line_mapping = {
    # ... other fields
    # Incorrectly trying to write directly to the computed field
    'price_subtotal': mapper.num('SubtotalFromSourceFile'),
}
```

**Correct Mapping (Import the source fields):**
```python
order_line_mapping = {
    # ... other fields
    # Import the raw ingredients and let Odoo do the calculation
    'product_uom_qty': mapper.num('Quantity'),
    'price_unit': mapper.num('UnitPrice'),
    'discount': mapper.num('DiscountPercent'),
}
```

By importing the source fields, you ensure that Odoo's business logic is triggered correctly and your data remains consistent.

---

## The `Processor` Class

The `Processor` is the central component of the transform phase. It handles reading the source file, applying the mapping, and generating the output files required for the load phase.

### Initialization

You initialize the processor by providing the path to your source data file and optional formatting parameters.

```python
from odoo_data_flow.lib.transform import Processor

processor = Processor(
    'origin/my_data.csv',      # Path to the source file
    separator=';',             # The character used to separate columns
)
```

The constructor takes the following arguments:

* **`filename` (str)**: The path to the CSV or XML file you want to transform.
* **`config_file` (str, optional)**: Path to the Odoo connection configuration file. This is used for operations that need to read metadata from the source database.
* **`separator` (str, optional)**: The column separator for CSV files. Defaults to `;`.
* **`preprocess` (function, optional)**: A function to modify the raw data _before_ mapping begins. See the [Data Transformations Guide](./data_transformations.md) for details.
* **`xml_root_tag` (str, optional)**: Required argument for processing XML files. See the [Advanced usage Guide](./advanced_usage.md) for details.


## The `process()` Method

This is the main method that executes the transformation. It takes your mapping dictionary and applies it to each row of the source file, writing the output to a new target file.

```python
processor.process(
    mapping=my_mapping_dict,
    filename_out='data/clean_data.csv',
    params=import_params_dict
)
```

The method takes these key arguments:

* **`mapping` (dict)**: **Required**. The mapping dictionary that defines the transformation rules for each column.
* **`filename_out` (str)**: **Required**. The path where the clean, transformed CSV file will be saved.
* **`params` (dict, optional)**: A crucial dictionary that holds the configuration for the `odoo-data-flow import` command. These parameters will be used when generating the `load.sh` script.

### Testing Your Mapping with a Dry Run

A crucial part of developing a transformation script is verifying that your mapping logic is correct without running a full import or writing files to disk. The `process` method includes a `dry_run=True` option for exactly this purpose.

When you use this option, the `Processor` will:
1.  Perform the complete data transformation in memory.
2.  **Not** write any CSV files or add any commands to the `write_to_file` queue.
3.  Print a beautifully formatted table to your console showing a sample of the first 10 rows of the transformed data.

This allows you to quickly inspect the output and debug your mappers.

**Example Usage:**
```python
# In your transform.py script

# ... (define your mapping and processor) ...

print("--- Running transformation in dry-run mode ---")
processor.process(
    mapping=my_mapping_dict,
    filename_out='data/clean_data.csv',
    params=import_params_dict,
    dry_run=True  # This is the key argument
)

# You can comment out the write_to_file call during a dry run
# processor.write_to_file("load_my_data.sh")
```

This will produce a table in your console, making it easy to see if your `concat`, `map_val`, or other mappers are producing the expected results.

### Configuring the Import Client with `params`

The `params` dictionary allows you to control the behavior of the import client without ever leaving your Python script. The keys in this dictionary map directly to the command-line options of the `odoo-data-flow import` command.

| `params` Key | `odoo-data-flow import` Option | Description                                                                                                       |
| ------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `config`     | `--config`                     | **For Migrations**. Path to the destination config file. Overrides the `config_file` from the Processor for the final import script. |
| `model`      | `--model`                      | **Optional**. The technical name of the Odoo model (e.g., `sale.order`). If you omit this, the tool infers it from the filename. |
| `context`    | `--context`                    | An Odoo context dictionary string. Essential for disabling mail threads, etc. (e.g., `"{'tracking_disable': True}"`) |
| `worker`     | `--worker`                     | The number of parallel processes to use for the import.                                                           |
| `size`       | `--size`                       | The number of records to process in a single Odoo transaction.                                                    |
| `ignore`     | `--ignore`                     | A comma-separated string of fields to completely exclude from the import process.                                 |
| `skip`       | `--skip`                       | The number of initial lines to skip in the source file before reading the header.                                 |                            |

## Generating the Script with `write_to_file()`

After calling `process()`, you can generate the final shell script that will be used in the load phase.

```python
processor.write_to_file("load_my_data.sh")
```

This method takes a single argument: the path where the `load.sh` script should be saved. It automatically uses the `filename_out` and `params` you provided to the `process()` method to construct the correct commands.

## Full Example for a Data Migration

Here is a complete `transform.py` script that ties everything together.

```{code-block} python
:caption: transform.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper
from files import * # Imports source_config_file and destination_config_file

# 1. Define the mapping rules
sales_order_mapping = {
    'id': mapper.m2o_map('import_so_', 'OrderRef'),
    'partner_id/id': mapper.m2o_map('main_customers_', 'CustomerCode'),
    'name': mapper.val('OrderRef'),
    # ... other fields
}

# 2. Define the parameters for the load script
#    Note that we specify the destination config file here.
import_params = {
    'model': 'sale.order',
    'config': destination_config_file, # <-- USES DESTINATION CONFIG
    'context': "{'tracking_disable': True, 'mail_notrack': True}",
    'worker': 4,
    'size': 500
}

# 3. Initialize the processor using the SOURCE config file
processor = Processor(
    filename=src_sales_orders.csv,
    config_file=source_config_file, # <-- USES SOURCE CONFIG
    separator=','
)

# 4. Run the transformation
processor.process(
    mapping=sales_order_mapping,
    filename_out='data/sale_order.csv',
    params=import_params
)

# 5. Generate the final script. This will now use the destination config.
processor.write_to_file("load_sales_orders.sh")

print("Transformation complete.")
