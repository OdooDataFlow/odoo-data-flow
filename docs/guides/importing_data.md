# Guide: A Deep Dive into Importing

This guide expands on the import workflow, providing a detailed look at the `Processor` class and, most importantly, the requirements for your input data files.

## Command-Line Usage

The primary way to import data is through the `import` command. If your configuration file is in the default location and your CSV file is named after the Odoo model, the command is very simple:

```bash
odoo-data-flow import --file path/to/res_partner.csv
```

### Key Options for `import`

* `--config`: **(Optional)** Path to your connection configuration file. **Defaults to `conf/connection.conf`**.
* `--file`: **(Required)** Path to the source CSV file you want to import.
* `--model`: **(Optional)** The target Odoo model (e.g., `res.partner`). If you omit this option, the tool will automatically infer the model name from your CSV filename. For example, a file named `res_partner.csv` will be imported into the `res.partner` model.
* `--worker`: Number of parallel threads to use for the import.
* `--fail`: Runs the import in "fail mode," retrying only the records from the corresponding `_fail.csv` file.
* `--skip`: The number of initial lines to skip in the source file before reading the header.
* `--sep`: The character separating columns. Defaults to a semicolon (`;`).

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

## Input File Requirements

For a successful import into Odoo, the clean CSV file you generate (the `filename_out` in your script) must follow some important rules.

* **Encoding**: The file must be in `UTF-8` encoding.
* **One Model per File**: Each CSV file should only contain data for a single Odoo model (e.g., all `res.partner` records).
* **Header Row**: The first line of the file must be the header row. All column names must be the technical field names from the Odoo model (e.g., `name`, `parent_id`, `list_price`).
* **External ID**: All rows must have an `id` column containing a unique External ID (also known as an XML ID). This is essential for the "upsert" logic described above.
* **Field Separator**: The character separating columns can be defined with the `--sep` command-line option. The default is a semicolon (`;`). **Crucially, if a field's value contains the separator character, the entire field value must be enclosed in double quotes (`"`).**
* **Skipping Lines**: If your source file contains introductory lines before the header, you can use the `--skip` option to ignore them during the import process.

### Special Field Naming Conventions

To handle relational data and updates by database ID, the tool uses special column headers:

* **`/id` Suffix (for External IDs)**: When mapping to a `Many2one` or `Many2many` field, you must append `/id` to the field name (e.g., `partner_id/id`). This tells Odoo to look up the related record using the provided External ID.

* **`.id` Field Name (for Database IDs)**: To update a record using its existing database ID (an integer, not an external ID), use the special field name `.id`. When you use this, you should also provide an empty `id` column to tell Odoo you are updating, not creating a new record.

### Field Formatting Rules

Odoo's `load` method expects data for certain field types to be in a specific format.

* **Boolean**: Must be `1` for True and `0` for False. The `mapper.bool_val` can help with this.
* **Binary**: Must be a base64 encoded string. The `mapper.binary` and `mapper.binary_url_map` functions handle this automatically.
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

### Configuring the Import Client with `params`

The `params` dictionary allows you to control the behavior of the import client without ever leaving your Python script. The keys in this dictionary map directly to the command-line options of the `odoo-data-flow import` command.

| `params` Key | `odoo-data-flow import` Option | Description                                                                                                       |
| ------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `model`      | `--model`                      | **Optional**. The technical name of the Odoo model (e.g., `sale.order`). If you omit this, the tool infers it from the filename. |
| `context`    | `--context`                    | An Odoo context dictionary string. Essential for disabling mail threads, etc. (e.g., `"{'tracking_disable': True}"`) |
| `worker`     | `--worker`                     | The number of parallel processes to use for the import.                                                           |
| `size`       | `--size`                       | The number of records to process in a single Odoo transaction.                                                    |
| `ignore`     | `--ignore`                     | A comma-separated string of fields to ignore during the import. Crucial for performance with related fields.       |
| `skip`       | `--skip`                       | The number of initial lines to skip in the source file before reading the header.                                 |

## Generating the Script with `write_to_file()`

After calling `process()`, you can generate the final shell script that will be used in the load phase.

```python
processor.write_to_file("load_my_data.sh")
```

This method takes a single argument: the path where the `load.sh` script should be saved. It automatically uses the `filename_out` and `params` you provided to the `process()` method to construct the correct commands.

## Full Example

Here is a complete `transform.py` script that ties everything together.

```{code-block} python
:caption: transform.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# 1. Define the mapping rules
sales_order_mapping = {
    'id': mapper.m2o_map('import_so_', 'OrderRef'),
    'partner_id/id': mapper.m2o_map('main_customers_', 'CustomerCode'),
    'name': mapper.val('OrderRef'),
    # ... other fields
}

# 2. Define the parameters for the load script
import_params = {
    'model': 'sale.order',
    'context': "{'tracking_disable': True, 'mail_notrack': True}",
    'worker': 4,
    'size': 500
}

# 3. Initialize the processor
processor = Processor('origin/sales_orders.csv', separator=',')

# 4. Run the transformation
processor.process(
    mapping=sales_order_mapping,
    filename_out='data/sale_order.csv',
    params=import_params
)

# 5. Generate the final script
processor.write_to_file("load_sales_orders.sh")

print("Transformation complete.")
