# Guide: A Deep Dive into Importing

This guide expands on the import workflow, providing a detailed look at the `Processor` class and, most importantly, the requirements for your input data files.

## Input File Requirements

For a successful import into Odoo, the clean CSV file you generate (the `target_file` in your script) must follow some important rules.

-   **Encoding**: The file must be in `UTF-8` encoding.
-   **One Model per File**: Each CSV file should only contain data for a single Odoo model (e.g., all `res.partner` records).
-   **Header Row**: The first line of the file must be the header row. All column names must be the technical field names from the Odoo model (e.g., `name`, `parent_id`, `list_price`).
-   **External ID**: All rows must have an `id` column containing a unique External ID (also known as an XML ID). This is essential for Odoo to identify records, allowing it to both create new records and update existing ones on re-import.
-   **Field Separator**: The character separating columns can be defined with the `--sep` command-line option. The default is a semicolon (`;`). **Crucially, if a field's value contains the separator character, the entire field value must be enclosed in double quotes (`"`).**
-   **Skipping Lines**: If your source file contains introductory lines before the header, you can use the `--skip` option to ignore them during the import process.


### Field Formatting Rules

Odoo's `load` method expects data for certain field types to be in a specific format.

-   **Boolean**: Must be `1` for True and `0` for False. The `mapper.bool_val` can help with this.
-   **Binary**: Must be a base64 encoded string. The `mapper.binary` and `mapper.binary_url_map` functions handle this automatically.
-   **Date & Datetime**: The format depends on the user's language settings in Odoo, but the standard, safe formats are `YYYY-MM-DD` for dates and `YYYY-MM-DD HH:MM:SS` for datetimes.
-   **Float**: The decimal separator must be a dot (`.`). The `mapper.num` function handles converting comma separators automatically.
-   **Selection**: Must contain the internal value for the selection, not the human-readable label (e.g., `'draft'` instead of `'Draft'`).
-   **Many2one**: The column header must be suffixed with `/id` (e.g., `partner_id/id`), and the value should be the external ID of the related record.
-   **Many2many**: The column header must be suffixed with `/id`, and the value should be a comma-separated list of external IDs for the related records.

### Automatic Model Detection

If you name your final CSV file using the technical name of the model (e.g., `res_partner.csv`), you do not need to specify the `--model` option when running the import command. The tool will automatically infer the model from the filename.

---

## The `Processor` Class

The `Processor` is the central component of the transform phase. It handles reading the source file, applying the mapping, and generating the output files required for the load phase.

### Initialization

You initialize the processor by providing the path to your source data file and optional formatting parameters.

```python
from odoo_data_flow.lib.transform import Processor

processor = Processor(
    'origin/my_data.csv',  # Path to the source file
    separator=';',         # The character used to separate columns
    quotechar='"'          # The character used for quoting fields
)
```

The constructor takes the following arguments:

-   **`source_file` (str)**: The path to the CSV or XML file you want to transform.
-   **`separator` (str, optional)**: The column separator for CSV files. Defaults to `;`.
-   **`quotechar` (str, optional)**: The field quote character for CSV files. Defaults to `"`.
-   **`preprocessor` (function, optional)**: A function to modify the raw data *before* mapping begins. See the [Data Transformations Guide](./03_data_transformations.md#pre-processing-data) for details.
-   **`xml_root_tag` / `xml_record_tag` (str, optional)**: Required arguments for processing XML files. See the [Advanced Usage Guide](./04_advanced_usage.md#processing-xml-files).

## The `process()` Method

This is the main method that executes the transformation. It takes your mapping dictionary and applies it to each row of the source file, writing the output to a new target file.

```python
processor.process(
    mapping=my_mapping_dict,
    target_file='data/clean_data.csv',
    params=import_params_dict
)
```

The method takes these key arguments:

-   **`mapping` (dict)**: **Required**. The mapping dictionary that defines the transformation rules for each column.
-   **`target_file` (str)**: **Required**. The path where the clean, transformed CSV file will be saved.
-   **`params` (dict, optional)**: A crucial dictionary that holds the configuration for the `odoo-data-flow import` command. These parameters will be used when generating the `load.sh` script.

### Configuring the Import Client with `params`

The `params` dictionary allows you to control the behavior of the import client without ever leaving your Python script. The keys in this dictionary map directly to the command-line options of the `odoo-data-flow import` command.

| `params` Key   | `odoo-data-flow import` Option | Description                                                                  |
| -------------- | -------------------------------- | ---------------------------------------------------------------------------- |
| `model`        | `--model`                        | **Required**. The technical name of the Odoo model (e.g., `sale.order`). |
| `context`      | `--context`                      | An Odoo context dictionary string. Essential for disabling mail threads, etc. (e.g., `"{'tracking_disable': True}"`) |
| `worker`       | `--worker`                       | The number of parallel processes to use for the import. |
| `size`         | `--size`                         | The number of records to process in a single Odoo transaction. |
| `ignore`       | `--ignore`                       | A comma-separated string of fields to ignore during the import. Crucial for performance with related fields. |
| `skip`         | `--skip`                         | The number of initial lines to skip in the source file before reading the header. |


## Generating the Script with `write_to_file()`

After calling `process()`, you can generate the final shell script that will be used in the load phase.

```python
processor.write_to_file("load_my_data.sh")
```

This method takes a single argument: the path where the `load.sh` script should be saved. It automatically uses the `target_file` and `params` you provided to the `process()` method to construct the correct commands.

## Full Example

Here is a complete `transform.py` script that ties everything together.

```python
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
    target_file='data/sale_order.csv',
    params=import_params
)

# 5. Generate the final script
processor.write_to_file("load_sales_orders.sh")

print("Transformation complete.")
