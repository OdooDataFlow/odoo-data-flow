# Guide: Exporting Data from Odoo

In addition to importing, `odoo-data-flow` provides a powerful command-line utility for exporting data directly from Odoo into a structured CSV file. This is ideal for creating backups, feeding data into other systems, or for analysis.

```{mermaid}
---
config:
  theme: redux
---
flowchart TD
    ExportA["Odoo Instance"] L_ExportA_ExportB_0@--> ExportB{"odoo-data-flow export"}
    ExportC["Configuration<br>(CLI Options)"] --> ExportB
    ExportB L_ExportB_ExportD_0@--> ExportD["Output File<br>(e.g., exported_partners.csv)"]
    ExportA@{ shape: cyl}
    ExportD@{ shape: doc}
    style ExportA fill:#AA00FF
    style ExportB fill:#BBDEFB
    style ExportD fill:#FFF9C4
    L_ExportA_ExportB_0@{ animation: slow }
    L_ExportB_ExportD_0@{ animation: slow }
```


## The `odoo-data-flow export` Command

The export process is handled by the `export` sub-command of the main `odoo-data-flow` tool. Unlike the import workflow, exporting is a single-step operation where you execute one command with the right parameters to pull data from your Odoo database. It now includes a smart export mode to handle both human-readable and raw data exports efficiently.

## High-Performance, Streaming Exports

The `export` command is built for performance and scalability. To handle massive datasets with millions of records, the export process uses a streaming pipeline:

* **Streaming mode - Low Memory Usage:** Instead of loading the entire dataset into memory, records are fetched from Odoo in batches, processed, and written directly to the output file. This ensures that even very large exports can run on machines with limited RAM or on very large datasets.
* **Type-Aware Cleaning:** The export process automatically cleans the data before writing. It intelligently inspects the field types on your Odoo model and corrects common data inconsistencies, such as converting Odoo's `False` values to empty strings for non-boolean fields (like `phone` or `website`), while preserving `False` for actual boolean fields.
* **High-Speed Writer:** The tool uses the high-performance, multi-threaded CSV writer from the `Polars` library, making the file writing process significantly faster than standard Python libraries.

### Command-Line Options

The command is configured using a set of options. Here are the most essential ones:

| Option              | Description                                                                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--config`          | **Required**. Path to your `connection.conf` file containing the Odoo credentials.                                                                                                                       |
| `--model`           | **Required**. The technical name of the Odoo model you want to export records from (e.g., `res.partner`).                                                                                                |
| `--fields`          | **Required**. A comma-separated list of the technical field names you want to include in the export file.                                                                                                |
| `--output`            | **Required**. The path and filename for the output CSV file (e.g., `data/exported_partners.csv`).                                                                                                        |
| `--domain`          | A filter to select which records to export, written as a string. Defaults to `[]` (export all records).                                                                                                  |
| `--worker`          | The number of parallel processes to use for the export. Defaults to `1`.                                                                                                                                 |
| `--size`            | The number of records to fetch in a single batch. Defaults to `10`.                                                                                                                                      |
| `--sep`             | The character separating columns. Defaults to a semicolon (`;`).                                                                                                                                         |
| `--technical-names` | A flag that, when present, exports the raw technical values for fields (e.g., `draft` for a selection field, `False` for a boolean). This is highly recommended for data migrations and is type-aware. |
| `--streaming`       | A flag that enables the streaming mode, This mode writes data to the output file in batches, ensuring a low and constant memory footprint. It is slower due to higher I/O overhead but is necessary for huge exports. |

### Understanding the `--domain` Filter

The `--domain` option allows you to precisely select which records to export. It uses Odoo's standard domain syntax, which is a list of tuples formatted as a string.

A domain is a list of search criteria. Each criterion is a tuple `('field_name', 'operator', 'value')`.

**Simple Domain Example:**
To export only companies (not individual contacts), the domain would be `[('is_company', '=', True)]`. You would pass this to the command line as a string:

`--domain "[('is_company', '=', True)]"`

**Complex Domain Example:**
To export all companies from the United States, you would combine two criteria:

`--domain "[('is_company', '=', True), ('country_id.code', '=', 'US')]"`

### Specifying Fields with `--fields`

The `--fields` option is a simple comma-separated list of the field names you want in your output file. You can also access fields on related records using dot notation.

- Simple fields: `name,email,phone`
- Relational fields: `name,parent_id/name,parent_id/city` (This would get the contact's name, their parent company's name, and their parent company's city).

## Full Export Example

Let's combine these concepts into a full example. We want to export the name, email, and city for all individual contacts (not companies) located in Belgium.

Here is the full command you would run from your terminal:

```bash
odoo-data-flow export \
    --config conf/connection.conf \
    --model "res.partner" \
    --domain "[('is_company', '=', False), ('country_id.code', '=', 'BE')]" \
    --fields "name,email,city,country_id/name" \
    --output "data/belgian_contacts.csv"
```

### Result

This command will:

1.  Connect to the Odoo instance defined in `conf/connection.conf`.
2.  Search the `res.partner` model for records that are not companies and have their country set to Belgium.
3.  For each matching record, it will retrieve the `name`, `email`, `city`, and the `name` of the related country.
4.  It will save this data into a new CSV file located at `data/belgian_contacts.csv`.


The `--fields` argument is used to specify which columns to export. It now has special syntax for handling different ID formats, making it powerful for data migration.

| Specifier | Mode Used | Resulting Value | Example |
| :--- | :--- | :--- | :--- |
| `id` | `export_data` | The record's XML ID (External ID) | `__export__.res_partner_123` |
| `.id` | `read` | The record's database ID (integer) | `123` |
| `field/id` | `export_data` | The related record's XML ID | `__export__.res_country_5` |
| `field/.id` | `read` | The related record's database ID (integer) | `5` |

The tool is smart: if you use `.id` or `field/.id`, it automatically switches to a high-performance "raw" export mode (using Odoo's `read` method). Otherwise, it defaults to a human-readable mode (using `export_data`).

#### Forcing Raw Export Mode

You can force the high-performance raw export mode using the `--technical-names` flag. This is useful if you need the raw values of `Many2one` fields (which will return the database ID) without explicitly using the `/.id` syntax.

**Example Usage:**

```bash
# Standard export with human-readable Many2one fields
odoo-data-flow export \
  --model "res.partner" \
  --fields "name,country_id"

# Export with the raw database ID for the country
odoo-data-flow export \
  --model "res.partner" \
  --fields "name,country_id/.id"

# Force raw export mode for all fields
odoo-data-flow export \
  --model "res.partner" \
  --fields "name,country_id" \
  --technical-names
```

### Automatic Batch Resizing

When exporting very large datasets, the Odoo server can sometimes run out of memory while preparing the data, causing the export of that batch to fail.

To make the process more resilient, this tool includes an **automatic batch resizing** feature. If the export of a specific batch fails due to a server-side `MemoryError`, the tool will not quit. Instead, it will:

1.  Automatically split the failed batch in half.
2.  Retry exporting each of the new, smaller sub-batches.
3.  This process continues recursively until the batch size is small enough for the server to process successfully.

This feature makes the export much more reliable and reduces the need to perfectly tune the `--batch-size` argument. However, for best performance, starting with a reasonable batch size (e.g., 1000-5000) is still recommended to avoid the small overhead of the retry mechanism.
