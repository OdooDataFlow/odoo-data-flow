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

```text
Odoo Instance ---> (odoo-data-flow export) ---> Output File
      ^                      ^
      |                      |
(Database)      (Configuration / CLI Options)
```


The export process is handled by the `export` sub-command. It's a single-step operation designed for performance, reliability, and intelligent data handling.

### Smart Export Mode: Automatic Method Selection

The exporter features a **smart mode** that automatically uses the best Odoo API method for the data you request. This ensures you get the most accurate and performant export without manual configuration.

The tool will automatically use the high-performance **`read` method** if any of the following are true:

1.  You request a raw database ID using special syntax (e.g., `.id` or `country_id/.id`).
2.  You request a **`selection`** field (to get the raw key, e.g., `done`, instead of the label "Done").
3.  You request a **`binary`** field (as `export_data` cannot handle these).
4.  You manually force it with the `--technical-names` flag.

Otherwise, it defaults to the human-readable `export_data` method.


### High-Performance, Streaming Exports

The `export` command is built for scalability. To handle massive datasets, it uses a streaming pipeline:

* **Low Memory Usage:** Records are fetched in batches, processed, and written directly to the output file, ensuring a low memory footprint even for huge exports.
* **Type-Aware Cleaning:** The tool inspects Odoo field types to correct common data inconsistencies, like converting `False` values to empty strings for non-boolean fields.
* **Automatic Batch Resizing:** If the Odoo server runs out of memory on a large batch, the tool automatically splits the batch and retries, making the export process highly resilient.
* **Record Count Validation**: After a successful export, the tool automatically verifies that the number of rows in the output CSV file matches the number of records found in Odoo, providing an extra layer of data integrity.

### Command-Line Options

| Option | Description |
| :--- | :--- |
| `--config` | **Required**. Path to your `connection.conf` file. |
| `--model` | **Required**. The technical name of the Odoo model to export (e.g., `res.partner`). |
| `--fields` | **Required**. A comma-separated list of fields to export, with support for special ID specifiers. |
| `--output` | **Required**. The path and filename for the output CSV file. |
| `--domain` | A filter to select which records to export, using Odoo's domain syntax as a string. Defaults to `[]` (all records). |
| `--worker` | The number of parallel processes to use. Defaults to `1`. |
| `--size` | The number of records to fetch in a single batch. Defaults to `1000`. |
| `--sep` | The character separating columns. Defaults to a semicolon (`;`). |
| `--technical-names` | A flag to force the use of the high-performance raw export mode. Often enabled automatically. |
| `--streaming` | A flag to enable streaming mode for very large datasets. Slower but uses minimal memory. |
| `--resume-session` | The ID of a failed export session to resume. The tool will append records to the existing output file. |


### Resuming Failed Exports

When exporting extremely large datasets, network outages or server restarts can interrupt the process. Starting over from the beginning is inefficient. To solve this, `odoo-data-flow` includes a session-based resume feature.

**How It Works**

1.  **Session ID Generation**: Every time a new export is started, a unique **Session ID** is generated based on the export parameters (model, domain, and fields). This ID is logged to the console.
2.  **State Tracking**: The tool creates a session directory inside `.odf_cache/sessions/`. It stores two files:
    *   `all_ids.json`: A complete list of all record IDs that match the export domain.
    *   `completed_ids.txt`: A list of record IDs that have been successfully exported and written to the CSV file. This file is updated after each batch.
3.  **Resuming**: If the export fails, you can restart it using the `--resume-session <session_id>` flag. The tool will:
    *   Read the two state files.
    *   Calculate the set of remaining IDs that still need to be exported.
    *   Continue the export process, fetching only the missing records and appending them to the output CSV file without a header.
4.  **Automatic Cleanup**: Upon a fully successful export, the corresponding session directory is automatically deleted to prevent clutter. If the job fails, the directory is kept, making it available for you to resume.

**Example Usage**

First, start a large export:

```bash
odoo-data-flow export \
    --config conf/connection.conf \
    --model "account.move.line" \
    --fields "id,name,move_id/.id,account_id/.id,debit,credit" \
    --output "data/all_journal_entries.csv"
```

The console will log the session ID:
`INFO - Starting new export session: a1b2c3d4e5f6a7b8`

If the process fails midway, you can find the session ID in the logs or in the final error message. To resume, simply add the `--resume-session` flag:

```bash
odoo-data-flow export \
    --config conf/connection.conf \
    --model "account.move.line" \
    --fields "id,name,move_id/.id,account_id/.id,debit,credit" \
    --output "data/all_journal_entries.csv" \
    --resume-session "a1b2c3d4e5f6a7b8"
```

The tool will then calculate the remaining records and continue where it left off.

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

The `--fields` option is a simple comma-separated list of the field names you want in your output file. You can also access fields on related records using slash notation (/). The tool will log a warning if you request a field that does not exist on the Odoo model, and an empty column will be created in the output.

- Simple fields: `name,email,phone`
- Relational fields: `name,parent_id/id,parent_id/name` (This would get the contact's name, their parent company's XML ID, and their parent company's name).


It now has special syntax for handling different ID formats, making it powerful for data migration.

| Specifier | Mode Used | Resulting Value | Example |
| :--- | :--- | :--- | :--- |
| `id` | `export_data` | The record's XML ID (External ID) | `__export__.res_partner_123` |
| `.id` | `read` | The record's database ID (integer) | `123` |
| `field/id` | `export_data` | The related record's XML ID | `__export__.res_country_5` |
| `field/.id` | `read` | The related record's database ID (integer) | `5` |

The tool is smart: if you use `.id` or `field/.id`, it automatically switches to a high-performance "raw" export mode (using Odoo's `read` method). Otherwise, it defaults to a human-readable mode (using `export_data`).

## Full Export Example

Let's combine these concepts into a full example. We want to export the name, email, and city for all individual contacts (not companies) located in Belgium.

Here is the full command you would run from your terminal:

```bash
odoo-data-flow export \
    --config conf/connection.conf \
    --model "res.partner" \
    --domain "[('is_company', '=', False), ('country_id.code', '=', 'BE')]" \
    --fields "id,name,email,city,country_id/id" \
    --output "data/belgian_contacts.csv"
```

### Result

This command will:

1.  Connect to the Odoo instance defined in `conf/connection.conf`.
2.  Search the `res.partner` model for records that are not companies and have their country set to Belgium.
3.  For each matching record, it will retrieve the `name`, `email`, `city`, and the `name` of the related country.
4.  It will save this data into a new CSV file located at `data/belgian_contacts.csv`.


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
