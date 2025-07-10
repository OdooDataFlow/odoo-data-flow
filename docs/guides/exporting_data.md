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

The export process is handled by the `export` sub-command of the main `odoo-data-flow` tool. Unlike the import workflow, exporting is a single-step operation where you execute one command with the right parameters to pull data from your Odoo database.

### Command-Line Options

The command is configured using a set of options. Here are the most essential ones:

| Option     | Description                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------------------- |
| `--config` | **Required**. Path to your `connection.conf` file containing the Odoo credentials.                        |
| `--model`  | **Required**. The technical name of the Odoo model you want to export records from (e.g., `res.partner`). |
| `--fields` | **Required**. A comma-separated list of the technical field names you want to include in the export file. |
| `--file`   | **Required**. The path and filename for the output CSV file (e.g., `data/exported_partners.csv`).         |
| `--domain` | A filter to select which records to export, written as a string. Defaults to `[]` (export all records).   |
| `--worker` | The number of parallel processes to use for the export. Defaults to `1`.                                  |
| `--size`   | The number of records to fetch in a single batch. Defaults to `10`.                                       |
| `--sep`    | The character separating columns. Defaults to a semicolon (`;`).                                          |
| `--technical-names` | A flag that, when present, exports the raw technical values for `Selection` fields (e.g., `delivery`) instead of the human-readable labels (e.g., `Shipping Address`). This is highly recommended for data migrations. |


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
    --file "data/belgian_contacts.csv"
```

### Result

This command will:

1.  Connect to the Odoo instance defined in `conf/connection.conf`.
2.  Search the `res.partner` model for records that are not companies and have their country set to Belgium.
3.  For each matching record, it will retrieve the `name`, `email`, `city`, and the `name` of the related country.
4.  It will save this data into a new CSV file located at `data/belgian_contacts.csv`.
