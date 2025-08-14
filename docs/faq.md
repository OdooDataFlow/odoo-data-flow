# FAQ & Troubleshooting

This document answers frequently asked questions and provides solutions to common problems you may encounter while using `odoo-data-flow`.

## Frequently Asked Questions

### What is `odoo-data-flow`?

It is a powerful Python library designed to handle the import and export of data to and from Odoo. It allows you to define complex data transformations in Python, providing a robust and repeatable process for data migrations.

### How is this different from Odoo's standard import tool?

While Odoo's built-in import is great for simple tasks, `odoo-data-flow` offers several key advantages for complex or large-scale migrations:

- **Separation of Concerns**: It cleanly separates the data **transformation** logic (cleaning your source data) from the data

**loading** logic (importing into Odoo).

- **Robust Error Handling**: Its `load` -> `create` fallback mechanism provides the speed of batch imports while automatically switching to single-record imports for failed batches to provide precise, row-level error messages. The smart, automatic two-pass strategy seamlessly handles complex relational data.

- **Powerful Transformations**: You can use the full power of Python and a rich set of built-in `mapper` functions to handle almost any data transformation challenge.

- **Repeatability and Version Control**: Since your transformation logic is code, it can be version-controlled (with Git), tested, and reused across multiple environments (like staging and production) with confidence.

### Can I use this for both importing and exporting?

Yes. The library provides tools for both workflows. The `Processor` and `mapper` modules are used for transforming and preparing data for import, while the `odoo-data-flow export` command is used to export data from Odoo into CSV files.

### Can I migrate data directly between two Odoo databases?

Yes. The library includes a powerful `odoo-data-flow migrate` command that performs a complete export, transform, and import from one Odoo instance to another in a single step, without creating intermediate files. This is ideal for migrating data from a staging server to production.

> For detailed instructions, see the [Server-to-Server Migration Guide](guides/server_to_server_migration.md).

### How do I process a CSV file that has no header?

The `Processor` can be initialized directly with in-memory data. If your source file has no header, you can read it manually using Python's standard `csv` module and provide your own header list.

1.  Read the raw data from the CSV file into a list of lists.
2.  Create a Python list containing the header names in the correct order.
3.  Initialize the `Processor` using the `header=` and `data=` arguments instead of `filename=`.

```python
import csv
from odoo_data_flow.lib.transform import Processor

# 1. Define the header manually
my_header = ['LegacyID', 'FirstName', 'LastName', 'Email']
my_data = []

# 2. Read the file into a list
with open('origin/contacts_no_header.csv', 'r') as f:
    reader = csv.reader(f)
    my_data = list(reader)

# 3. Initialize the Processor with the in-memory data
processor = Processor(header=my_header, data=my_data)

# You can now proceed with your mapping as usual
# my_mapping = {'name': mapper.concat(' ', 'FirstName', 'LastName'), ...}
```

### Where can I find a complete, real-world example?

A full example project, demonstrating a realistic data migration workflow with multiple models and complex transformations, is available on GitHub. This is an excellent resource for seeing how all the pieces fit together.

- **[Odoo Data Flow Example Repository](https://github.com/OdooDataFlow/odoo-data-flow-example/tree/18.0)**

### Can `odoo-data-flow` connect directly to Google Sheets?

No, the `odoo-data-flow` library cannot connect directly to Google Sheets to read data.

The tool is designed to read data from local files on your computer, specifically in either **CSV** or **XML** format. It does not have the built-in functionality to authenticate with Google's services and pull data directly from a spreadsheet URL.

#### Recommended Workflow

The standard and easiest way to use your data from Google Sheets is to first download the sheet as a CSV file and then use that local file with the tool.

1.  Open your spreadsheet in Google Sheets.
2.  From the top menu, select **File** -> **Download**.
3.  Choose the **Comma-separated values (.csv)** option.


4.  This will save the current sheet as a `.csv` file to your computer's "Downloads" folder.
5.  You can then use that downloaded file with the `odoo-data-flow` command:

    ```bash
    odoo-data-flow import --file /path/to/your/downloaded-sheet.csv
    ```

This workflow ensures that you have a local copy of the data at the time of import and allows you to use all the powerful transformation features of the library on your spreadsheet data.


## I can't connect to my cloud-hosted Odoo instance (e.g., Odoo.sh). What should I do?

This is a common issue. When connecting to a cloud-hosted Odoo instance, you often need to use a secure connection protocol.

The solution is typically to set the `protocol` in your `conf/connection.conf` file to **`jsonrpcs`** (note the `s` at the end for "secure").

While Odoo's external API has historically used XML-RPC, modern cloud instances often require the secure JSON-RPC protocol for integrations.

### Example Configuration for a Cloud Instance

Your `conf/connection.conf` should look something like this:

```{code-block} ini
:caption: conf/connection.conf
[Connection]
hostname = my-project.odoo.com
port = 443
database = my-project-production
login = admin
password = xxxxxxxxxx
uid = 2
protocol = jsonrpcs
```

### Key things to check:

1.  **Protocol**: Ensure it is set to `jsonrpcs`.
2.  **Port**: Secure connections almost always use port `443`.
3.  **Hostname & Database**: Make sure you are using the correct hostname and database name provided by your cloud hosting platform (e.g., from your Odoo.sh dashboard). These are often different from the simple names used for local instances.


---

## Troubleshooting Common Errors

When an import fails, understanding why is key. Here are some of the most common issues and how to solve them.

### Understanding the Failure Files

The library uses a powerful two-tier failure handling system to maximize success and provide clear feedback.

#### Tier 1: The `load` -> `create` Fallback (During a Normal Import)

When you run a normal import, the tool first tries to import each batch of records using Odoo's fast, multi-record `load` method.

- **If `load` fails for a batch**: Instead of writing the entire batch to a fail file, the importer automatically **falls back** to a slower, more robust `create` method for each record *within that failed batch*.
- **This "rescues" the good records**: Any records that succeed during the `create` retry are imported successfully.
- **`_fail.csv` is created**: Only the single, specific record that failed the `create` call is written to the `<model_name>_fail.csv` file.  (e.g., `res_partner_fail.csv`)**:

  * **When it's created**: During the **first pass** (a normal import).

  * **What it contains**: If a batch of records fails to import, this file will contain the *entire original, unmodified batch* that failed.

  * **Purpose**: This file is for **automated processing**. It's the input for the second pass (`--fail` mode).

#### Tier 2: The `--fail` Mode

When you run the import again with the `--fail` flag, it enters a dedicated recovery mode:

-   It reads the highly-focused `<model_name>_fail.csv` file.
-   It uses a safe, single-threaded, "create-only" mode to retry each record.
-   Any records that *still* fail are written to a final, timestamped `..._failed.csv` file for your manual review.


* `<original_filename>_YYYYMMDD_HHMMSS_failed.csv` (e.g., **`data_20250626_095500_failed.csv`)**:

  * **When it's created**: During the **second pass** (when you run with the `--fail` flag).

  * **What it contains**: This file contains only the individual records that *still* failed during the record-by-record retry. Crucially, it includes an extra **`_ERROR_REASON`** column explaining exactly why each record failed.

  * **Purpose**: This file is for **human review**. The error messages help you find and fix the specific data problems.

**Your recommended workflow should be:**

1. Run your `load.sh` script or the `odoo-data-flow import` command.

2. If a `<model_name>_fail.csv` file is created, run the command again with the `--fail` flag.

3. If a timestamped `..._failed.csv` file is created, open it to identify the data issues using the `_ERROR_REASON` column.

4. Fix the issues in your original source file or your `transform.py` script.

5. Delete the `_fail.csv` and `_failed.csv` files and rerun the entire process from the beginning.


### Record Count Mismatch

Sometimes, the number of records in your source file doesn't match the number of records created in Odoo, even if there are no errors in the final failure file.

* **Cause:** This usually happens when your mapping logic unintentionally filters out rows. For example, using a `postprocess` function that can return an empty value for an external ID (`id` field). If the external ID is empty, the entire record is skipped without error.

* **Solution:**

  1. **Check your `id` field**: The most common culprit is the mapping for the `id` field. Ensure it *always* returns a non-empty, unique value for every row you intend to import.

  2. **Use a `preprocessor`**: For complex debugging, you can use a [preprocessor function](guides/data_transformations.md) to add a unique line number to each row. Import this line number into a custom field in Odoo (`x_studio_import_line_number`). After the import, you can easily compare the line numbers in your source file with those in Odoo to find exactly which rows were skipped.


### Connection Errors

These errors usually happen when the `odoo-data-flow` client cannot reach your Odoo instance.

- **Error:** `Connection refused`
  - **Cause:** The `hostname` or `port` in your `conf/connection.conf` is incorrect, or the Odoo server is not running.
  - **Solution:** Double-check your connection details and ensure the Odoo instance is active and accessible.

- **Error:** `Wrong login/password`
  - **Cause:** The credentials in `conf/connection.conf` are incorrect.
  - **Solution:** Verify your `database`, `login`, and `password`.

### Odoo Access & Validation Errors

These errors come directly from Odoo when the data is not valid enough to save.

- **Error:** `AccessError`, `You are not allowed to modify this document`
  - **Cause:** The user specified by `uid` in your `conf/connection.conf` lacks the necessary permissions (e.g., Create or Write access) for the target model.
  - **Solution:** Check the user's Access Rights in Odoo's settings.

- **Error:** `ValidationError: A required field was not provided`
  - **Cause:** Your transformed CSV file is missing a column for a field marked as `required=True` on the Odoo model.
  - **Solution:** Check the model's definition in Odoo and ensure your `transform.py` script generates a value for that field.

- **Error:** `No matching record found for external id '__export__.my_external_id_123'`
  - **Cause:** You are trying to link to a related record (e.g., setting the `partner_id` on a sales order), but the external ID you are providing does not exist in the database.
  - **Solution:**
    1. Ensure you have successfully imported the parent records first.
    2. Check for typos. The prefix and value used in your `m2o_map` must exactly match the external ID of the parent record.
    3. See the section below on Import Order.

### Understanding Import Order for Relational Data

A very common reason for the `No matching record found` error is that you are trying to import records in the wrong order.

* **The Rule:** You must always import "parent" records **before** you import the "child" records that refer to them.

* **Example:** Imagine you are importing Contacts (`res.partner`) and assigning them to Contact Tags (`res.partner.category`). Odoo cannot assign a contact to the "VIP" tag if that "VIP" tag doesn't exist in the database yet.

* **Correct Import Sequence**:

  1. **First, import `res.partner.category`**: Run a transformation and load process for your contact tags. This creates the tags and their external IDs in Odoo.

  2. **Then, import `res.partner`**: Run a separate process for your contacts. The mapping for the `category_id/id` field can now successfully use `mapper.m2o_map` to look up the external IDs of the tags you created in the first step.

However, for self-referential models (like importing partners with a `parent_id` that refers to another partner in the **same file**), the tool's new **automatic two-pass import strategy** handles this for you. It will automatically create all records in the first pass and then link the relationships in a second pass, removing the need for manual sorting or multiple import runs for a single file.

## Why is one of my exported columns completely empty?

his can happen for two main reasons:

  1.  The field is genuinely empty for all records in your exported dataset.

  2.  There might be a typo in the field name you provided in the --fields argument.

To check for the second case, look at the console output when you run the export command. If the field name is invalid, odoo-data-flow will show a warning like this:

`WARNING  Field 'your_field_name' (base: 'your_field_name') not found on model 'res.partner'. An empty column will be created.`
If you see this warning, correct the field name in your command and run the export again.
