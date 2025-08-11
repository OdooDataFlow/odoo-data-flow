# Guide: The File-Based Migration Workflow

This guide details the recommended best-practice for a robust, repeatable, and debuggable data migration using `odoo-data-flow`. This workflow is centered around a three-step, file-based process:

1.  **Export:** Extract data from the source Odoo instance into a local CSV file.
2.  **Transform:** Apply data cleaning, mapping, and transformation logic to the CSV file, producing a new, clean CSV file ready for import.
3.  **Import:** Load the transformed CSV file into the destination Odoo instance.

This approach provides maximum visibility and control. You can inspect the intermediate files at each stage, manually correct data if needed, and re-run steps independently, which is invaluable for complex migrations.

---

## The Scenario

Let's imagine we need to migrate `res.partner` records from an old Odoo 14 instance (the "source") to a new Odoo 17 instance (the "destination").

Our requirements are:
- All partners from the "Customers" category in the source should be migrated.
- The `name` of each partner should be prefixed with `[MIGRATED]`.
- A new field `migration_source_id` on the destination `res.partner` model should be populated with the original ID from the source database.
- We only want to migrate a few key fields: `id`, `name`, `email`, and `phone`.

## Step 1: Export the Source Data

First, we use the `odf export` command to pull the data from our source database. We'll create two configuration files, one for the source and one for the destination.

**`source.conf`**:
```ini
[Connection]
hostname = source.odoo.com
database = source_db
login = admin
password = xxx
```

**`destination.conf`**:
```ini
[Connection]
hostname = destination.odoo.com
database = destination_db
login = admin
password = yyy
```

Now, run the export command:

```bash
odf export --config source.conf \
           --model res.partner \
           --fields "id,name,email,phone" \
           --domain "[('category_id.name', '=', 'Customers')]" \
           --output source_partners.csv \
           --technical-names
```

**Key Parameters:**
- `--fields "id,name,email,phone"`: We specify exactly which fields we need.
- `--domain "[('category_id.name', '=', 'Customers')]" `: We filter to only get the partners we care about.
- `--output source_partners.csv`: The data is saved to a local CSV file.
- `--technical-names`: **This is critical for migrations.** It ensures we get raw database values (like database IDs for relations) instead of human-readable names, which makes the data stable and reliable for processing.

After this step, you will have a `source_partners.csv` file on your machine.

## Step 2: Transform the Data

This is where the power of `odoo-data-flow` shines. We will create a Python script to define our transformations.

Create a file named `partner_mapper.py`:

```python
import polars as pl
from odoo_data_flow.lib.transform import Processor

# Define the transformation logic using Polars expressions.
mapping = {
    # 1. Rename 'id' from the source to our new custom field.
    "migration_source_id": pl.col("id"),

    # 2. Add a prefix to the partner's name.
    "name": "[MIGRATED] " + pl.col("name"),

    # 3. Keep email and phone as they are (1-to-1 mapping).
    "email": pl.col("email"),
    "phone": pl.col("phone"),
}

# Create the Processor instance. This is what odf will use.
# The `dataframe` will be automatically injected by the `transform` command.
processor = Processor(mapping=mapping)
```

Now, run the `transform` command, pointing it to our source data and our new mapper file:

```bash
odf transform --file-in source_partners.csv \
              --file-out partners_transformed.csv \
              --transformer-file partner_mapper.py
```

This command reads `source_partners.csv`, applies the logic from `partner_mapper.py`, and saves the result in `partners_transformed.csv`. You can now open this file to verify that the transformations were applied correctly before attempting the import.

## Step 3: Import the Transformed Data

Finally, we import the clean data into our destination database using the `odf import` command.

```bash
odf import --config destination.conf \
           --file partners_transformed.csv \
           --model res.partner
```

**Key Points:**
- We use our `destination.conf` to connect to the target database.
- The tool automatically matches the columns in `partners_transformed.csv` to the fields on the `res.partner` model.
- By default, the import uses the `id` column from the CSV as the unique external ID. Since our transformed file doesn't have an `id` column, the tool will let Odoo generate new IDs, which is exactly what we want for a migration. The original ID is safely stored in our `migration_source_id` field.

## Conclusion

This file-based workflow provides a clear, robust, and auditable path for your data. By separating the export, transform, and import steps, you gain full control over the migration process, making it easier to handle the complexities of real-world data projects.
