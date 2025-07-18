# Guide: Server-to-Server Migration

The `odoo-data-flow` library includes a powerful `migrate` command designed to perform a direct, in-memory data migration from one Odoo database to another. This is an advanced feature that chains together the export, transform, and import processes into a single step, without needing to create intermediate CSV files on your local machine.

```{mermaid}
---
config:
  theme: redux
---
flowchart LR
 subgraph subGraph0["Source Environment"]
        A[("Source Odoo DB")]
  end
 subgraph subGraph1["Destination Environment"]
        F[("Destination Odoo DB")]
  end
 subgraph subGraph2["Migration Process (In-Memory)"]
        B["odoo-data-flow migrate"]
        C{"Exporter"}
        D{"Processor & Mappers"}
        E{"Importer"}
  end
    B -- 1 Connect & Export --> A
    A -- 2 Data Stream --> C
    C -- Raw Data (Header & Rows) --> D
    D -- Transformed Data --> E
    E -- 3 Load Data --> F
    style A fill:#AA00FF
    style F fill:#AA00FF
    style B fill:#BBDEFB,stroke:#1976D2
    style C fill:#FFE0B2
    style D fill:#FFCC80
    style E fill:#FFE0B2
    style subGraph2 fill:transparent
    style subGraph0 fill:transparent
    style subGraph1 fill:transparent

```

## Use Case

This command is ideal for scenarios such as:

- Migrating data from a staging or development server to a production server.
- Consolidating data from one Odoo instance into another.
- Performing a data transformation and re-importing into the same database.

## The `odoo-data-flow migrate` Command

The migration is handled by the `migrate` sub-command. It works by exporting data from a source instance, applying an in-memory transformation using the same `mapper` functions, and then immediately importing the result into a destination instance.

### Command-Line Options

The command is configured using a set of options that combine parameters from both the `export` and `import` commands.

| Option                | Description                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--config-export`     | **Required**. Path to the `connection.conf` file for the **source** Odoo instance (where data is exported from).    |
| `--config-import`     | **Required**. Path to the `connection.conf` file for the **destination** Odoo instance (where data is imported to). |
| `--model`             | **Required**. The technical name of the Odoo model you want to migrate (e.g., `res.partner`).                       |
| `--fields`            | **Required**. A comma-separated list of the technical field names you want to migrate.                              |
| `--domain`            | An Odoo domain filter to select which records to export from the source instance. Defaults to `[]` (all records).   |
| `--mapping`           | A dictionary string defining the transformation rules. If omitted, a direct 1-to-1 mapping is used.                 |
| `--export-worker`     | The number of parallel workers to use for the export phase. Defaults to `1`.                                        |
| `--export-batch-size` | The batch size for the export phase. Defaults to `100`.                                                             |
| `--import-worker`     | The number of parallel workers to use for the import phase. Defaults to `1`.                                        |
| `--import-batch-size` | The batch size for the import phase. Defaults to `10`.                                                              |

> **Note on Data Integrity:** The migration process includes several features to ensure data is handled correctly between different Odoo versions and configurations.
>
> * **Technical Values for Selection Fields:** The migration automatically exports the raw **technical values** for `Selection` fields (e.g., `delivery`) instead of the human-readable labels (e.g., `Shipping Address`). This is a deliberate design choice to ensure that the migration is robust and not dependent on the languages installed in the source or destination databases.
> * **Type-Aware Cleaning of Empty Fields:** The export process intelligently handles empty fields. It inspects the field types on your source model and corrects common data inconsistencies, such as converting Odoo's `False` values to empty strings for non-boolean fields (like `phone` or `website`), while preserving `False` for actual boolean fields. This prevents incorrect data from being imported into your destination database.

## Full Migration Example

Let's say we want to migrate all partners from a staging server to a production server. We also want to add a prefix to their names during the migration to indicate they came from the staging environment.

**Step 1: Create two connection files**

You would have two configuration files: `conf/staging.conf` and `conf/production.conf`.

**Step 2: Define the mapping (optional)**

If you need to transform the data, you can define a mapping. For this example, we'll pass it as a string on the command line.
The mapping would look like this in Python:

```python
my_mapping = {
    'id': mapper.concat('migrated_partner_', 'id'),
    'name': mapper.concat('Staging - ', 'name'),
    'phone': mapper.val('phone'),
    # ... other fields
}
```

As a command-line string, it would be: `"{'id': mapper.concat('migrated_partner_', 'id'), 'name': mapper.concat('Staging - ', 'name'), ...}"`

**Step 3: Run the `migrate` command**

You would run the following command from your terminal:

```bash
odoo-data-flow migrate \
    --config-export "conf/staging.conf" \
    --config-import "conf/production.conf" \
    --model "res.partner" \
    --fields "id,name,phone" \
    --mapping "{'name': mapper.concat('Staging - ', 'name'), 'phone': mapper.val('phone')}"
```

### Result

This single command will:

1.  Connect to the staging Odoo database.
2.  Export the `id`, `name`, and `phone` fields for all `res.partner` records.
3.  In memory, transform the data by prepending "Staging - " to each partner's name.
4.  Connect to the production Odoo database.
5.  Import the transformed data, creating new partners with the updated names.
