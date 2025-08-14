# Guide: Direct Server-to-Server Migration

The `odoo-data-flow` library includes a powerful `migrate` command designed to perform a **direct, in-memory** data migration from one Odoo database to another. This is an advanced feature that chains together the export, transform, and import processes into a single step, without creating intermediate CSV files.

> **When to use this?** This method is fast and convenient for simple, one-shot migrations where you don't need to inspect or modify the data mid-process.
>
> For more complex, robust, and auditable migrations, we strongly recommend the **[File-Based Migration Workflow](./file_based_migration_workflow.md)**, which gives you much greater control and visibility.

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

- Simple data transfers from a staging server to a production server.
- Consolidating data from one Odoo instance into another where the data structure is identical.

## The `odf migrate` Command

The migration is handled by the `migrate` sub-command. It works by exporting data from a source instance, applying an in-memory transformation, and then immediately importing the result into a destination instance.

### Command-Line Options

The command is configured using a set of options that combine parameters from both the `export` and `import` commands.

| Option                | Description                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--config-export`     | **Required**. Path to the `connection.conf` file for the **source** Odoo instance (where data is exported from).    |
| `--config-import`     | **Required**. Path to the `connection.conf` file for the **destination** Odoo instance (where data is imported to). |
| `--model`             | **Required**. The technical name of the Odoo model you want to migrate (e.g., `res.partner`).                       |
| `--transformer-file`  | Optional. Path to a Python file containing a `Processor` instance named `processor` for data transformations.       |
| `--fields`            | A comma-separated list of the technical field names you want to migrate. If omitted, all fields will be exported.   |
| `--domain`            | An Odoo domain filter to select which records to export from the source instance. Defaults to `[]` (all records).   |
| `--export-worker`     | The number of parallel workers to use for the export phase. Defaults to `1`.                                        |
| `--export-batch-size` | The batch size for the export phase. Defaults to `100`.                                                             |
| `--import-worker`     | The number of parallel workers to use for the import phase. Defaults to `1`.                                        |
| `--import-batch-size` | The batch size for the import phase. Defaults to `10`.                                                              |

> **Note on Data Integrity:** The migration process automatically uses the equivalent of the `--technical-names` flag during export. This is a deliberate design choice to ensure that the migration is robust and not dependent on the languages installed in the source or destination databases.

## Full Migration Example

Let's say we want to migrate all partners from a staging server to a production server. We also want to add a prefix to their names during the migration.

**Step 1: Create two connection files**

You would have two configuration files: `conf/staging.conf` and `conf/production.conf`.

**Step 2: Create a Transformer File (Optional)**

If you need to transform the data, create a transformer file (e.g., `partner_migrator.py`):

```python
# partner_migrator.py
import polars as pl
from odoo_data_flow.lib.transform import Processor

processor = Processor(
    mapping={
        "name": "[STAGING] " + pl.col("name"),
        # other fields will be mapped 1-to-1 automatically
    }
)
```

**Step 3: Run the `migrate` command**

You would run the following command from your terminal:

```bash
odf migrate \
    --config-export "conf/staging.conf" \
    --config-import "conf/production.conf" \
    --model "res.partner" \
    --fields "id,name,phone" \
    --transformer-file "partner_migrator.py"
```

If you don't provide a `--transformer-file`, the data will be migrated with a direct 1-to-1 field mapping.

### Result

This single command will:

1.  Connect to the staging Odoo database.
2.  Export the `id`, `name`, and `phone` fields for all `res.partner` records.
3.  In memory, apply the transformations from `partner_migrator.py`.
4.  Connect to the production Odoo database.
5.  Import the transformed data.
