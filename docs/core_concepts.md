# Core Concepts

The `odoo-data-flow` library is built on a few key concepts that enable robust and manageable data migrations. Understanding these will help you get the most out of the tool for both importing and exporting data.

## The Two-Phase Import Workflow

For importing data, the library promotes a two-phase workflow to separate data manipulation from the actual loading process.

1. **Transform Phase**: This phase focuses purely on data manipulation. A Python script reads your raw source files, applies cleaning and transformation rules using **mappers**, and produces clean, structured CSV files that are perfectly formatted for Odoo.


```{mermaid}
---
config:
  theme: redux
---
flowchart TD
    A(["Odoo-Data-Flow"]) -- Processor
    Mapper --- B["Transform Python Script"]
    B --- C["Client CSV File"]
    B --> D["Transformed CSV Files for import"]
    A@{ shape: proc}
    C@{ shape: doc}
    D@{ shape: docs}
    style A fill:#BBDEFB
    style B fill:#C8E6C9
    style C fill:#FFF9C4
    style D fill:#FFF9C4
```

2. **Load Phase**: This phase focuses purely on data import. The generated shell script or the direct `odoo-data-flow import` command takes the clean CSV files and loads them into Odoo.


```{mermaid}
---
config:
  theme: redux
---
flowchart TD
    A["Odoo-Data-Flow"] -- Import --- B["odoo-client lib"]
    B --- C["Transformed CSV Files"]
    B L_B_D_0@--> D["odoo"]
    n1["Configuration File"] --> B
    A@{ shape: proc}
    B@{ shape: proc}
    C@{ shape: docs}
    D@{ shape: cyl}
    n1@{ shape: doc}
    style A fill:#BBDEFB
    style B fill:#FFE0B2
    style C fill:#FFF9C4
    style D fill:#AA00FF
    style n1 fill:#C8E6C9
    L_B_D_0@{ animation: slow }
```

This separation provides several key advantages:

* **Debugging**: If there's a problem, you can easily tell if it's a data transformation issue or an Odoo connection issue.
* **Reusability**: You can run the time-consuming transformation once and then use the resulting clean data to load into multiple Odoo instances (e.g., testing, staging, and production).
* **Simplicity**: Each script has a single, clear responsibility.

## The Import Strategy: One File, One Model

It is important to understand that the `odoo-data-flow import` command is designed to load **one data file into one specific Odoo model** at a time. This means that a complete data migration (e.g., for partners, products, and sales orders) will require you to run the transform and load process several times with different data files and different target models.

This deliberate design ensures clarity and respects Odoo's internal logic. Data is not inserted directly into the database; instead, it is loaded by calling Odoo's standard `load` method. This ensures that all the business logic, validations, and automations associated with each model are triggered correctly, just as they would be in the Odoo user interface.

## Automation: Actions and Workflows

In addition to transforming and loading data, the library provides a powerful system for running automated actions directly in Odoo. These are separated into two distinct categories:

* **Module Management (`module` command):** For administrative tasks that prepare the Odoo environment, such as installing or uninstalling modules. These are typically run *before* a data import.
* **Data Workflows (`workflow` command):** For running multi-step processes on records *after* they have been imported, such as validating a batch of invoices.

### Overall Data Flow Including Automation

This diagram shows how the different automation commands fit into a complete deployment process.

This diagram shows how the workflow phase fits in after the main transform and load phases.

```{mermaid}
---
config:
  theme: redux
---
flowchart TD
 subgraph subGraph0["1 Environment Setup"]
    direction LR
        B[("Odoo Database")]
        A{"odoo-data-flow module<br>update-list / install"}
  end
 subgraph subGraph1["2 Data Migration"]
    direction LR
        D{"Transform Script"}
        C["Raw Source Data"]
        E["Cleaned / Transformed Data"]
        F{"odoo-data-flow import"}
  end
 subgraph subGraph2["3 Post-Import Workflow"]
    direction LR
        G{"odoo-data-flow workflow<br>(e.g., invoice-v9)"}
  end
    A L_A_B_0@--> B
    C --> D
    D --> E
    E --> F
    F L_F_B_0@--> B
    B --> G
    G -- "Validate, Pay, etc." --> B
    C@{ shape: docs}
    E@{ shape: docs}
    style B fill:#AA00FF
    style A fill:#BBDEFB
    style C fill:#FFF9C4
    style E fill:#FFF9C4
    style F fill:#BBDEFB
    style G fill:#BBDEFB
    style subGraph2 fill:transparent
    style subGraph0 fill:transparent
    style subGraph1 fill:transparent
    L_A_B_0@{ animation: slow }
    L_F_B_0@{ animation: slow }

```

## Core Components of the Transform Phase

The transformation is driven by three main components in your Python script:

### 1. The `Processor`

The `Processor` is the engine of the library. You initialize it with your source file path and its settings (like the separator). Its main job is to apply your mapping rules and generate the clean data and the load script.

### 2. The `mapper` Functions

Mappers are the individual building blocks for your transformations. They are simple, reusable functions that define *how* to create the value for a single column in your destination file. The library provides a rich set of mappers for concatenation, direct value mapping, static values, and handling complex relationships.

> For a complete list of all available mappers and their options, see the [Data Transformations Guide](guides/data_transformations.md).


### 3. The Mapping Dictionary

This standard Python `dict` ties everything together. The keys are the column names for your **destination** CSV file, and the values are the `mapper` functions that will generate the data for that column.

## Understanding the Load Phase and Error Handling

A key strength of this library is its robust error handling, which ensures that a few bad records won't cause an entire import to fail. This is managed through a clever two-pass system.

## The Smart Import Engine: Performance, Robustness, and Caching

A key strength of this library is its "smart" import engine, which is designed to maximize both speed and success rates. It automatically handles common issues like individual bad records, complex data relationships, and redundant data lookups without requiring manual intervention.

This is managed through several core automatic strategies:

### Tier 1: The `load` -> `create` Fallback for Robustness

A common problem with batch imports is that a single invalid record can cause the entire batch of hundreds or thousands of records to be rejected. The smart importer solves this with a two-tier fallback system.

1.  **Attempt `load`:** The importer first attempts to import each batch using Odoo's highly performant, multi-record `load` method.
2.  **Fallback to `create`:** If the `load` method fails for a batch, the importer does **not** immediately write the entire batch to a fail file. Instead, it automatically retries each record within that single failed batch one-by-one using the slower but more precise `create` method.

This powerful feature "rescues" all the good records from a failed batch, ensuring they are imported successfully. Only the specific record that also fails the `create` call is written to the `_fail.csv` file, along with a detailed error message.

This gives you the best of both worlds: the speed of `load` for clean batches and the pinpoint error accuracy of `create` for problematic ones.

#### Error Handling Flow Diagram

This diagram visualizes how a single batch flows through the smart import engine.
(with res.partner model as an example.)
```{mermaid}
---
config:
  theme: redux
---
flowchart TD
    A["res_partner.csv<br>(100 records)"] --> B{"First Pass<br>odoo-data-flow import"}
    B -- 95 successful records --> C["Odoo Database"]
    B -- 5 failed records --> D["res_partner_fail.csv<br>(5 records)"]
    D --> E{"Second Pass<br>odoo-data-flow import --fail"}
    E -- 3 recovered records --> C
    E -- 2 true errors --> F["fa:fa-user-edit res_partner_YYMMDD_failed.csv<br>(2 records to fix)"]

    A@{ shape: doc}
    C@{ shape: cyl}
    D@{ shape: doc}
    F@{ shape: doc}
    style A fill:#FFF9C4
    style B fill:#BBDEFB
    style C fill:#AA00FF
    style D fill:#FFD600
    style E fill:#BBDEFB
    style F fill:#FF6D00

```

### Tier 2: Automatic Strategy Selection for Relational Fields

Importing data with interdependent relationships (like a `parent_id` on a partner that refers to another partner in the same file) is a complex challenge. The smart importer tackles this by automatically detecting relational fields and choosing the most performant import strategy.

During the **pre-flight check**, the importer inspects the header of your CSV file and queries Odoo for the type of each field. Based on the field types it discovers, it automatically selects one of the following strategies:

- **Standard Load**: For files with no relational fields, it uses the robust `load` -> `create` fallback method for maximum speed.
- **Two-Pass Strategy (for `Many2one`)**: When it detects `Many2one` fields that create dependencies within the same file (e.g., `parent_id`), it automatically deactivates those columns for the first pass. It creates all the base records and then, in a second, multi-threaded pass, it efficiently `write`s the relational values.
- **Three-Pass Strategy (for `Many2many` and `One2many`)**: For the most complex cases involving `Many2many` or `One2many` fields, it performs a three-pass import. This ensures that all records exist before attempting to create the complex relationships, maximizing the success rate.

This automatic behavior means you no longer need to manually use `--ignore` or split your files to handle these common scenarios. The tool selects the best strategy for you.

### Tier 3: Internal Caching for Speed

To further accelerate the import process, `odoo-data-flow` implements an internal caching mechanism. It automatically caches metadata fetched from Odoo, such as:

- Field definitions for models
- `ir.model.data` records (external IDs)

This cache is stored in the `.odf_cache` directory in your project root. On subsequent runs, the importer uses the cached data instead of repeatedly querying the Odoo server, leading to a significant speed-up, especially in large projects with many files or complex models. The cache is intelligently invalidated when the corresponding model in Odoo changes.


## The Export Concept

The library can also be used to export data from Odoo, which is useful for backups, analysis, or migrating data between systems. The export process is a direct command-line call.

### Export Flow Diagram

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

> For detailed instructions, see the [Exporting Data Guide](guides/exporting_data.md).
