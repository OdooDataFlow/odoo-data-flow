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

### The Two-Pass Load Sequence

The generated `load.sh` script contains two commands designed to maximize both speed and accuracy.

```bash
# First pass (Normal Mode): Fast, parallel import. Writes recoverable errors to a .fail file.
odoo-data-flow import --config conf/connection.conf --file data/res_partner.csv --model res.partner
# Second pass: Slower, precise import of the failed records.
odoo-data-flow import --config conf/connection.conf --fail --file data/res_partner.csv --model res.partner
```

1. **First Pass (Normal Mode)**: The command runs in its default, high-speed mode, importing records in batches. If an entire batch is rejected for any reason, the original records from that batch are written to an intermediate failure file named **`<model_name>.fail.csv`** (e.g., `res_partner_fail.csv`).

2. **Second Pass (`--fail` Mode)**: The command is invoked again with the `--fail` flag. In this mode, it automatically targets the `_fail.csv` file and retries each failed record individually. Records that still fail are written to a final, timestamped error file: **`<original_filename>_YYYYMMDD_HHMMSS_failed.csv`**. This file includes an additional **`_ERROR_REASON`** column to explain why each record failed, making it easy to identify and fix the problematic data manually.


### Error Handling Flow Diagram

This diagram visualizes how records flow through the two-pass system.
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
