# Guide: Creating a Migration Mapping with a Field Diff

One of the most time-consuming tasks in an Odoo-to-Odoo migration is identifying which fields have been added, removed, or renamed between your source and destination databases.

This guide provides a powerful, systematic workflow to quickly generate a complete list of all field differences, allowing you to build an accurate transformation mapping with confidence.

---

## The Goal

The objective is to produce two CSV files: one listing all the fields for a specific model from your **source** database, and another listing all the fields for the same model from your **destination** database. By comparing these two files, you can instantly see every change.

## The Workflow

The entire process is done using the `odoo-data-flow export` command, which we will use to query Odoo's internal data dictionary.

```{mermaid}
---
config:
  theme: redux
---
flowchart TD
 subgraph subGraph0["Source Database"]
        A[("Source Odoo DB")]
  end
 subgraph subGraph1["Destination Database"]
        B[("Destination Odoo DB")]
  end
 subgraph Analysis["Analysis"]
        G["Diff Tool<br>(e.g., VS Code)"]
        E["source_fields.csv"]
        F["destination_fields.csv"]
  end
 subgraph subGraph3["Developer's Local Machine"]
    direction LR
        C{"odoo-data-flow export<br>--model ir.model.fields"}
        D{"odoo-data-flow export<br>--model ir.model.fields"}
        Analysis
        H["Developer / LLM<br>(fa:fa-user-edit)"]
        I["transform.py<br>(mapping dictionary)"]
  end
    C --> A & E
    D --> B & F
    E --> G
    F --> G
    G -- List of<br>Renamed/Removed<br>Fields --> H
    H -- Writes the mapping logic --> I
    style A fill:#AA00FF
    style B fill:#C8E6C9,stroke:#388E3C
    style G fill:#FFE0B2
    style E fill:#FFF9C4
    style F fill:#FFF9C4
    style C fill:#BBDEFB
    style D fill:#BBDEFB
    style I fill:#E1F5FE
    style subGraph3 fill:transparent
    L_C_E_0@{ animation: slow }
    L_D_F_0@{ animation: slow }
```

### Step 1: Export Field Definitions from the Source Database

First, run the `export` command pointed at your **source** database configuration. This command targets the `ir.model.fields` model, which is Odoo's internal dictionary of all model fields.

* `--model ir.model.fields`: We are querying the model that holds field definitions.
* `--domain "[('model', '=', 'res.partner')]"`: This is the crucial filter. It tells Odoo to only return records where the `model` field is `res.partner`.
* `--fields "name,field_description,ttype"`: We export the technical name, the user-friendly label, and the field type, which is excellent information for comparison.

```bash
odoo-data-flow export \
  --config conf/source_db.conf \
  --model ir.model.fields \
  --domain "[('model', '=', 'res.partner')]" \
  --fields "name,field_description,ttype" \
  --output source_res_partner_fields.csv
```

### Step 2: Export Field Definitions from the Destination Database

Next, run the exact same command, but change the configuration to point to your **destination** database.

```bash
odoo-data-flow export \
  --config conf/destination_db.conf \
  --model ir.model.fields \
  --domain "[('model', '=', 'res.partner')]" \
  --fields "name,field_description,ttype" \
  --output destination_res_partner_fields.csv
```

### Step 3: Compare the Files with a Diff Tool

You will now have two CSV files:
* `source_res_partner_fields.csv`
* `destination_res_partner_fields.csv`

Open these two files in a visual "diff tool." Most modern code editors have a built-in file comparison feature (e.g., VS Code). You can also use dedicated tools like `Meld`, `Beyond Compare`, or `vimdiff`.

The diff tool will give you a clear, side-by-side view of every change.

---

## From Diff to Mapping: A Practical Example

With the comparison open, building your mapping becomes a simple task of "filling in the blanks."

Let's imagine your diff tool shows the following differences:

| Source Field (`source_res_partner_fields.csv`) | Destination Field (`destination_res_partner_fields.csv`) | Analysis |
| :--- | :--- | :--- |
| `name` | `name` | No change. A direct 1-to-1 mapping. |
| `street2` | (missing) | This field was removed in the new version. |
| (missing) | `street_two` | A new field was added. It looks like `street2` was renamed. |
| `ref` | `partner_ref` | The `ref` field was renamed to `partner_ref`. |
| `customer` | (missing) | This old boolean field was replaced. |
| (missing) | `customer_rank` | A new integer field `customer_rank` was added to replace `customer`. |
| `some_legacy_field` | (missing) | This custom field from the old system is no longer needed. |

### Building the Python Mapping

Based on this analysis, you can now construct your mapping dictionary in your `transform.py` script.

```python
from odoo_data_flow.lib import mapper

partner_migration_mapping = {
    # Direct 1-to-1 mapping for unchanged fields
    'id': mapper.m2o_map('mig_partner_', 'name'),
    'name': mapper.val('name'),
    'city': mapper.val('city'), # Assuming city was unchanged

    # Handle renamed fields: map the old name to the new name
    'street_two': mapper.val('street2'),
    'partner_ref': mapper.val('ref'),

    # Handle changed logic: convert the old boolean to the new rank
    # If the old 'customer' field was '1' (True), set rank to 1, else 0.
    'customer_rank': mapper.val('customer', postprocess=lambda x: 1 if x == '1' else 0),

    # Fields to ignore: simply omit 'some_legacy_field' from the mapping.
    # It will not be included in the output file.
}
```

### Tools to Accelerate the Process

* **Diff Tools:** As mentioned, a visual diff tool is your most valuable asset in this process. It makes spotting changes effortless.

* **AI Assistants (like Gemini, ChatGPT, etc.):** You can significantly speed up the creation of the final mapping dictionary by using an AI assistant.
    1.  Copy the full content of `source_res_partner_fields.csv`.
    2.  Copy the full content of `destination_res_partner_fields.csv`.
    3.  Use a prompt like the following:

    > "I am migrating data between two Odoo databases. Below are two CSV files listing the field definitions for the `res.partner` model from the source and destination databases.
    >
    > Compare these two files and generate a Python dictionary for the `odoo-data-flow` library that maps the source fields to the destination fields.
    >
    > -   For fields that have the same name, create a direct `mapper.val()` mapping.
    > -   For fields that appear to have been renamed (e.g., based on the description), map the old name to the new one.
    > -   For fields that only exist in the source, add a Python comment indicating they have been removed.
    >
    > **Source Fields:**
    > ```text
    > [Paste content of source_res_partner_fields.csv here]
    > ```
    >
    > **Destination Fields:**
    > ```text
    > [Paste content of destination_res_partner_fields.csv here]
    > ```"

The AI can generate a nearly complete mapping dictionary for you in seconds, which you can then review and refine. This combination of automated export and AI-assisted mapping can reduce the time it takes to create a migration plan from hours to minutes.
