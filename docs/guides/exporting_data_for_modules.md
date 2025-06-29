# Guide: Exporting Data for Odoo Modules (CSV to XML)

A common requirement for Odoo developers is to package data (e.g., default settings, master data like countries or states) into XML files within a custom module. This ensures the data is loaded automatically when the module is installed.

While `odoo-data-flow` does not export directly to Odoo's XML format, it is the perfect tool for the first and most critical step: extracting the data from a database into a clean, reliable format.

This guide provides a standard workflow for exporting data to a CSV file and then converting that file into a properly formatted Odoo XML data file.

```{mermaid}
---
config:
  theme: redux
---
flowchart LR
 subgraph subGraph0["Step 1: Export from Odoo"]
        B{"odoo-data-flow export"}
        A[("Odoo Database")]
        C["temp_data.csv"]
  end
 subgraph subGraph1["Step 2: Convert on Local Machine"]
        D{"Python Script<br>(create_data_file.py)"}
        E["my_module/data/my_data.xml"]
  end
 subgraph subGraph2["Step 3: Update Custom Module"]
        F{"Your Custom Module"}
        G["__manifest__.py"]
  end
    A L_A_B_0@--> B
    B --> C
    C --> D
    D --> E
    E --> F
    G --> F
    style B fill:#BBDEFB
    style A fill:#AA00FF
    style C fill:#FFF9C4
    style D fill:#FFE0B2
    style E fill:#FFF9C4
    style F fill:#C8E6C9,stroke:#388E3C
    style G fill:#E1F5FE
    style subGraph0 fill:transparent
    style subGraph1 fill:transparent
    L_A_B_0@{ animation: slow }
```

## The Workflow

The process involves two simple steps:

1. **Export to CSV**: Use the `odoo-data-flow export` command to pull the data you need from your Odoo instance into a clean CSV file.
2. **Convert to XML**: Use a simple Python script to read the CSV and generate an XML file in the exact format Odoo requires for data files.

### Step 1: Export the Data to a CSV File

First, use the `export` command to get the data you want to include in your module. For this example, let's export all the US states from the `res.country.state` model.

```bash
odoo-data-flow export \
  --config conf/my_db.conf \
  --model res.country.state \
  --domain "[('country_id.code', '=', 'US')]" \
  --fields "name,code,country_id/id" \
  --file temp_us_states.csv
```

This command will create a file named `temp_us_states.csv` that looks something like this:


```{code-block} text
:caption: temp_us_states.csv
name;code;country_id/id
Alabama;AL;base.us
Alaska;AK;base.us
...
```

### Step 2: Convert the CSV to an Odoo XML Data File

Now, we can use a Python script to convert this CSV into an XML file suitable for your module's `data` directory.

The script below will:
- Read each row from the CSV.
- Create a `<record>` tag for each state.
- Automatically generate a unique XML ID for each record (e.g., `state_us_al`).
- Create `<field>` tags for each column.

```{code-block} python
:caption: create_data_file.py
import csv
from lxml import etree

CSV_FILE_PATH = "temp_us_states.csv"
XML_OUTPUT_PATH = "my_awesome_module/data/res_country_state_data.xml"
MODEL_NAME = "res.country.state"
ID_PREFIX = "state_us_"

# Create the root <odoo> tags
odoo_tag = etree.Element("odoo", noupdate="1")

# Read the CSV and create a <record> for each row
with open(CSV_FILE_PATH, "r", encoding="utf-8") as f:
    # Use DictReader to easily access columns by header name
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        # Create a unique XML ID for the record, e.g., "state_us_al"
        record_id = f"{ID_PREFIX}{row['code'].lower()}"

        # Create the <record> tag
        record_tag = etree.SubElement(
            odoo_tag, "record", id=record_id, model=MODEL_NAME
        )

        # Create a <field> tag for each column in the CSV
        for header, value in row.items():
            # Skip empty values to keep the XML clean
            if not value:
                continue

            field_tag = etree.SubElement(record_tag, "field", name=header)

            # Use a 'ref' attribute for relational fields ending in /id
            if header.endswith("/id"):
                field_tag.set("ref", value)
            else:
                field_tag.text = value

# Write the final XML to a file with pretty printing for readability
with open(XML_OUTPUT_PATH, "wb") as f:
    f.write(
        etree.tostring(
            odoo_tag, pretty_print=True, xml_declaration=True, encoding='utf-8'
        )
    )

print(f"Successfully generated Odoo XML data file at: {XML_OUTPUT_PATH}")
```

Running this script will produce the following XML file, perfectly formatted for Odoo.

```{code-block} xml
:caption: my_awesome_module/data/res_country_state_data.xml
<?xml version='1.0' encoding='utf-8'?>
<odoo noupdate="1">
    <record id="state_us_al" model="res.country.state">
      <field name="name">Alabama</field>
      <field name="code">AL</field>
      <field name="country_id/id" ref="base.us"/>
    </record>
    <record id="state_us_ak" model="res.country.state">
      <field name="name">Alaska</field>
      <field name="code">AK</field>
      <field name="country_id/id" ref="base.us"/>
    </record>
    <!-- ... and so on for all other states -->
</odoo>
```

### Step 3: Add the Data File to Your Module

The final step is to tell Odoo to load this file when your module is installed or updated.

1.  Move the generated XML file to your module's `data/` directory.
2.  Add the path to the `data` key in your module's `__manifest__.py` file.

```{code-block} python
:caption: my_awesome_module/__manifest__.py
{
    'name': 'My Awesome Module',
    'version': '1.0',
    # ... other manifest keys
    'data': [
        'security/ir.model.access.csv',
        'data/res_country_state_data.xml', # Add this line
        'views/my_views.xml',
    ],
    'installable': True,
}
```

Now, when you install or upgrade your module, Odoo will automatically load all the US states from your XML file. This workflow combines the power of `odoo-data-flow` for data extraction with a simple, reusable script for generating module data.
