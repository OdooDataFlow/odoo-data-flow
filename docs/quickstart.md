# Quickstart
## A Real-World Import Workflow

This guide demonstrates a realistic and robust workflow for importing data. Instead of a single script that does everything, we will separate the process into two distinct phases, which is highly recommended for any serious data migration:

1.  **Transform Phase**: A Python script reads a raw source file, cleans the data using the library's powerful **mappers**, and produces a clean CSV file ready for Odoo. It also generates a shell script for the next phase.
2.  **Load Phase**: The generated shell script uses the new `odoo-data-flow` command-line tool to efficiently load the clean CSV data into Odoo.

This separation makes the process more manageable, easier to debug, and allows you to reuse the transformed data for multiple Odoo instances (e.g., staging and production).

## Step 1: Project Setup

First, create the recommended directory structure for a data flow project.

```
.
├── conf/
│   └── connection.conf
├── origin/
│   └── clients.csv
├── data/
│   └── (this will be created by our script)
└── transform.py
```

- `conf/`: Holds configuration files, like Odoo connection details.
- `origin/`: Contains the original, raw data files from the source system.
- `data/`: Will store the transformed, clean CSV files ready for import.
- `transform.py`: Our main Python script for the transformation logic.

## Step 2: Connection Configuration (`connection.conf`)

Create the `conf/connection.conf` file. The section header `[Connection]` and the keys (`database`, `login`) must match this example, as they are used by the import client.


```{code-block} ini
:caption: conf/connection.conf

[Connection]
hostname = my-odoo-instance.odoo.com
database = my_odoo_db
login = admin
password = <your_odoo_password>
protocol = jsonrpcs
port = 443
uid = 2
```


## Step 3: The Raw Data (`origin/clients.csv`)

Create a raw data file in `origin/clients.csv`.

```{code-block} text
:caption: origin/clients.csv
ID,Firstname,Lastname,EmailAddress
C001,John,Doe,john.doe@test.com
C002,Jane,Smith,jane.s@test.com
```

## Step 4: The Transformation Script (`transform.py`)

This script is the core of our logic. It uses the `Processor` to read the source file and a `mapping` dictionary to define the transformations.

Create the file `transform.py`:

```{code-block} python
:caption: transform.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# 1. Define the mapping rules in a dictionary.
res_partner_mapping = {
    'id': mapper.concat('example_client_', 'ID'),
    'name': mapper.concat(' ', 'Firstname', 'Lastname'),
    'email': mapper.val('EmailAddress'),
    'is_company': mapper.const(False),
}

# 2. Initialize the Processor.
processor = Processor(
    'origin/clients.csv',
    separator=','
)

# 3. Define parameters for the import client.
params = {
    'model': 'res.partner',
    'context': "{'tracking_disable': True}"
}

# 4. Run the process.
processor.process(
    mapping=res_partner_mapping,
    filename_out='data/res_partner.csv',
    params=params
)

# 5. Generate the shell script for the loading phase.
processor.write_to_file("load.sh")

print("Transformation complete. Clean data and load script are ready.")
```

## Step 5: Run the Transformation

Execute the script from your terminal:

```bash
python transform.py
```

## Step 6: Review the Generated Files

Let's look at what was created.

**File: `data/res_partner.csv` (Transformed & Clean Data)**

```{code-block} text
:caption: data/res_partner.csv
id,name,email,is_company
example_client_C001,"John Doe",john.doe@test.com,False
example_client_C002,"Jane Smith",jane.s@test.com,False
```

**File: `load.sh` (The Loading Script)**
This file now contains commands that use the new, clean `odoo-data-flow` command-line interface.

```{code-block} bash
:caption: load.sh
#!/bin/bash
odoo-data-flow import --config conf/connection.conf --file data/res_partner.csv --model res.partner --context "{'tracking_disable': True}"
odoo-data-flow import --config conf/connection.conf --fail --file data/res_partner.csv --model res.partner --context "{'tracking_disable': True}"
```

## Step 7: Load the Data into Odoo

Finally, execute the generated shell script to upload the data.

```bash
bash load.sh
```

The `odoo-data-flow` tool will connect to your database and import the records. Log in to your Odoo instance and navigate to the **Contacts** app to see your newly imported contacts.

Congratulations! You have successfully completed a full transform and load workflow with the new `odoo-data-flow` tool.
