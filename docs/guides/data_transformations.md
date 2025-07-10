# Guide: Data Transformations with Mappers

Mappers are the core of the data transformation process. They are powerful, reusable functions that you use within your mapping dictionary to define how each column of your destination file should be generated.

This guide provides a comprehensive reference for all mappers available in the `odoo_data_flow.lib.mapper` module.

---

## Data Quality Validation (`Processor.check`)

Before you start the main transformation process, it's often a good idea to validate the quality and structure of your source data. The library provides a `.check()` method on the `Processor` object for this purpose.

You can call `.check()` multiple times with different "checker" functions to validate your data against a set of rules. If a check fails, a warning will be logged to the console, and you can prevent the transformation from continuing.

### Using Checkers

In your `transform.py` script, after initializing the `Processor` but before calling `.process()`, you can add your checks:

```{code-block} python
:caption: transform.py
from odoo_data_flow.lib import checker
from odoo_data_flow.lib.transform import Processor

# Initialize processor
processor = Processor('origin/my_data.csv')

# --- Add Data Quality Checks ---
print("Running data quality checks...")
processor.check(checker.line_length_checker(15))
processor.check(checker.cell_len_checker(120))
processor.check(checker.id_validity_checker('SKU', r'^[A-Z]{2}-\d{4}$'))

# Now, proceed with the mapping and processing
# processor.process(...)
```

### Available Checker Functions

The following checkers are available in the `odoo_data_flow.lib.checker` module.

#### `checker.line_length_checker(expected_length)`

Verifies that every row in your data file has exactly the `expected_length` number of columns. This is useful for catching malformed CSV rows.

#### `checker.cell_len_checker(max_cell_len)`

Verifies that no single cell (field) in your entire dataset exceeds the `max_cell_len` number of characters.

#### `checker.line_number_checker(expected_line_count)`

Verifies that the file contains exactly `expected_line_count` number of data rows (not including the header).

#### `checker.id_validity_checker(id_field, pattern)`

Verifies that the value in the specified `id_field` column for every row matches the given regex `pattern`. This is extremely useful for ensuring key fields like SKUs or external IDs follow a consistent format.

---

## Basic Mappers

### `mapper.val(field, [postprocess])`

Retrieves the value from a single source column, identified by `field`. This is the most fundamental mapper.

- **`field` (str)**: The name of the column in the source file.
- **`postprocess` (function, optional)**: A function to modify the value after it has been read.

### `mapper.const(value)`

Fills a column with a fixed, constant `value` for every row.

- **`value`**: The static value to use (e.g., string, bool, integer).

#### How it works

**Input Data (`source.csv`)**
| AnyColumn |
| --------- |
| a         |
| b         |

**Transformation Code**

```python
'sale_type': mapper.const('service')
```

**Output Data**
| sale_type |
| --------- |
| service   |
| service   |

---

## Combining and Formatting

### `mapper.concat(separator, *fields)`

Joins values from one or more source columns together, separated by a given `separator`.

- **`separator` (str)**: The string to place between each value.
- **`*fields` (str)**: A variable number of source column names (`field`) or static strings to join.

---

## Conditional and Boolean Logic

### `mapper.cond(field, true_value, false_value)`

Checks the value of the source column `field`. If it's considered "truthy" (not empty, not "False", not 0), it returns `true_value`, otherwise it returns `false_value`.

### `mapper.bool_val(field, true_values=None, false_values=None, default=False)`

Checks the value in the source column `field` and converts it to a boolean `1` or `0`.

- **`field` (str)**: The column to check.
- **`true_values` (list, optional)**: A list of strings that should be considered `True`.
- **`false_values` (list, optional)**: A list of strings that should be considered `False`.
- **`default` (bool, optional)**: The default boolean value to return if the value is not in `true_values` or `false_values`.

#### How it works

**Input Data (`source.csv`)**
| Status        |
| ------------- |
| Active        |
| Inactive      |
| Done          |

**Transformation Code**

```python
'is_active': mapper.bool_val('Status', true_values=['Active', 'In Progress'], false_values=['Inactive']),
```

**Output Data**
| is_active |
| --------- |
| 1         |
| 0         |
| 0         |

---

## Numeric Mappers

### `mapper.num(field, default='0.0')`

Takes the numeric value of the source column `field`. It automatically transforms a comma decimal separator (`,`) into a dot (`.`). Use it for `Integer` or `Float` fields in Odoo.

- **`field` (str)**: The column containing the numeric string.
- **`default` (str, optional)**: A default value to use if the source value is empty. Defaults to `'0.0'`.

#### How it works

**Input Data (`source.csv`)**
| my_column |
| --------- |
| 01        |
| 2,3       |
|           |

**Transformation Code**

```python
'my_field': mapper.num('my_column'),
'my_field_with_default': mapper.num('my_column', default='-1.0')
```

**Output Data**
| my_field | my_field_with_default |
| -------- | --------------------- |
| 1        | 1                     |
| 2.3      | 2.3                   |
| 0.0      | -1.0                  |

---

## Relational Mappers

### `mapper.m2o_map(prefix, *fields)`

A specialized `concat` for creating external IDs for **Many2one** relationship fields (e.g., `partner_id`). This is useful when the unique ID for a record is spread across multiple columns.

---
## Many-to-Many Relationships

Handling many-to-many relationships often requires a two-step process:
1.  **Extract and Create Related Records**: First, you need to identify all the unique values for the related records (e.g., all unique "Tags" or "Categories"), create a separate CSV file for them, and assign each one a unique external ID.
2.  **Link to Main Records**: In the main record file (e.g., partners), you create a comma-separated list of the external IDs of the related records.

The library provides special mappers and a processing flag (`m2m=True`) to make this easy.

### Example: Importing Partners with Categories

Let's assume you have a source file where partner categories are listed in a single column, separated by commas.

**Input Data (`client_file.csv`)**
| Company             | Firstname | Lastname | Birthdate  | Category            |
| ------------------- | --------- | -------- | ---------- | ------------------- |
| The World Company   | John      | Doe      | 31/12/1980 | Premium             |
| The Famous Company  | David     | Smith    | 28/02/1985 | Normal, Bad Payer   |

#### Step 1: Extract and Create Unique Categories

We need to create a `res.partner.category.csv` file. The key is to use `mapper.m2m_id_list` and `mapper.m2m_value_list` combined with the `m2m=True` flag in the `.process()` method. This tells the processor to automatically find all unique values in the 'Category' column, split them, and create one row for each.

**Transformation Code**
```python
# This mapping is specifically for extracting unique categories.
partner_category_mapping = {
   'id': mapper.m2m_id_list('res_partner_category', 'Category'),
   'name':  mapper.m2m_value_list('Category'),
}

# The m2m=True flag activates the special processing mode.
processor.process(partner_category_mapping, 'res.partner.category.csv', m2m=True)
```

**Output File (`res.partner.category.csv`)**
This file will contain one row for each unique category found across all partner records.
| id                            | name       |
| ----------------------------- | ---------- |
| res_partner_category.Premium  | Premium    |
| res_partner_category.Normal   | Normal     |
| res_partner_category.Bad_Payer| Bad Payer  |

#### Step 2: Create the Partner File with M2M Links

Now that the categories have their own external IDs, you can create the partner records and link them using the `mapper.m2m` function. This mapper will create the required comma-separated list of external IDs for Odoo.

**Transformation Code**
```python
res_partner_mapping = {
    'id': mapper.m2o_map('my_import_res_partner', 'Firstname', 'Lastname', 'Birthdate'),
    'name': mapper.concat(' ', 'Firstname', 'Lastname'),
    'parent_id/id': mapper.m2o_map('my_import_res_partner', 'Company'),
    # Use mapper.m2m to create the comma-separated list of external IDs
    'category_id/id': mapper.m2m('res_partner_category', 'Category', sep=','),
}

processor.process(res_partner_mapping, 'res.partner.csv')
```

**Output File (`res.partner.csv`)**
| id                                     | parent_id/id                             | name        | category_id/id                                                  |
| -------------------------------------- | ---------------------------------------- | ----------- | --------------------------------------------------------------- |
| my_import_res_partner.John_Doe_31/12/1980 | my_import_res_partner.The_World_Company | John Doe    | res_partner_category.Premium                                    |
| my_import_res_partner.David_Smith_28/02/1985| my_import_res_partner.The_Famous_Company| David Smith | res_partner_category.Normal,res_partner_category.Bad_Payer |

---

## Importing Product Variants: Legacy (v9-v12) vs. Modern (v13+)

Importing product variants (e.g., a T-shirt that comes in different colors and sizes) is a common but complex task. The data model for product attributes changed significantly in Odoo 13. The library provides two distinct workflows to handle both the old and new systems.

### Modern Approach (Odoo v13+)

This is the recommended approach for all modern Odoo versions. Odoo can now automatically create product variants if you provide the attribute values directly.

* **Processor Class**: `ProductProcessorV10`
* **Key Mapper**: `mapper.m2m_template_attribute_value`

**How it Works:**
You provide the attribute values (e.g., "Blue", "L") as a comma-separated string. The `ProductProcessorV10` sets `create_variant: 'Dynamically'` on the attribute, telling Odoo to find or create the corresponding `product.attribute.value` records and link them to the product template automatically.

#### Example Transformation Script (Modern)
```python
# In your transform.py
from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import ProductProcessorV10

# Initialize the modern processor
processor = ProductProcessorV10('origin/products.csv')

# --- 1. Create the product.attribute records ---
# This step tells Odoo which attributes can create variants
attributes = ['Color', 'Size']
processor.process_attribute_data(
    attributes, 'prod_attrib', 'data/product.attribute.csv', {}
)

# --- 2. Create the product.template records ---
# The key is to map the raw values to the attribute's technical name
template_mapping = {
    'id': mapper.m2o_map('prod_template_', 'template_id'),
    'name': mapper.val('Product Name'),
    'attribute_line_ids/Color/value_ids': mapper.val('Color'),
    'attribute_line_ids/Size/value_ids': mapper.val('Size'),
}
processor.process(template_mapping, 'data/product.template.csv')
```

### Legacy Approach (Odoo v9-v12)

This approach is for older Odoo versions and requires a more manual, three-file process. You must create the attributes, then the attribute values with their own external IDs, and finally link them to the product template.

* **Processor Class**: `ProductProcessorV9`
* **Key Mappers**: `mapper.m2m_attribute_value`, `mapper.val_att`, `mapper.m2o_att`

#### Example Transformation Script (Legacy)
```python
# In your transform.py
from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import ProductProcessorV9

# Initialize the legacy processor
processor = ProductProcessorV9('origin/products.csv')

# --- This single call creates all three required files ---
attributes = ['Color', 'Size']
attribute_prefix = 'prod_attrib'

# Mapping for product.attribute.value file
value_mapping = {
    'id': mapper.m2m_attribute_value(attribute_prefix, *attributes),
    'name': mapper.val_att(attributes),
    'attribute_id/id': mapper.m2o_att_name(attribute_prefix, attributes),
}

# Mapping for product.template.attribute.line file
line_mapping = {
    'product_tmpl_id/id': mapper.m2o_map('prod_template_', 'template_id'),
    'attribute_id/id': mapper.m2o_att_name(attribute_prefix, attributes),
    'value_ids/id': mapper.m2o_att(attribute_prefix, attributes),
}

processor.process_attribute_mapping(
    value_mapping, line_mapping, attributes, attribute_prefix, 'data/', {}
)
```

---

## Advanced Mapping

### `mapper.map_val(map_dict, key, default=None, m2m=False)`

Looks up a `key` in a `map_dict` and returns the corresponding value. This is extremely useful for translating values from a source system to Odoo values.

- **`map_dict` (dict)**: The Python dictionary to use as a translation table.
- **`key` (mapper)**: A mapper that provides the key to look up in the dictionary (often `mapper.val`).
- **`default` (optional)**: A default value to return if the key is not found.
- **`m2m` (bool, optional)**: If set to `True`, the `key` is expected to be a list of values. The mapper will look up each value in the list and return a comma-separated string of the results.

#### Example: Advanced Country Mapping

**Transformation Code**
```python
# The mapping dictionary translates source codes to Odoo external IDs.
country_map = {
    'BE': 'base.be',
    'FR': 'base.fr',
    'NL': 'base.nl',
}

# Use map_val to look up the code and return the external ID.
'country_id/id': mapper.map_val(country_map, mapper.val('CountryCode'))
```

---

## Binary Mappers

### `mapper.binary(field)`

Reads a local file path from the source column `field` and converts the file content into a base64-encoded string.

- **`field` (str)**: The name of the column that contains the relative path to the image file.

#### How it works

**Input Data (`images.csv`)**
| ImagePath             |
| --------------------- |
| images/product_a.png  |

**Transformation Code**
```python
# Reads the file at the path and encodes it for Odoo
'image_1920': mapper.binary('ImagePath')
```

**Output Data**
| image_1920                         |
| ---------------------------------- |
| iVBORw0KGgoAAAANSUhEUg... (etc.) |

### `mapper.binary_url_map(field)`

Reads a URL from the source column `field`, downloads the content from that URL, and converts it into a base64-encoded string.

- **`field` (str)**: The name of the column that contains the full URL to the image or file.

#### How it works

**Input Data (`image_urls.csv`)**
| ImageURL                               |
| -------------------------------------- |
| https://www.example.com/logo.png       |

**Transformation Code**
```python
# Downloads the image from the URL and encodes it
'image_1920': mapper.binary_url_map('ImageURL')
```

**Output Data**
| image_1920                         |
| ---------------------------------- |
| iVBORw0KGgoAAAANSUhEUg... (etc.) |

---

## Advanced Techniques

(pre-processing-data)=
### Pre-processing Data

For complex manipulations before the mapping starts, you can pass a `preprocess` function to the `Processor` constructor. This function receives the CSV header and data and must return them after modification.

#### Adding Columns

```python
def my_preprocessor(header, data):
    header.append('NEW_COLUMN')
    for i, j in enumerate(data):
        data[i].append('NEW_VALUE')
    return header, data

processor = Processor('source.csv', preprocess=my_preprocessor)
```

#### Removing Lines

```python
def my_preprocessor(header, data):
    data_new = []
    for i, j in enumerate(data):
        line = dict(zip(header, j))
        if line['Firstname'] != 'John':
            data_new.append(j)
    return header, data_new

processor = Processor('source.csv', preprocess=my_preprocessor)
```

### Sharing Data Between Mappers (The `state` Dictionary)

For complex, stateful transformations, every mapper function receives a `state` dictionary as its second argument. This dictionary is persistent and shared across the entire processing of a file, allowing you to "remember" values from one row to the next.

This is essential for handling hierarchical data, like sales orders and their lines.

#### Example: Remembering the Current Order ID

```python
def get_order_id(val, state):
    # When we see a new Order ID, save it to the state
    if val:
        state['current_order_id'] = val
    return val

sales_order_mapping = {
    'id': mapper.val('OrderID', postprocess=get_order_id),
    'order_line/product_id/id': mapper.m2o_map('prod_', 'SKU'),
    'order_line/order_id/id': lambda line, state: state.get('current_order_id')
}
```

### Conditionally Skipping Rows (`_filter`)

You can filter out rows from your source data by adding a special `_filter` key to your mapping dictionary. The mapper for this key should return `True` for any row that you want to **skip**.

**Input Data (`source.csv`)**
| Name  | Status    |
| ----- | --------- |
| John  | Active    |
| Jane  | Cancelled |
|       |           |

**Transformation Code**
```python
my_mapping = {
    '_filter': mapper.val('Status', postprocess=lambda x: x == 'Cancelled' or not x),
    'name': mapper.val('Name'),
    # ... other fields
}
```
In this example, the rows for "Jane" and the blank line will be skipped, and only the row for "John" will be processed.

### Creating Custom Mappers

Any Python function can act as a custom mapper when used with `postprocess`. The function will receive the value from the source column as its first argument and the shared `state` dictionary as its second.

### Updating Records With Database IDs

To update records using their database ID, map your source ID to the special `.id` field and provide an empty `id` field.

```python
my_mapping = {
    'id': mapper.const(''),
    '.id': mapper.val('id_column_from_source'),
    'name': mapper.val('name_from_source'),
    # ... other fields to update
}
```

### Creating Related Records (`mapper.record`)

This special mapper takes a full mapping dictionary to create related records (e.g., sales order lines) during the transformation of a main record.

#### Example: Importing Sales Orders and their Lines

**Input Data (`orders.csv`)**
| OrderID | Warehouse | SKU      | Qty |
| ------- | --------- | -------- | --- |
| SO001   | MAIN      |          |     |
|         |           | PROD_A   | 2   |
|         |           | PROD_B   | 5   |

**Transformation Code**
```python
from odoo_data_flow.lib import mapper

def get_order_id(val, state):
    if val:
        state['current_order_id'] = val
        return val
    return None

def remember_value(key):
    def postprocess(val, state):
        if val:
            state[key] = val
        return val
    return postprocess

order_line_mapping = {
    'order_id/id': lambda line, state: state.get('current_order_id'),
    'product_id/id': mapper.m2o_map('prod_', 'SKU'),
    'product_uom_qty': mapper.num('Qty'),
    'warehouse_id/id': lambda line, state: state.get('current_warehouse_id')
}

sales_order_mapping = {
    'id': mapper.val('OrderID', postprocess=get_order_id),
    'name': mapper.val('OrderID'),
    'warehouse_id/id': mapper.m2o_map('wh_', 'Warehouse', postprocess=remember_value('current_warehouse_id')),
    'order_line': mapper.cond('SKU', mapper.record(order_line_mapping))
}
