# Guide: Data Transformations with Mappers

Mappers are the core of the data transformation process. They are powerful, reusable functions that you use within your mapping dictionary to define how each column of your destination file should be generated.

This guide provides a comprehensive reference for all mappers available in the `odoo_data_flow.lib.mapper` module.

---

## Data Quality Validation (`Processor.check`)

Before you start the main transformation process, it's often a good idea to validate the quality and structure of your source data. The library provides a `.check()` method on the `Processor` object for this purpose.

You can call `.check()` multiple times with different "checker" functions to validate your data against a set of rules. If a check fails, a warning will be logged to the console, and you can prevent the transformation from continuing.

### Using Checkers

In your `transform.py` script, after initializing the `Processor` but before calling `.process()`, you can add your checks:

```python
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
| a |
| b |

**Transformation Code**

```python
'sale_type': mapper.const('service')
```

**Output Data**
| sale_type |
| --------- |
| service |
| service |

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

### `mapper.bool_val(field, true_values)`

Checks if the value in the source column `field` exists within the `true_values` list and returns a boolean.

- **`field` (str)**: The column to check.
- **`true_values` (list)**: A list of strings that should be considered `True`.

#### How it works

**Input Data (`source.csv`)**
| Status |
| ------ |
| Active |
| Done |

**Transformation Code**

```python
'is_active': mapper.bool_val('Status', ['Active', 'In Progress']),
```

**Output Data**
| is_active |
| --------- |
| True |
| False |

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
| 01 |
| 2,3 |
| |

**Transformation Code**

```python
'my_field': mapper.num('my_column'),
'my_field_with_default': mapper.num('my_column', default='-1.0')
```

**Output Data**
| my_field | my_field_with_default |
| -------- | --------------------- |
| 1 | 1 |
| 2.3 | 2.3 |
| 0.0 | -1.0 |

---

## Relational Mappers

### `mapper.m2o_map(prefix, *fields)`

A specialized `concat` for creating external IDs for **Many2one** relationship fields (e.g., `partner_id`).

### `mapper.relation(model, search_field, value, raise_if_not_found=False, skip=False)`

Finds a single record in Odoo and returns its database ID. **Note:** This can be slow as it performs a search for each row.

- **`model` (str)**: The Odoo model to search in.
- **`search_field` (str)**: The field to search on.
- **`value` (mapper)**: A mapper that provides the value to search for.
- **`raise_if_not_found` (bool, optional)**: If `True`, the process will stop if no record is found. Defaults to `False`.
- **`skip` (bool, optional)**: If `True` and the record is not found, the entire source row will be skipped. Defaults to `False`.

### Many-to-Many Mappers

These mappers create a comma-separated list of external IDs or database ID command tuples for **Many2many** fields.

#### `mapper.m2m(*args, **kwargs)`

Has two modes:

1.  **Multiple Columns**: Joins non-empty values from multiple source columns. `mapper.m2m('Tag1', 'Tag2')`
2.  **Single Column with Separator**: Splits a single column by a separator. `mapper.m2m('Tags', sep=';')`

#### `mapper.m2m_map(prefix, field, sep)`

Splits a single source column `field` by `sep` and prepends a `prefix` to each value.

#### `mapper.m2m_id_list(field, sep=',')`

Takes a source column `field` containing a list of database IDs and formats them for Odoo's `(6, 0, [IDs])` command, which replaces all existing relations with the new list.

#### `mapper.m2m_value_list(model, field, sep=',')`

Takes a source column `field` containing a list of values (e.g., names). For each value, it finds the corresponding record in the specified `model` (by searching on the `name` field) and returns a list of their database IDs, formatted as a command tuple.

#### `mapper.m2m_template_attribute_value(field, sep=',')`

A highly specialized mapper for product template attributes. It takes a list of attribute values from the source column `field`, finds or creates them (`product.attribute.value`), and returns them formatted for the `attribute_line_ids` field.

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
| ImagePath |
| --------------------- |
| images/product_a.png |

**Transformation Code**

```python
# Reads the file at the path and encodes it for Odoo
'image_1920': mapper.binary('ImagePath')
```

**Output Data**
| image_1920 |
| ---------------------------------- |
| iVBORw0KGgoAAAANSUhEUg... (etc.) |

### `mapper.binary_url_map(field)`

Reads a URL from the source column `field`, downloads the content from that URL, and converts it into a base64-encoded string.

- **`field` (str)**: The name of the column that contains the full URL to the image or file.

#### How it works

**Input Data (`image_urls.csv`)**
| ImageURL |
| -------------------------------------- |
| https://www.example.com/logo.png |

**Transformation Code**

```python
# Downloads the image from the URL and encodes it
'image_1920': mapper.binary_url_map('ImageURL')
```

**Output Data**
| image_1920 |
| ---------------------------------- |
| iVBORw0KGgoAAAANSUhEUg... (etc.) |

---

## Advanced Techniques

### Pre-processing Data

For complex manipulations before the mapping starts, you can pass a `preprocessor` function to the `Processor`. This function receives the CSV header and data and must return them after modification.

#### Adding Columns

```python
def myPreprocessor(header, data):
    header.append('NEW_COLUMN')
    for i, j in enumerate(data):
        data[i].append('NEW_VALUE')
    return header, data
```

#### Removing Lines

```python
def myPreprocessor(header, data):
    data_new = []
    for i, j in enumerate(data):
        line = dict(zip(header, j))
        if line['Firstname'] != 'John':
            data_new.append(j)
    return header, data_new
```

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
| OrderID | Warehouse | SKU | Qty |
| ------- | --------- | ------ | --- |
| SO001 | MAIN | | |
| | | PROD_A | 2 |
| | | PROD_B | 5 |

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
    'order_id/id': lambda state: state.get('current_order_id'),
    'product_id/id': mapper.m2o_map('prod_', 'SKU'),
    'product_uom_qty': mapper.num('Qty'),
    'warehouse_id/id': lambda state: state.get('current_warehouse_id')
}

sales_order_mapping = {
    # Using a postprocess on val() is a flexible way to filter
    '_filter': mapper.val('OrderID', postprocess=lambda x: not x),
    'id': mapper.val('OrderID'),
    'name': mapper.val('OrderID', postprocess=get_order_id),
    'warehouse_id/id': mapper.m2o_map('wh_', 'Warehouse', postprocess=remember_value('current_warehouse_id')),
    'order_line': mapper.cond('SKU', mapper.record(order_line_mapping))
}
```
