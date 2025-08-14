# Guide: Advanced Usage

This guide covers more complex scenarios and advanced features of the library that can help you solve specific data transformation challenges.


## Processing XML Files

While CSV is common, you may have source data in XML format. The `Processor` can handle XML files with a couple of extra configuration arguments.

- **`xml_root_tag` (str)**: The name of the root tag in your XML document that contains the collection of records.
- **`xml_record_tag` (str)**: The name of the tag that represents a single record.

### Example XML Input (`origin/clients.xml`)

Here is an example of an XML file that the `Processor` can parse. Note the `<ClientList>` container tag and the repeating `<Client>` tags for each record. The processor can also handle nested tags like `<Contact>`.

```{code-block} xml
:caption: origin/clients.xml
<?xml version="1.0" encoding="UTF-8"?>
<ClientList>
    <Client>
        <ClientID>C1001</ClientID>
        <Name>The World Company</Name>
        <Contact>
            <Email>contact@worldco.com</Email>
            <Phone>111-222-3333</Phone>
        </Contact>
    </Client>
    <Client>
        <ClientID>C1002</ClientID>
        <Name>The Famous Company</Name>
        <Contact>
            <Email>info@famous.com</Email>
            <Phone>444-555-6666</Phone>
        </Contact>
    </Client>
</ClientList>
```

### Example Transformation Code
To process this XML, you provide an XPath expression to the xml_root_tag argument. This tells the Processor which nodes in the XML tree represent the individual records (rows) you want to process. The tags inside each record are then treated as columns.

xml_root_tag: An XPath expression to select the list of records. For the example above, './Client' tells the processor to find every <Client> tag within the document.

The Processor automatically flattens the nested structure, so you can access tags like <Email> and <Phone> directly in your mapping.


```python
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# Access nested XML tags using dot notation.
res_partner_mapping = {
    'id': mapper.m2o_map('xml_client_', 'ClientID'),
    'name': mapper.val('Name'),
    'email': mapper.val('Contact.Email'),
    'phone': mapper.val('Contact.Phone'),
}

# Initialize the Processor with XML-specific arguments
processor = Processor(
    'origin/clients.xml',
    xml_root_tag='ClientList',
    xml_record_tag='Client'
)
# ... rest of the process
```

---

## Importing Data for Multiple Companies

When working in a multi-company Odoo environment, you need a clear strategy to ensure records are created in the correct company. There are two primary methods to achieve this.

### Method 1: The Procedural Approach (Recommended)

This is the safest and most common approach. The core idea is to separate your data by company and run a distinct import process for each one.

1.  **Separate your source files:** Create one set of data files for Company A and a completely separate set for Company B.
2.  **Set the User's Company:** In Odoo, log in as the user defined in your `connection.conf`. In the user preferences, set their default company to **Company A**.
3.  **Run the Import for Company A:** Execute your transformation and load scripts for Company A's data. All records created will be assigned to Company A by default.
4.  **Change the User's Company:** Go back to Odoo and change the same user's default company to **Company B**.
5.  **Run the Import for Company B:** Execute the import process for Company B's data. These new records will now be correctly assigned to Company B.

This method is robust because it relies on Odoo's standard multi-company behavior and prevents accidental data mixing.

### Method 2: The Programmatic Approach (`company_id`)

This method is useful when your source file contains data for multiple companies mixed together. You can explicitly tell Odoo which company a record belongs to by mapping a value to the `company_id/id` field.

**Example: A source file with mixed-company products**

```text
SKU,ProductName,CompanyCode
P100,Product A,COMPANY_US
P101,Product B,COMPANY_EU
```

**Transformation Script**
Your mapping dictionary can use the `CompanyCode` to link to the correct company record in Odoo using its external ID.

```python
from odoo_data_flow.lib import mapper

product_mapping = {
    'id': mapper.m2o_map('prod_', 'SKU'),
    'name': mapper.val('ProductName'),
    # This line explicitly sets the company for each row.
    # Assumes your res.company records have external IDs like 'main_COMPANY_US'.
    'company_id/id': mapper.m2o_map('main_', 'CompanyCode'),
}
```

**Warning:** While powerful, this method requires that you have stable and correct external IDs for your `res.company` records. The procedural approach is often simpler and less error-prone.

---

## Importing Translations

The most efficient way to import translations is to perform a standard import with a special `lang` key in the context. This lets Odoo's ORM handle the translation creation process correctly.

The process involves two steps:

1.  **Import the base terms:** First, import your records with their default language values (e.g., English).
2.  **Import the translated terms:** Then, import a second file containing only the external IDs and the translated values, while setting the target language in the context.

### Example: Translating Product Names to French

**Step 1: Import the base product data in English**

**Source File (`product_template.csv`):**

```text
id;name;price
my_module.product_wallet;Wallet;10.0
my_module.product_bicyle;Bicycle;400.0
```

You would import this file normally. The `id` column provides the stable external ID for each product.

**Step 2: Import the French translations**

**Source File (`product_template_FR.csv`):**
This file only needs to contain the external ID and the fields that are being translated.

```text
id;name
my_module.product_wallet;Portefeuille
my_module.product_bicyle;Bicyclette
```

**Transformation and Load**
While you can use a `transform.py` script to generate the load script, for a simple translation update, you can also run the command directly.

**Command-line Example:**

```bash
odoo-data-flow import \
    --config conf/connection.conf \
    --file product_template_FR.csv \
    --model product.template \
    --context "{'lang': 'fr_FR'}"
```

This does not overwrite the English name; instead, it correctly creates or updates the French translation for the `name` field on the specified products.

---

## Importing Account Move Lines

Importing journal entries (`account.move`) with their debit/credit lines (`account.move.line`) is a classic advanced use case that requires creating related records using `mapper.record` and stateful processing.

### Performance Tip: Skipping Validation

For a significant performance boost when importing large, pre-validated accounting entries, you can tell Odoo to skip its balancing check (debits == credits) during the import. This is done by passing a special context key.

### Example: Importing an Invoice

**Source File: `invoices.csv`**

```text
Journal,Reference,Date,Account,Label,Debit,Credit
INV,INV2023/12/001,2023-12-31,,,
,,"Customer Invoices",600,"Customer Debtor",250.00,
,,"Customer Invoices",400100,"Product Sales",,200.00
,,"Customer Invoices",451000,"VAT Collected",,50.00
```

**Transformation Script**

```python
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# ... (see Data Transformations guide for full stateful processing example)

# Define parameters, including the crucial context key
params = {
    'model': 'account.move',
    # WARNING: Only use check_move_validity: False if you are certain
    # your source data is balanced.
    'context': "{'check_move_validity': False, 'tracking_disable': True}"
}

processor = Processor('origin/invoices.csv')
# ... rest of process
```

---

## Importing One-to-Many Relationships (`--o2m` flag)

The `--o2m` flag enables a special import mode for handling source files where child records (the "many" side) are listed directly under their parent record (the "one" side).

### Use Case and File Structure

This mode is designed for files structured like this, where a master record has lines for two different one-to-many fields (`child1_ids` and `child2_ids`):

**Source File (`master_with_children.csv`)**

```text
MasterID,MasterName,Child1_SKU,Child2_Ref
M01,Master Record 1,field_value1_of_child1,field_value1_of_child2
, , , field_value2_of_child1,field_value2_of_child2
, , , ,field_value3_of_child2
```

With the `--o2m` option, the processor understands that the lines with empty master fields belong to the last master record encountered. It will import "Master Record 1" with two `child1` records and three `child2` records simultaneously.

!!! info "When to use --o2m vs. Automatic Two-Pass"
The `--o2m` flag is specifically for the file format shown above, where child records do not have their own unique ID and are identified only by being on the lines below their parent.
For standard relational fields (like `parent_id`) where **every record in the file has its own unique ID**, you do not need this flag. The importer will automatically detect the relationship and use the two-pass strategy.



### Transformation and Load

Your mapping would use `mapper.record` and `mapper.cond` to process the child lines, similar to the `account.move.line` example. The key difference is enabling the `o2m` flag in your `params` dictionary.

```python
# In your transform.py
params = {
    'model': 'master.model',
    'o2m': True # Enable the special o2m handling
}
```

The generated `load.sh` script will then include the `--o2m` flag in the `odoo-data-flow import` command.

### Important Limitations

This method is convenient but has significant consequences because **it is impossible to set XML_IDs on the child records**. As a result:

- You **cannot run the import again to update** the child records. Any re-import will create new child records.
- The child records **cannot be referenced** by their external ID in any other import file.

This method is best suited for simple, one-off imports of transactional data where the child lines do not need to be updated or referenced later.

---

## Advanced Product Imports: Creating Variants

When you import `product.template` records along with their attributes and values, Odoo does not create the final `product.product` variants by default. You must explicitly tell Odoo to do so using a context key.

### The `create_product_product` Context Key

By setting `create_product_product: True` in the context of your `product.template` import, you trigger the Odoo mechanism that generates all possible product variants based on the attribute lines you have imported for that template.

This is typically done as the final step _after_ you have already imported the product attributes, attribute values, and linked them to the templates via attribute lines.

### Example: Triggering Variant Creation

Assume you have already run separate imports for `product.attribute`, `product.attribute.value`, and `product.attribute.line`. Now, you want to trigger the variant creation.

The easiest way is to re-import your `product.template.csv` file with the special context key.

**Transformation and Load**
In the `params` dictionary of your `product.template` transformation script, add the key:

```python
# In your transform.py for product templates

params = {
    'model': 'product.template',
    # This context key tells Odoo to generate the variants
    'context': "{'create_product_product': True, 'tracking_disable': True}"
}

# The mapping would be the same as your initial template import
template_mapping = {
    'id': mapper.m2o_map('prod_tmpl_', 'Ref'),
    'name': mapper.val('Name'),
    # ... other template fields
}
```

When you run the generated `load.sh` script for this process, Odoo will find each product template, look at its attribute lines, and create all the necessary `product.product` variants (e.g., a T-Shirt in sizes S, M, L and colors Red, Blue).

---

## Merging Data from Multiple Files (`join_file`)

Sometimes, the data you need for a single import is spread across multiple source files. The `.join_file()` method allows you to enrich your main dataset by merging columns from a second file, similar to a VLOOKUP in a spreadsheet.

### The `.join_file()` Method

You first initialize a `Processor` with your primary file. Then, you call `.join_file()` to merge data from a secondary file based on a common key.

- **`filename` (str)**: The path to the secondary file to merge in.
- **`key1` (str)**: The name of the key column in the **primary** file.
- **`key2` (str)**: The name of the key column in the **secondary** file.

### Example: Merging Customer Details into an Order File

**Transformation Script (`transform_merge.py`)**

```{code-block} python
:caption: transform_merge.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# 1. Initialize a processor with the primary file (orders)
processor = Processor('origin/orders.csv')

# 2. Join the customer details file.
print("Joining customer details into orders data...")
processor.join_file('origin/customer_details.csv', 'CustomerCode', 'Code')

# 3. Define a mapping that uses columns from BOTH files
order_mapping = {
    'id': mapper.m2o_map('import_so_', 'OrderID'),
    'name': mapper.val('OrderID'),
    'date_order': mapper.val('OrderDate'),
    # 'ContactPerson' comes from the joined file
    'x_studio_contact_person': mapper.val('ContactPerson'),
}

# The processor now contains the merged data and can be processed as usual
processor.process(
    mapping=order_mapping,
    filename_out='data/orders_with_details.csv',
    params={'model': 'sale.order'}
)
```

---

## Splitting Large Datasets for Import

When dealing with extremely large source files, processing everything in a single step can be memory-intensive and unwieldy. The library provides a `.split()` method on the `Processor` to break down a large dataset into smaller, more manageable chunks.

### The `.split()` Method

The `.split()` method divides the processor's in-memory dataset into a specified number of parts. It does not write any files itself; instead, it returns a dictionary where each key is an index and each value is a new, smaller `Processor` object containing a slice of the original data.

You can then iterate over this dictionary to process each chunk independently.

### Example: Splitting a Large File into 4 Parts

**Transformation Script (`transform_split.py`)**

```{code-block} python
:caption: transform_split.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

# 1. Define your mapping as usual
product_mapping = {
    'id': mapper.concat('large_prod_', 'SKU'),
    'name': mapper.val('ProductName'),
}

# 2. Initialize a single processor with the large source file
processor = Processor('origin/large_products.csv')

# 3. Split the processor into 4 smaller, independent processors
split_processors = processor.split(mapper.split_file_number(4))

# 4. Loop through the dictionary of new processors
for index, chunk_processor in split_processors.items():
    output_filename = f"data/products_chunk_{index}.csv"
    chunk_processor.process(
        mapping=product_mapping,
        filename_out=output_filename,
        params={'model': 'product.product'}
    )
```
