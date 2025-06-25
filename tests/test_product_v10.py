"""Test Odoo version 10 product import.

This test script generates a complete set of data files for importing
products with variants, including categories, attributes, and attribute lines.
This is based on the v10 product structure.
"""

import os

from odoo_data_flow.lib import mapper

# We assume ProductProcessorV10 is a custom class inheriting from Processor
from odoo_data_flow.lib.transform import ProductProcessorV10

# --- Configuration ---
TEMPLATE_PREFIX = "PRODUCT_TEMPLATE"
PRODUCT_PREFIX = "PRODUCT_PRODUCT"
CATEGORY_PREFIX = "PRODUCT_CATEGORY"
ATTRIBUTE_PREFIX = "PRODUCT_ATTRIBUTE"
ATTRIBUTE_VALUE_PREFIX = "PRODUCT_ATTRIBUTE_VALUE"
ATTRIBUTE_LINE_PREFIX = "PRODUCT_ATTRIBUTE_LINE"

# Define the attributes to be processed from the source file
attribute_list = ["Color", "Gender", "Size_H", "Size_W"]
source_file = os.path.join("tests", "origin", "product.csv")
context = {"create_product_variant": True, "tracking_disable": True}


# --- Main Logic ---
# STEP 1: Initialize the custom processor with the source file
print(f"Initializing processor for product import from: {source_file}")
processor = ProductProcessorV10(source_file, separator=",")

# STEP 2: Generate data for Parent and Child Categories
print("Generating data for product categories...")
categ_parent_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("categoy"),
}
categ_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "parent_id/id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("Sub Category"),
}
processor.process(
    categ_parent_map,
    os.path.join("data", "product.category.parent.csv"),
    {"model": "product.category", "worker": 1, "batch_size": 5},
    "set",
    m2m=True,  # Use m2m=True to get a unique set of parent categories
)
processor.process(
    categ_map,
    os.path.join("data", "product.category.csv"),
    {"model": "product.category", "worker": 1, "batch_size": 20},
    "set",
    m2m=True,  # Use m2m=True to get a unique set of child categories
)

# STEP 3: Generate data for Product Templates
print("Generating data for product templates...")
template_map = {
    "id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "categ_id/id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "standard_price": mapper.num("cost"),
    "list_price": mapper.num("public_price"),
    "default_code": mapper.val("ref"),
    "name": mapper.val("name"),
    "type": mapper.const("product"),
}
processor.process(
    template_map,
    os.path.join("data", "product.template.csv"),
    {
        "model": "product.template",
        "worker": 4,
        "batch_size": 10,
        "context": context,
    },
    m2m=True,  # A product template should only be created once per ref
)

# STEP 4: Generate data for Attributes
print("Generating data for product attributes...")
# The custom processor method handles creating a simple list of attributes
processor.process_attribute_data(
    attribute_list,
    ATTRIBUTE_PREFIX,
    os.path.join("data", "product.attribute.csv"),
    {
        "model": "product.attribute",
        "worker": 4,
        "batch_size": 10,
        "context": context,
    },
)

# STEP 5: Generate data for Attribute Values
print("Generating data for product attribute values...")
attribute_value_mapping = {
    "id": mapper.m2m_template_attribute_value(ATTRIBUTE_VALUE_PREFIX, *attribute_list),
    "name": mapper.m2m_value_list(*attribute_list),
    "attribute_id/id": mapper.m2m_id_list(
        ATTRIBUTE_PREFIX, *[mapper.field(f) for f in attribute_list]
    ),
}
processor.process(
    attribute_value_mapping,
    os.path.join("data", "product.attribute.value.csv"),
    {
        "model": "product.attribute.value",
        "worker": 3,
        "batch_size": 50,
        "context": context,
        "groupby": "attribute_id/id",
    },
    m2m=True,
)

# STEP 6: Generate data for Attribute Lines (linking attributes to templates)
print("Generating data for product attribute lines...")
line_mapping = {
    "id": mapper.m2m_id_list(
        ATTRIBUTE_LINE_PREFIX,
        *[
            mapper.concat_mapper_all("_", mapper.field(f), mapper.val("ref"))
            for f in attribute_list
        ],
    ),
    "product_tmpl_id/id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "attribute_id/id": mapper.m2m_id_list(
        ATTRIBUTE_PREFIX, *[mapper.field(f) for f in attribute_list]
    ),
    "value_ids/id": mapper.m2m_template_attribute_value(
        ATTRIBUTE_VALUE_PREFIX, *attribute_list
    ),
}
context_with_update = context.copy()
context_with_update["update_many2many"] = True
processor.process(
    line_mapping,
    os.path.join("data", "product.attribute.line.csv"),
    {
        "model": "product.attribute.line",
        "worker": 3,
        "batch_size": 50,
        "context": context_with_update,
        "groupby": "product_tmpl_id/id",
    },
    m2m=True,
)

# STEP 7: Generate data for final Product Variants (product.product)
print("Generating data for product variants...")
product_mapping = {
    "id": mapper.m2o_map(PRODUCT_PREFIX, "barcode", skip=True),
    "barcode": mapper.val("barcode"),
    "product_tmpl_id/id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    # This mapper seems to handle the complex logic of finding the correct
    # attribute values for a given variant.
    "attribute_value_ids/id": mapper.m2m_template_attribute_value(
        ATTRIBUTE_VALUE_PREFIX, "Color", "Gender", "Size_H", "Size_W"
    ),
    "default_code": mapper.val("ref"),
    "standard_price": mapper.num("cost"),
}
processor.process(
    product_mapping,
    os.path.join("data", "product.product.csv"),
    {
        "model": "product.product",
        "worker": 3,
        "batch_size": 50,
        "groupby": "product_tmpl_id/id",
        "context": context,
    },
)

print("Product v10 test data generation complete.")
