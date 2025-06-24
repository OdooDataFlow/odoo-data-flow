"""Test Odoo version 9 product import.

This test script generates a complete set of data files for importing
products with variants, including categories and attributes.
This is based on the v9 product structure.
"""

import os

from odoo_data_flow.lib import mapper

# We assume ProductProcessorV9 is a custom class inheriting from Processor
from odoo_data_flow.lib.transform import ProductProcessorV9

# --- Configuration ---
TEMPLATE_PREFIX = "PRODUCT_TEMPLATE"
PRODUCT_PREFIX = "PRODUCT_PRODUCT"
CATEGORY_PREFIX = "PRODUCT_CATEGORY"
ATTRIBUTE_PREFIX = "PRODUCT_ATTRIBUTE"
ATTRIBUTE_VALUE_PREFIX = "PRODUCT_ATTRIBUTE_VALUE"

# Define the attributes to be processed from the source file
attribute_list = ["Color", "Gender", "Size_H", "Size_W"]
source_file = os.path.join("tests", "origin", "product.csv")
context = {"create_product_variant": True, "tracking_disable": True}

# --- Main Logic ---
# STEP 1: Initialize the custom processor with the source file
print(f"Initializing processor for v9 product import from: {source_file}")
processor = ProductProcessorV9(source_file, separator=",")

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
    os.path.join("data", "product.category.parent.v9.csv"),
    params={"model": "product.category"},
    m2m=True,
)
processor.process(
    categ_map,
    os.path.join("data", "product.category.v9.csv"),
    params={"model": "product.category"},
    m2m=True,
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
    os.path.join("data", "product.template.v9.csv"),
    params={"model": "product.template", "context": context},
    m2m=True,
)

# STEP 4: Generate data for Attributes and Values (in one go for v9)
# This was handled by a custom process_attribute_mapping in the original script.
# We now standardize this to create two separate, clean files.
print("Generating data for product attributes and values...")

# Attribute mapping
attribute_map = {
    "id": mapper.m2m_map(
        ATTRIBUTE_PREFIX, *[mapper.const(attr) for attr in attribute_list]
    ),
    "name": mapper.m2m(*[mapper.const(attr) for attr in attribute_list]),
}
processor.process(
    attribute_map,
    os.path.join("data", "product.attribute.v9.csv"),
    params={"model": "product.attribute"},
    m2m=True,
)

# Attribute Value mapping
attribute_value_map = {
    "id": mapper.m2m_template_attribute_value(ATTRIBUTE_VALUE_PREFIX, *attribute_list),
    "name": mapper.m2m_value_list(*attribute_list),
    "attribute_id/id": mapper.m2m_map(
        ATTRIBUTE_PREFIX, *[mapper.field(f) for f in attribute_list]
    ),
}
processor.process(
    attribute_value_map,
    os.path.join("data", "product.attribute.value.v9.csv"),
    params={
        "model": "product.attribute.value",
        "context": context,
        "groupby": "attribute_id/id",
    },
    m2m=True,
)


# STEP 5: Generate data for Product Variants (product.product)
print("Generating data for product variants...")
product_mapping = {
    "id": mapper.m2o_map(PRODUCT_PREFIX, "barcode", skip=True),
    "barcode": mapper.val("barcode"),
    "product_tmpl_id/id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "attribute_value_ids/id": mapper.m2m_template_attribute_value(
        ATTRIBUTE_VALUE_PREFIX, "Color", "Gender", "Size_H", "Size_W"
    ),
}
processor.process(
    product_mapping,
    os.path.join("data", "product.product.v9.csv"),
    params={
        "model": "product.product",
        "worker": 3,
        "batch_size": 50,
        "groupby": "product_tmpl_id/id",
        "context": context,
    },
)

print("Product v9 test data generation complete.")
