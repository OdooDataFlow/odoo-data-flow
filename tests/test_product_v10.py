"""Test Odoo version 10 product import.

This test script generates a complete set of data files for importing
products with variants, including categories, attributes, and attribute lines.
This is based on the v10 product structure.
"""

import os

import polars as pl

from odoo_data_flow.lib import mapper
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

categ_parent_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("categoy"),
}
categ_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "parent_id/id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("Sub Category"),
}

template_map = {
    "id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "categ_id/id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "standard_price": mapper.num("cost"),
    "list_price": mapper.num("public_price"),
    "default_code": mapper.val("ref"),
    "name": mapper.val("name"),
    "type": mapper.const("product"),
}
attribute_value_mapping = {
    "id": mapper.m2m_template_attribute_value(ATTRIBUTE_VALUE_PREFIX, *attribute_list),
    "name": mapper.m2m_value_list(*attribute_list),
    "attribute_id/id": mapper.m2m_id_list(
        ATTRIBUTE_PREFIX, *[mapper.field(f) for f in attribute_list]
    ),
}
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
        ATTRIBUTE_VALUE_PREFIX, "Color", "Gender", "Size_H", "Size_W"
    ),
}
product_mapping = {
    "id": mapper.m2o_map(PRODUCT_PREFIX, "barcode", skip=True),
    "barcode": mapper.val("barcode"),
    "product_tmpl_id/id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "attribute_value_ids/id": mapper.m2m_template_attribute_value(
        ATTRIBUTE_VALUE_PREFIX, "Color", "Gender", "Size_H", "Size_W"
    ),
    "default_code": mapper.val("ref"),
    "standard_price": mapper.num("cost"),
}
# --- Main Logic ---
print(f"Loading source data from: {source_file}")
source_df = pl.read_csv(source_file, separator=",")


# STEP 1: Parent Categories
print("Generating data for product categories (parents)...")
parent_categ_processor = ProductProcessorV10(
    mapping=categ_parent_map, dataframe=source_df.clone()
)
parent_categ_processor.process(
    filename_out=os.path.join("data", "product.category.parent.csv"),
    params={"model": "product.category", "worker": 1, "batch_size": 5},
    t="set",
)

# STEP 2: Child Categories
print("Generating data for product categories (children)...")
child_categ_processor = ProductProcessorV10(
    mapping=categ_map, dataframe=source_df.clone()
)
child_categ_processor.process(
    filename_out=os.path.join("data", "product.category.csv"),
    params={"model": "product.category", "worker": 1, "batch_size": 20},
    t="set",
)

# STEP 3: Product Templates
print("Generating data for product templates...")
template_processor = ProductProcessorV10(
    mapping=template_map, dataframe=source_df.clone()
)
template_processor.process(
    filename_out=os.path.join("data", "product.template.csv"),
    params={
        "model": "product.template",
        "worker": 4,
        "batch_size": 10,
        "context": context,
    },
    t="set",
)

# STEP 4: Product Attributes
print("Generating data for product attributes...")
# Provide an empty mapping to satisfy the constructor
attribute_processor = ProductProcessorV10(mapping={}, dataframe=source_df.clone())
attribute_processor.process_attribute_data(
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

# STEP 5: Product Attribute Values
print("Generating data for product attribute values...")
simple_value_mapping = {
    "id": mapper.concat(
        ATTRIBUTE_VALUE_PREFIX,
        "_",
        "m2m_source_column",
        "_",
        "m2m_source_value",
    ),
    "name": mapper.val("m2m_source_value"),
    "attribute_id/id": mapper.m2o_map(ATTRIBUTE_PREFIX, "m2m_source_column"),
}

# The Processor now takes the simple mapping.
attr_value_processor = ProductProcessorV10(
    mapping=simple_value_mapping, dataframe=source_df.clone()
)

# The process call is clean. It just needs to know which columns to unpivot.
attr_value_processor.process(
    filename_out=os.path.join("data", "product.attribute.value.csv"),
    params={
        "model": "product.attribute.value",
        "worker": 3,
        "batch_size": 50,
        "context": context,
        "groupby": "attribute_id/id",
    },
    t="set",
    m2m=True,
    m2m_columns=attribute_list,
)


# STEP 6: Product Attribute Lines
print("Generating data for product attribute lines...")

# Define a new, simple mapping that works on the long-format data
# generated internally by the m2m processor.
simple_line_mapping = {
    "id": mapper.concat(ATTRIBUTE_LINE_PREFIX, "_", "m2m_source_column", "_", "ref"),
    "product_tmpl_id/id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "attribute_id/id": mapper.m2o_map(ATTRIBUTE_PREFIX, "m2m_source_column"),
    "value_ids/id": mapper.concat(
        ATTRIBUTE_VALUE_PREFIX,
        "_",
        "m2m_source_column",
        "_",
        "m2m_source_value",
    ),
}

context_with_update = context.copy()
context_with_update["update_many2many"] = True

# The Processor now takes the new, simple mapping.
line_processor = ProductProcessorV10(
    mapping=simple_line_mapping, dataframe=source_df.clone()
)

# The process call now includes the 'm2m_columns' argument.
line_processor.process(
    filename_out=os.path.join("data", "product.attribute.line.csv"),
    params={
        "model": "product.attribute.line",
        "worker": 3,
        "batch_size": 50,
        "context": context_with_update,
        "groupby": "product_tmpl_id/id",
    },
    m2m=True,
    m2m_columns=attribute_list,
)

# STEP 7: Product Variants
print("Generating data for product variants...")
product_processor = ProductProcessorV10(
    mapping=product_mapping, dataframe=source_df.clone()
)
product_processor.process(
    filename_out=os.path.join("data", "product.product.csv"),
    params={
        "model": "product.product",
        "worker": 3,
        "batch_size": 50,
        "groupby": "product_tmpl_id/id",
        "context": context,
    },
)

print("Product v10 test data generation complete.")
