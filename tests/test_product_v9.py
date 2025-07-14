"""Test Odoo version 9 product import.

This test script generates a complete set of data files for importing
products with variants, including categories and attributes.
This is based on the v9 product structure.
"""

import os

import polars as pl

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import ProductProcessorV9

# --- Configuration ---
TEMPLATE_PREFIX = "PRODUCT_TEMPLATE"
PRODUCT_PREFIX = "PRODUCT_PRODUCT"
CATEGORY_PREFIX = "PRODUCT_CATEGORY"
ATTRIBUTE_PREFIX = "PRODUCT_ATTRIBUTE"
ATTRIBUTE_VALUE_PREFIX = "PRODUCT_ATTRIBUTE_VALUE"

attribute_list = ["Color", "Gender", "Size_H", "Size_W"]
source_file = os.path.join("tests", "origin", "product.csv")
context = {"create_product_variant": True, "tracking_disable": True}


# --- Main Logic ---
print(f"Loading source data from: {source_file}")
source_df = pl.read_csv(source_file, separator=",")

# STEP 1: Categories
print("Generating data for product categories...")
categ_parent_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("categoy"),
}
parent_categ_processor = ProductProcessorV9(
    mapping=categ_parent_map, dataframe=source_df.clone()
)
parent_categ_processor.process(
    filename_out=os.path.join("data", "product.category.parent.v9.csv"),
    params={"model": "product.category"},
)
categ_map = {
    "id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "parent_id/id": mapper.m2o_map(CATEGORY_PREFIX, "categoy"),
    "name": mapper.val("Sub Category"),
}
child_categ_processor = ProductProcessorV9(
    mapping=categ_map, dataframe=source_df.clone()
)
child_categ_processor.process(
    filename_out=os.path.join("data", "product.category.v9.csv"),
    params={"model": "product.category"},
)

# STEP 2: Product Templates
print("Generating data for product templates...")
template_map = {
    "id": mapper.m2o_map(TEMPLATE_PREFIX, "ref"),
    "categ_id/id": mapper.m2o_map(CATEGORY_PREFIX, "Sub Category"),
    "name": mapper.val("name"),
}
template_processor = ProductProcessorV9(
    mapping=template_map, dataframe=source_df.clone()
)
template_processor.process(
    filename_out=os.path.join("data", "product.template.v9.csv"),
    params={"model": "product.template", "context": context},
    t="set",
)

# STEP 3: Attributes
print("Generating data for product attributes...")
attribute_map = {
    "id": mapper.const(ATTRIBUTE_PREFIX),
    "name": mapper.const(None),
}
attribute_processor = ProductProcessorV9(
    mapping={
        "id": mapper.m2m_id_list(ATTRIBUTE_PREFIX, *attribute_list),
        "name": mapper.m2m_value_list(*attribute_list),
    },
    dataframe=pl.DataFrame({"placeholder": [1]}),
)

# STEP 4: Attribute Values (using the robust unpivot strategy)
print("Generating data for product attribute values...")
id_vars = [col for col in source_df.columns if col not in attribute_list]
unpivoted_df = source_df.unpivot(
    index=id_vars,
    on=attribute_list,
    variable_name="attribute_name",
    value_name="attribute_value",
).filter(pl.col("attribute_value").is_not_null())

attribute_value_map = {
    "id": mapper.concat(
        ATTRIBUTE_VALUE_PREFIX, "_", "attribute_name", "_", "attribute_value"
    ),
    "name": mapper.val("attribute_value"),
    "attribute_id/id": mapper.m2o_map(ATTRIBUTE_PREFIX, "attribute_name"),
}
value_processor = ProductProcessorV9(
    mapping=attribute_value_map, dataframe=unpivoted_df
)
value_processor.process(
    filename_out=os.path.join("data", "product.attribute.value.v9.csv"),
    params={"model": "product.attribute.value", "context": context},
    t="set",
)

# STEP 5: Attribute Lines
print("Generating data for product attribute lines...")
line_mapping = {
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

line_processor = ProductProcessorV9(mapping=line_mapping, dataframe=source_df.clone())

line_processor.process(
    filename_out=os.path.join("data", "product.attribute.line.v9.csv"),
    params={"model": "product.attribute.line", "groupby": "product_tmpl_id/id"},
    m2m=True,
    m2m_columns=attribute_list,
)

print("Product v9 test data generation complete.")
