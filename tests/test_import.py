"""Generate Test data.

This test script generates data for partner categories and partners
to be used in the main test suite.
"""

import random

import polars as pl

from odoo_data_flow.lib import mapper, transform

# --- Configuration ---
PARTNER_PREFIX = "partner_generated"
TAG_PREFIX = "partner_tag"
PARTNER_OUTPUT = "data/res.partner.generated.csv"
TAG_OUTPUT = "data/res.partner.category.csv"

# --- Test Data Generation ---
# Create 100 unique tags
tags = [f"Tag {i}" for i in range(100)]

# Create a dataset for 200 partners, each assigned 5 random tags
header = ["id", "tags"]
data = [
    [str(i), ",".join(random.choice(tags) for _ in range(5))]  # noqa
    for i in range(200)
]

# --- Mapping Definitions ---

# Mapping to create the partner category records.
# This will be processed in a special m2m mode to create one record
# per unique tag.
tag_mapping = {
    "id": mapper.m2m_id_list(TAG_PREFIX, "tags"),
    "name": mapper.m2m("tags", sep=","),
    "parent_id/id": mapper.const("base.res_partner_category_0"),
}

# Mapping to create the partner records, linking them to the tags created above.
partner_mapping = {
    "id": mapper.concat(PARTNER_PREFIX, "_", "id"),
    "name": mapper.val("id", postprocess=lambda x: f"Partner {x}"),
    "phone": mapper.val("id", postprocess=lambda x: f"0032{int(x) * 11}"),
    "website": mapper.val("id", postprocess=lambda x: f"http://website-{x}.com"),
    "street": mapper.val("id", postprocess=lambda x: f"Street {x}"),
    "city": mapper.val("id", postprocess=lambda x: f"City {x}"),
    "zip": mapper.val("id", postprocess=lambda x: str(x).zfill(6)),
    "country_id/id": mapper.const("base.be"),
    "company_type": mapper.const("company"),
    "customer_rank": mapper.val("id", postprocess=lambda x: int(x) % 2),
    "supplier_rank": mapper.val("id", postprocess=lambda x: (int(x) + 1) % 2),
    "lang": mapper.const("en_US"),
    "category_id/id": mapper.m2m(TAG_PREFIX, "tags"),
}

# --- Processing ---

# Initialize the processor with the in-memory data
df = pl.DataFrame(data, schema=header, orient="row")
processor = transform.Processor(
    dataframe=df,
    mapping={},
)

# Process the tags first, using the special m2m=True mode.
# This will find all unique tags from the 'tags' column and create a clean
# CSV file with one row for each unique tag.
print(f"Generating partner category data at: {TAG_OUTPUT}")
processor.process(
    tag_mapping,
    TAG_OUTPUT,
    {"model": "res.partner.category"},
    m2m=True,
)

# Next, process the main partner records.
print(f"Generating partner data at: {PARTNER_OUTPUT}")
processor.process(
    partner_mapping,
    PARTNER_OUTPUT,
    {"model": "res.partner"},
)

print("Test data generation complete.")
