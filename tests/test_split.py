"""Test Split.

This test script generates a large dataset and then splits it into multiple
files to test the processor's split functionality.
"""

import random

import polars as pl

from odoo_data_flow.lib import mapper, transform

# --- Configuration ---
PARTNER_PREFIX = "partner_generated"
TAG_PREFIX = "partner_tag"
PARTNER_OUTPUT_PREFIX = "data/res.partner.generated.split"
TAG_OUTPUT = "data/res.partner.category.split.csv"

# --- Test Data Generation ---
# Create 100 unique tags
tags = [f"Tag {i}" for i in range(100)]

# Create a larger dataset for 10,000 partners
header = ["id", "tags"]
data = [
    [str(i), ",".join(random.choice(tags) for _ in range(5))]  # noqa nosec B311
    for i in range(10000)
]  # nosec B311

# --- Mapping Definitions (consistent with test_import.py) ---

# Mapping to create the partner category records.
tag_mapping = {
    "id": mapper.m2m_id_list(TAG_PREFIX, "tags"),
    "name": mapper.m2m("tags", sep=","),
    "parent_id/id": mapper.const("base.res_partner_category_0"),
}

# Mapping to create the partner records.
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
    "customer": mapper.val("id", postprocess=lambda x: int(x) % 2),
    "supplier": mapper.val("id", postprocess=lambda x: (int(x) + 1) % 2),
    "lang": mapper.const("en_US"),
    "category_id/id": mapper.m2m(TAG_PREFIX, "tags"),
}

# --- Processing ---
print("Initializing processor with 10,000 records.")
df = pl.DataFrame(data, schema=header, orient="row")
processor = transform.Processor(dataframe=df)

# This first split is primarily for test coverage purposes.
print("Running split by line number (for coverage)...")
processor.split(mapper.split_line_number(1000))

# This is the main test: split the dataset into 8 separate files.
print("Splitting data into 8 files...")
processor_dictionary = processor.split(mapper.split_file_number(8))

# First, process the tags into a single file from the main processor.
print(f"Generating single tag file for all splits at: {TAG_OUTPUT}")
processor.process(
    tag_mapping,
    TAG_OUTPUT,
    {"model": "res.partner.category"},
    m2m=True,
)

# Now, loop through the dictionary of split processors and have each one
# generate its own numbered output file.
print("Processing each data split into a separate partner file...")
for index, p in processor_dictionary.items():
    output_filename = f"{PARTNER_OUTPUT_PREFIX}.{index}.csv"
    print(f"  - Generating {output_filename}")
    p.process(
        partner_mapping,
        output_filename,
        {"model": "res.partner"},
    )

print("Split file generation complete.")
