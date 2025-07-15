"""Test The mapper from file.

This test script reads a CSV file from the 'origin' directory,
applies a mapping with various mappers, checks data quality,
and generates a clean CSV file ready for import.
"""

import os
import pprint

from odoo_data_flow.lib import checker, mapper
from odoo_data_flow.lib.transform import Processor

# --- Configuration ---
# Define translation maps and prefixes
lang_map = {
    "": "",
    "French": "French (BE) / Français (BE)",
    "English": "English",
    "Dutch": "Dutch / Nederlands",
}

country_map = {
    "Belgique": "base.be",
    "BE": "base.be",
    "FR": "base.fr",
    "U.S": "base.us",
    "US": "base.us",
    "NL": "base.nl",
}

PARTNER_PREFIX = "TEST_PARTNER"
IMAGE_PATH_PREFIX = "tests/origin/img/"

# --- Main Logic ---

# STEP 1: Initialize the processor with the source file
source_file = os.path.join("tests", "origin", "contact.csv")
processor = Processor(
    mapping={},
    filename=source_file,
    separator=";",
)

# Print the 1-to-1 mapping for debugging purposes
print("--- Auto-detected o2o Mapping ---")
pprint.pprint(processor.get_o2o_mapping())
print("---------------------------------")


# STEP 2: Define the mapping for every object to import
mapping = {
    "id": mapper.concat(PARTNER_PREFIX, "_", "Company_ID", skip=True),
    "name": mapper.val("Company_Name", skip=True),
    "phone": mapper.val("Phone"),
    "website": mapper.val("www"),
    "street": mapper.val("address1"),
    "city": mapper.val("city"),
    "zip": mapper.val("zip code"),
    "country_id/id": mapper.map_val(country_map, mapper.val("country")),
    "company_type": mapper.const("company"),
    # CORRECTED: bool_val now only takes a list of true values.
    "customer_rank": mapper.bool_val("IsCustomer", ["1"]),
    "supplier_rank": mapper.bool_val("IsSupplier", ["1"]),
    "lang": mapper.map_val(lang_map, mapper.val("Language")),
    # CORRECTED: Prepend the image path prefix using a postprocess function.
    # "image_1920": mapper.binary(
    #     "Image",
    #     postprocess=lambda p: os.path.join(IMAGE_PATH_PREFIX, p) if p else "",
    # ), TODO
    "image_1920": mapper.binary("Image", "origin/img/"),
}

# Step 3: Check data quality (Optional)
print("Running data quality checks...")
processor.check(checker.cell_len_checker(30))
processor.check(checker.id_validity_checker("Company_ID", r"COM\d"))
processor.check(checker.line_length_checker(13))
processor.check(checker.line_number_checker(21))

# Step 4: Process data
print("Processing data transformation...")
output_file = os.path.join("data", "res.partner.from_file.csv")
params = {"model": "res.partner", "worker": 2, "batch_size": 5}
processor.process(mapping, output_file, params)

print(f"File transformation complete. Output at: {output_file}")
