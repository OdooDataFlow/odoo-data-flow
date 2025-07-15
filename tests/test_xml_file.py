"""Test XML Files.

This test script checks the XML processing functionality.
It reads a source XML file, applies a mapping, and generates a
clean CSV file.
"""

import os

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import Processor

# --- Configuration ---
SOURCE_FILE = os.path.join("tests", "origin", "data.xml")
OUTPUT_FILE = os.path.join("data", "info_from_xml.csv")

# --- Mapping Definition ---
# This mapping is updated to use dot notation for tags, which is the
# standard way the Processor handles nested data.
# Note: The new Processor may not support XPath features like accessing
# attributes (@name) or indexed elements (neighbor[1]). This test
# focuses on the documented tag-based mapping.
mapping = {
    "name": mapper.val("year"),
    "gdp": mapper.val("gdppc"),
    # Assuming 'nom' and 'neighbor' are now represented as tags in the XML.
    "nom": mapper.val("name"),
    "neighbor": mapper.val("neighbor.name"),
}

# --- Main Logic ---

# Initialize the standard Processor, but with XML-specific arguments.
# We tell the processor that the records are enclosed in <country> tags,
# and the whole list is inside a root tag (e.g., <data>).
print(f"Initializing XML processor for source file: {SOURCE_FILE}")
processor = Processor(
    SOURCE_FILE,
    mapping={},
    separator=";",
    xml_root_tag="data",  # The root element containing all records
    xml_record_tag="country",  # The tag representing a single record
)

# Define the parameters for the eventual import.
params = {
    "model": "res.country.info",  # Example model
    "worker": 2,
    "batch_size": 5,
}

# Process the XML data using the mapping and write to a CSV file.
print(f"Processing XML data and writing to: {OUTPUT_FILE}")
processor.process(mapping, OUTPUT_FILE, params)

print("XML file transformation complete.")
