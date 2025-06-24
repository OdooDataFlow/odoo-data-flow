"""This test script checks the file joining functionality of the Processor.

It merges two source CSV files based on a common key.
"""

import os

from odoo_data_flow.lib import transform

# --- Configuration ---
SOURCE_FILE_1 = os.path.join("tests", "origin", "test_merge1.csv")
SOURCE_FILE_2 = os.path.join("tests", "origin", "test_merge2.csv")

# --- Main Logic ---
print(f"Initializing processor with primary file: {SOURCE_FILE_1}")
# The 'filename' argument is deprecated, but we keep it for now
# to match the existing test file structure.
# A future refactor could update the Processor to use a more explicit name.
processor = transform.Processor(filename=SOURCE_FILE_1)

print(f"Joining with secondary file: {SOURCE_FILE_2}")
# Join the second file into the processor's data buffer.
# The join happens where the value in the 'category' column of file 1
# matches the value in the 'name' column of file 2.
processor.join_file(SOURCE_FILE_2, "category", "name")

print("File join complete. The processor now holds the merged data in memory.")
# Note: This test script only performs the in-memory join.
# A subsequent step in a test runner would be needed to process
# this merged data into a final output file.
