#!/usr/bin/env bash
#
# Main test suite for odoo-data-flow.
# This script should be run from the root of the repository.
#
# Prerequisites:
# 1. A virtual environment is active.
# 2. The package has been installed in editable mode:
#    uv pip install -e .
# 3. An Odoo database is running and accessible via the configuration
#    in tests/conf/connection.conf

# Exit immediately if a command exits with a non-zero status.
set -e

echo "============== Starting Odoo Data Flow Test Suite =============="

# --- Cleanup and Setup ---
echo "> Cleaning up previous test runs..."
rm -rf data/ .coverage *.fail *.fail.bis error.log
mkdir -p data/

echo "> Erasing previous coverage data..."
coverage erase

# --- Running Tests ---
# Python scripts that prepare data are run under coverage.
# The subsequent steps use the new 'odoo-data-flow' CLI to test functionality.

echo "> 1. Generating data and running initial partner import..."
coverage run -a tests/test_import.py
# Replaces 0_partner_generated.sh
odoo-data-flow import --config tests/conf/connection.conf --file data/res_partner.csv --model res.partner

echo "> 2. Testing file split functionality..."
coverage run -a tests/test_split.py

echo "> 3. Testing mapping from file..."
coverage run -a tests/test_from_file.py

echo "> 4. Importing data with expected errors..."
# Replaces 2_contact_import.sh
# Assumes test_import.py also creates this file.
odoo-data-flow import --config tests/conf/connection.conf --file data/contact.csv --model res.partner 2> error.log

echo "> 5. Importing Product (v9)..."
coverage run -a tests/test_product_v9.py
# Replaces 3_product_import.sh
# Assumes the python script generates 'product_template_v9.csv'
odoo-data-flow import --config tests/conf/connection.conf --file data/product_template_v9.csv --model product.template

echo "> 6. Importing Product (v10)..."
coverage run -a tests/test_product_v10.py
# Replaces 4_product_import.sh
# Assumes the python script generates 'product_template_v10.csv'
odoo-data-flow import --config tests/conf/connection.conf --file data/product_template_v10.csv --model product.template

echo "> 7. Exporting Partners..."
# Replaces 5_partner_export.sh
odoo-data-flow export --config tests/conf/connection.conf --model res.partner --fields "name,email" --output data/exported_partners.csv

echo "> 8. Importing One2Many relations..."
# Replaces 6_o2m_import.sh
# Assumes a file like 'res.partner_o2m.csv' is generated/present
odoo-data-flow import --config tests/conf/connection.conf --file tests/origin/res.partner_o2m.csv --model res.partner --o2m

echo "> 9. Converting Binary from Path..."
# Replaces 7_convert_binary.sh
# Assumes a source file with image paths exists for this test
odoo-data-flow path-to-image tests/origin/contact.csv --fields "image" --out data/contacts_with_images.csv

echo "> 10. Testing merge functionality..."
coverage run -a tests/test_merge.py


# --- Finalizing ---
echo ""
echo "> Generating coverage report..."
coverage html

echo "============== Test Suite Finished Successfully =============="
