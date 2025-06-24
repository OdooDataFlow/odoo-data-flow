#!/usr/bin/env bash
#
# Tests the binary conversion commands.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Testing binary conversion from local file paths ---"
odoo-data-flow path-to-image \
    tests/origin/contact.csv \
    --path "tests/origin/img/" \
    --fields "Image" \
    --out "data/contacts_from_path.csv"

echo "--- Testing binary conversion from URLs ---"
odoo-data-flow url-to-image \
    tests/origin/contact_url.csv \
    --fields "Image" \
    --out "data/contacts_from_url.csv"

echo "Binary conversion tests complete."
