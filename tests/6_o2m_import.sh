#!/usr/bin/env bash
#
# Tests the import of one-to-many (o2m) relationships.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Testing one-to-many (o2m) import ---"
odoo-data-flow import \
    --config "tests/conf/connection.conf" \
    --file "tests/origin/res.partner_o2m.csv" \
    --model "res.partner" \
    --size 1 \
    --worker 1 \
    --o2m

echo "o2m import test complete."
