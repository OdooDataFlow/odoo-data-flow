#!/usr/bin/env bash
#
# Tests the data export functionality.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Testing data export ---"
odoo-data-flow export \
    --config "tests/conf/connection.conf" \
    --file "data/res.partner.exported.csv" \
    --model "res.partner" \
    --fields "id,name,phone,website,street,city,country_id/id" \
    --domain "[]" \
    --worker 4 \
    --size 200 \
    --separator ";" \
    --encoding "utf-8-sig"

echo "Data export test complete."
