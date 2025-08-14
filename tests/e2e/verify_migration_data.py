"""This file handles the data verification for the migration e2e tests."""

import logging
import sys

import odoo

_logger = logging.getLogger(__name__)

try:
    db_name = sys.argv[1]
except IndexError:
    _logger.error(
        "Database name not provided. Usage: python3 verify_migration_data.py <db_name>"
    )
    sys.exit(1)


def verify_migration(db_name: str) -> None:
    """Verify the partner data after migration."""
    print("Verifying migrated partner data...")

    registry = odoo.sql_db.db_connect(db_name)
    cr = registry.cursor()
    env = odoo.api.Environment(cr, odoo.SUPERUSER_ID, {})

    # 1. Verify country 'United States' exists
    country_model = env["res.country"]
    us_country = country_model.search([("code", "=", "US")])
    if not us_country:
        raise AssertionError(
            "Country 'United States' not found in the target database."
        )
    us_country_id = us_country.id

    # 2. Verify partner count
    partner_model = env["res.partner"]
    total_partners = partner_model.search_count([])
    expected_count = 1002
    if total_partners != expected_count:
        raise AssertionError(
            f"Expected {expected_count} partner records in the target database, "
            f"but found {total_partners}. Migration might have failed."
        )
    print(f"Total partner count is correct: {total_partners}")

    # 3. Verify the migrated data
    migrated_partners = partner_model.search_read(
        [("name", "like", "Test Partner")], ["name", "country_id"]
    )

    migrated_count = len(migrated_partners)
    expected_migrated_count = 1000
    if migrated_count != expected_migrated_count:
        raise AssertionError(
            f"Expected {expected_migrated_count} migrated partner records, "
            f"but found {migrated_count}."
        )
    print(f"Found {migrated_count} migrated partners.")

    # 4. Verify the relational field (country_id)
    for partner in migrated_partners:
        if not partner["country_id"]:
            raise AssertionError(
                f"Partner '{partner['name']}' is missing country information."
            )
        if partner["country_id"][0] != us_country_id:
            raise AssertionError(
                f"Partner '{partner['name']}' has incorrect country. "
                f"Expected 'United States' (ID: {us_country_id}), "
                f"but got ID: {partner['country_id'][0]}."
            )

    print("All migrated partners have the correct country.")

    # 5. Final success message
    print(
        f"Verification successful: Found {total_partners} total partner records, "
        f"and all {migrated_count} migrated partners have the correct relational data."
    )

    cr.close()


if __name__ == "__main__":
    try:
        db_name = sys.argv[1]
    except IndexError:
        _logger.error(
            "Database name not provided. Usage: python3 verify_migration_data.py <db_name>"
        )
        sys.exit(1)

    verify_migration(db_name)
