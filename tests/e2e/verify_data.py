"""This file handles the data verification for the e2e tests."""

import logging
import sys

import odoo

_logger = logging.getLogger(__name__)

try:
    db_name = sys.argv[1]
except IndexError:
    _logger.error("Database name not provided. Usage: python3 verify_data.py <db_name>")
    sys.exit(1)


def verify_partners(db_name: str) -> None:
    "Verify the partner data."
    print("Verifying partner data...")

    registry = odoo.sql_db.db_connect(db_name)
    cr = registry.cursor()
    env = odoo.api.Environment(cr, odoo.SUPERUSER_ID, {})

    target_partners = env["res.partner"].search_read(
        [], ["name", "email", "is_company"]
    )

    expected_count = 1002
    if len(target_partners) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} partner records in the target database, "
            f"but found {len(target_partners)}. Import failed."
        )

    # Using print() here to ensure the message is always visible in the console.
    print(
        f"Verification successful: Found {len(target_partners)} "
        f"partner records in the target database."
    )

    cr.close()


if __name__ == "__main__":
    try:
        db_name = sys.argv[1]
    except IndexError:
        _logger.error(
            "Database name not provided. Usage: python3 verify_data.py <db_name>"
        )
        sys.exit(1)

    verify_partners(db_name)
