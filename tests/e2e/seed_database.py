"""This file handles the seedint of test data verification for the e2e tests."""

import logging
import sys

import odoo

_logger = logging.getLogger(__name__)

# This script must be run with Odoo's environment already set up.
# We'll get the database name from a command-line argument for flexibility.
try:
    db_name = sys.argv[1]
except IndexError:
    _logger.error(
        "Database name not provided. Usage: python3 seed_database.py <db_name>"
    )
    sys.exit(1)


def seed_database(db_name: str) -> None:
    "Seed partner data for e2e test."
    # This is the correct way to connect to a database from an external script
    registry = odoo.sql_db.db_connect(db_name)
    cr = registry.cursor()
    env = odoo.api.Environment(cr, odoo.SUPERUSER_ID, {})

    _logger.info("Starting to seed the database with partner data...")

    partner_model = env["res.partner"]

    num_partners = 1000
    partners_to_create = []
    for i in range(num_partners):
        partners_to_create.append(
            {
                "name": f"Test Partner {i + 1}",
                "is_company": True,
                "email": f"test.partner.{i + 1}@example.com",
            }
        )

    partner_model.create(partners_to_create)

    _logger.info(f"Successfully created {num_partners} partner records.")

    # Explicitly commit and close the cursor
    cr.commit()
    cr.close()


if __name__ == "__main__":
    try:
        db_name = sys.argv[1]
    except IndexError:
        _logger.error(
            "Database name not provided. Usage: python3 seed_database.py <db_name>"
        )
        sys.exit(1)

    seed_database(db_name)
