"""This file handles the seeding of test data for the advanced e2e tests."""

import logging
import os
import sys

import odoo
from odoo.tools import config

_logger = logging.getLogger(__name__)


def get_baseline_partner_count(env):
    """Get the baseline count of partners."""
    partner_model = env["res.partner"]
    return partner_model.search_count([])


def seed_database(db_name: str, num_partners: int = 550) -> None:
    """Seed partner data for advanced e2e test.

    Args:
        db_name: The name of the database to seed.
        num_partners: The number of partners to create.
    """
    config["db_host"] = os.environ.get("HOST")
    config["db_port"] = int(os.environ.get("PORT", 5432))
    config["db_user"] = os.environ.get("USER")
    config["db_password"] = os.environ.get("PASSWORD")

    registry = odoo.registry(db_name)
    with odoo.api.Environment.manage(), registry.cursor() as cr:
        env = odoo.api.Environment(cr, odoo.SUPERUSER_ID, {})

        _logger.info("Starting to seed the database with advanced partner data...")

        # 1. Create Partner Categories with External IDs
        category_model = env["res.partner.category"]
        ir_model_data_model = env["ir.model.data"]

        categories_to_create = [
            {"name": "VIP", "xml_id": "__export__.partner_category_vip"},
            {"name": "Standard", "xml_id": "__export__.partner_category_standard"},
        ]
        categories = []
        for cat_data in categories_to_create:
            category = category_model.create({"name": cat_data["name"]})
            ir_model_data_model.create(
                {
                    "name": cat_data["xml_id"].split(".")[1],
                    "module": cat_data["xml_id"].split(".")[0],
                    "model": "res.partner.category",
                    "res_id": category.id,
                }
            )
            categories.append(category)
        _logger.info(f"Successfully created {len(categories)} partner categories.")

        # 2. Create Parent Companies
        partner_model = env["res.partner"]
        num_companies = 10
        companies_to_create = []
        for i in range(num_companies):
            companies_to_create.append(
                {
                    "name": f"Parent Company {i + 1}",
                    "is_company": True,
                    "email": f"company.{i + 1}@example.com",
                }
            )
        parent_companies = partner_model.create(companies_to_create)
        _logger.info(f"Successfully created {len(parent_companies)} parent companies.")

        # 3. Create Child Partners
        child_partners_to_create = []
        for i in range(num_partners):
            child_partners_to_create.append(
                {
                    "name": f"Child Partner {i + 1}",
                    "is_company": False,
                    "email": f"child.partner.{i + 1}@example.com",
                }
            )
        child_partners = partner_model.create(child_partners_to_create)
        _logger.info(f"Successfully created {len(child_partners)} child partners.")

        # 4. Create Partners with Parent, Categories, and Children
        partners_to_create = []
        for i in range(num_partners):
            parent_company = parent_companies[i % num_companies]
            partner_categories = []
            if i % 2 == 0:
                partner_categories.append(categories[0].id)
            if i % 3 == 0:
                partner_categories.append(categories[1].id)

            partners_to_create.append(
                {
                    "name": f"Test Partner {i + 1}",
                    "is_company": False,
                    "email": f"test.partner.{i + 1}@example.com",
                    "parent_id": parent_company.id,
                    "category_id": [(6, 0, partner_categories)],
                    "child_ids": [(6, 0, [child_partners[i].id])],
                }
            )

        partner_model.create(partners_to_create)
        _logger.info(f"Successfully created {num_partners} partner records.")


if __name__ == "__main__":
    try:
        db_name = sys.argv[1]
    except IndexError:
        _logger.error(
            "Database name not provided. "
            "Usage: python3 seed_advanced_database.py <db_name> [num_partners]"
        )
        sys.exit(1)

    try:
        num_partners_arg = int(sys.argv[2])
    except (IndexError, ValueError):
        num_partners_arg = 550

    seed_database(db_name, num_partners_arg)
