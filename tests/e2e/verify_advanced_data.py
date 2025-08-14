"""This file handles the data verification for the advanced e2e tests."""

import logging
import os
import sys

import odoo
from odoo.tools import config

_logger = logging.getLogger(__name__)


def _get_baseline_partner_count(env):
    """Get the baseline count of partners."""
    partner_model = env["res.partner"]
    return partner_model.search_count([])


def _verify_categories(env):
    """Verify the partner categories."""
    category_model = env["res.partner.category"]
    categories = category_model.search([])
    if len(categories) != 2:
        raise AssertionError(
            f"Expected 2 partner categories, but found {len(categories)}."
        )
    _logger.info("Partner categories verification successful.")


def _verify_companies(env, baseline_company_count):
    """Verify the parent companies."""
    partner_model = env["res.partner"]
    parent_companies = partner_model.search([("is_company", "=", True)])
    if len(parent_companies) != baseline_company_count + 10:
        raise AssertionError(
            f"Expected {baseline_company_count + 10} parent companies, "
            f"but found {len(parent_companies)}."
        )
    _logger.info("Parent companies verification successful.")


def _verify_partners(env, num_partners, baseline_partner_count):
    """Verify the partners."""
    partner_model = env["res.partner"]
    partners = partner_model.search([("name", "like", "Test Partner")])
    if len(partners) != num_partners:
        raise AssertionError(
            f"Expected {num_partners} partners, but found {len(partners)}."
        )
    _logger.info(f"Partners verification successful. Found {len(partners)} partners.")

    total_partners = partner_model.search_count([])
    expected_total = baseline_partner_count + num_partners * 2 + 10
    if total_partners != expected_total:
        raise AssertionError(
            f"Expected {expected_total} total partners, but found {total_partners}."
        )
    _logger.info("Total partner count verification successful.")
    return partners


def _verify_child_partners(env, num_partners):
    """Verify the child partners."""
    partner_model = env["res.partner"]
    child_partners = partner_model.search([("name", "like", "Child Partner")])
    if len(child_partners) != num_partners:
        raise AssertionError(
            f"Expected {num_partners} child partners, but found {len(child_partners)}."
        )
    _logger.info(
        f"Child partners verification successful. Found {len(child_partners)} partners."
    )


def _verify_relationships(partners):
    """Verify the relationships between partners."""
    for _, partner in enumerate(partners):
        # Verify parent
        if not partner.parent_id:
            raise AssertionError(f"Partner {partner.name} has no parent.")

        # Verify categories
        has_vip = any(cat.name == "VIP" for cat in partner.category_id)
        has_standard = any(cat.name == "Standard" for cat in partner.category_id)

        partner_index = int(partner.name.split(" ")[-1]) - 1
        if partner_index % 2 == 0 and not has_vip:
            raise AssertionError(f"Partner {partner.name} should have VIP category.")
        if partner_index % 3 == 0 and not has_standard:
            raise AssertionError(
                f"Partner {partner.name} should have Standard category."
            )

        # Verify children
        if len(partner.child_ids) != 1:
            raise AssertionError(
                f"Partner {partner.name} should have 1 child, "
                f"but has {len(partner.child_ids)}."
            )
        if partner.child_ids[0].name != f"Child Partner {partner_index + 1}":
            raise AssertionError(
                f"Partner {partner.name} has wrong child: {partner.child_ids[0].name}."
            )
    _logger.info("Partner relationships verification successful.")


def verify_data(db_name: str, num_partners: int = 550) -> None:
    """Verify the partner data for the advanced e2e test.

    Args:
        db_name: The name of the database to verify.
        num_partners: The expected number of partners.
    """
    _logger.info("Verifying advanced partner data...")

    config["db_host"] = os.environ.get("HOST")
    config["db_port"] = int(os.environ.get("PORT", 5432))
    config["db_user"] = os.environ.get("USER")
    config["db_password"] = os.environ.get("PASSWORD")

    registry = odoo.registry(db_name)
    with odoo.api.Environment.manage(), registry.cursor() as cr:
        env = odoo.api.Environment(cr, odoo.SUPERUSER_ID, {})

        baseline_partner_count = _get_baseline_partner_count(env)
        baseline_company_count = env["res.partner"].search_count(
            [("is_company", "=", True)]
        )

        _verify_categories(env)
        _verify_companies(env, baseline_company_count)
        partners = _verify_partners(env, num_partners, baseline_partner_count)
        _verify_child_partners(env, num_partners)
        _verify_relationships(partners)


if __name__ == "__main__":
    try:
        db_name = sys.argv[1]
    except IndexError:
        _logger.error(
            "Database name not provided. "
            "Usage: python3 verify_advanced_data.py <db_name> [num_partners]"
        )
        sys.exit(1)

    try:
        num_partners_arg = int(sys.argv[2])
    except (IndexError, ValueError):
        num_partners_arg = 550

    verify_data(db_name, num_partners_arg)
