"""This module contains the logic for installing languages in Odoo."""

from typing import Any

from ...lib import conf_lib, odoo_lib
from ...logging_config import log


def _install_languages_v18_plus(connection: Any, languages: list[str]) -> None:
    """Activates languages directly for Odoo 18 and newer."""
    log.info("Using direct activation method (Odoo 18+).")
    lang_model = connection.get_model("res.lang")

    # Find inactive languages with the given codes
    lang_ids = lang_model.search([("code", "in", languages), ("active", "=", False)])

    if not lang_ids:
        log.warning(f"Languages are already active or do not exist: {languages}")
        return

    log.info(f"Activating language IDs: {lang_ids}")
    # Directly call the write method to set them to active
    lang_model.write(lang_ids, {"active": True})


def _install_languages_v15_plus(
    connection: Any, wizard_obj: Any, languages: list[str]
) -> None:
    """Installs languages using the method for Odoo 15 and newer."""
    log.info("Using modern installation method (Odoo 15+).")
    lang_model = connection.get_model("res.lang")
    lang_ids = lang_model.search([("code", "in", languages)])
    if not lang_ids:
        log.warning(f"None of the specified languages were found in Odoo: {languages}")
        return
    wizard_data = {"langs": [(6, 0, lang_ids)]}
    wizard_id = wizard_obj.create(wizard_data)
    log.info(f"Created installation wizard with ID: {wizard_id}")
    wizard_obj.browse(wizard_id).lang_install()


def _install_languages_legacy(
    connection: Any, wizard_obj: Any, languages: list[str]
) -> None:
    """Installs languages using the legacy method for Odoo 14 and older."""
    log.info("Using legacy installation method (Odoo <15).")
    # Legacy versions expect one language per wizard. We loop through them.
    for lang_code in languages:
        try:
            log.info(f"Installing language: {lang_code}")
            wizard_id = wizard_obj.create({"lang": lang_code})
            wizard_obj.lang_install([wizard_id])
            log.info(f"Triggered installation for '{lang_code}'.")
        except Exception as e:
            log.error(f"Failed to install language '{lang_code}': {e}")


def run_language_installation(config: str, languages: list[str]) -> None:
    """Connects to Odoo and installs a list of languages, auto-detecting the version."""
    log.info(f"--- Starting Language Installation for: {', '.join(languages)} ---")
    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        odoo_version = odoo_lib.get_odoo_version(connection)
    except Exception as e:
        log.error(f"Failed to connect to Odoo or detect version: {e}")
        return

    try:
        # New logic for Odoo 18 and newer
        if odoo_version >= 18:
            _install_languages_v18_plus(connection, languages)
            log.info("Language installation process triggered successfully.")
            log.info("--- Language Installation Finished ---")
            return

        # Logic for Odoo 15, 16, 17
        elif odoo_version >= 15:
            wizard_obj = connection.get_model("base.language.install")
            _install_languages_v15_plus(connection, wizard_obj, languages)

        # Fallback for Odoo 14 and older
        else:
            wizard_obj = connection.get_model("base.language.install")
            _install_languages_legacy(connection, wizard_obj, languages)

        log.info("Language installation process triggered successfully.")
    except Exception as e:
        log.error(f"An unexpected error occurred during language installation: {e}")

    log.info("--- Language Installation Finished ---")
