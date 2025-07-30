"""This module contains the logic for installing languages in Odoo."""

import time
from typing import Any

from ...logging_config import log
from .. import conf_lib, odoo_lib


def _wait_for_languages_to_be_active(
    connection: Any, languages: list[str], timeout: int = 300
) -> bool:
    """Polls Odoo until the specified languages are active, with a timeout."""
    log.info(
        f"Waiting for languages to become active: {languages} (timeout: {timeout}s)"
    )
    lang_model = connection.get_model("res.lang")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            installed_langs_data = lang_model.search_read(
                [("code", "in", languages), ("active", "=", True)], ["code"]
            )
            active_langs = {lang["code"] for lang in installed_langs_data}

            if set(languages).issubset(active_langs):
                log.info("All requested languages are now active.")
                return True

            log.info(f"Still waiting... Active so far: {sorted(list(active_langs))}")
            time.sleep(5)  # Wait 5 seconds before polling again

        except Exception as e:
            log.error(f"An error occurred while checking language status: {e}")
            return False

    log.error("Timeout reached while waiting for languages to become active.")
    return False


def _install_languages_modern(
    connection: Any, languages: list[str], version: int
) -> None:
    """Installs languages using the wizard method for Odoo 15, 16, and 17."""
    log.info(f"Using modern installation wizard (Odoo {version}).")
    wizard_obj = connection.get_model("base.language.install")
    lang_model = connection.get_model("res.lang")

    # Use `active_test: False` to find inactive language records.
    lang_ids = lang_model.search(
        [("code", "in", languages)], context={"active_test": False}
    )
    if not lang_ids:
        log.warning(
            f"None of the specified languages could be found in Odoo: {languages}"
        )
        return

    # Odoo 17 and newer use 'lang_ids'; older versions (15, 16) use 'langs'.
    key = "lang_ids" if version >= 17 else "langs"
    wizard_data = {key: [(6, 0, lang_ids)]}

    try:
        wizard_id = wizard_obj.create(wizard_data)
        log.info(f"Created installation wizard with ID: {wizard_id}")
        wizard_obj.browse(wizard_id).lang_install()
    except Exception as e:
        # Handle the edge case where a version might have an unexpected key
        log.error(f"Failed to create language wizard with key '{key}': {e}")
        if key == "lang_ids":
            try:
                log.debug("Attempting fallback with 'langs' key for wizard.")
                fallback_data = {"langs": [(6, 0, lang_ids)]}
                wizard_id = wizard_obj.create(fallback_data)
                log.info(f"Created installation wizard with ID: {wizard_id}")
                wizard_obj.browse(wizard_id).lang_install()
            except Exception as fallback_e:
                log.error(f"Fallback attempt also failed: {fallback_e}")
                raise fallback_e
        else:
            raise e


def _install_languages_v18_and_legacy(connection: Any, languages: list[str]) -> None:
    """Installs languages using the method for Odoo <=14 and Odoo 18+."""
    log.info("Using per-language wizard installation method (Odoo <=14 or 18+).")
    wizard_obj = connection.get_model("base.language.install")

    for lang_code in languages:
        try:
            log.info(f"Creating install wizard for language: {lang_code}")
            wizard_id = wizard_obj.create({"lang": lang_code})
            wizard_obj.browse(wizard_id).lang_install()
            log.info(f"Successfully triggered installation for '{lang_code}'.")
        except Exception as e:
            log.error(f"Failed to install language '{lang_code}': {e}")


def run_language_installation(config: str, languages: list[str]) -> bool:
    """Connects to Odoo and installs a list of languages, auto-detecting the version."""
    log.info(f"--- Starting Language Installation for: {', '.join(languages)} ---")
    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        odoo_version = odoo_lib.get_odoo_version(connection)
    except Exception as e:
        log.error(f"Failed to connect to Odoo or detect version: {e}")
        return False

    try:
        # Odoo 18 and legacy versions (<15) share the same simple installation method.
        if odoo_version >= 18 or odoo_version < 15:
            _install_languages_v18_and_legacy(connection, languages)

        # Odoo 15, 16, and 17 use a more complex wizard that takes a list of IDs.
        else:  # This covers versions 15, 16, and 17
            _install_languages_modern(connection, languages, odoo_version)

        log.info("Language installation process triggered successfully.")

        # After triggering, wait for the languages to become active.
        if not _wait_for_languages_to_be_active(connection, languages):
            log.error("Installation failed or timed out.")
            return False

        log.info("--- Language Installation Finished ---")
        return True
    except Exception as e:
        log.error(
            f"An unexpected error occurred during language installation: {e}",
            exc_info=True,
        )
        return False
