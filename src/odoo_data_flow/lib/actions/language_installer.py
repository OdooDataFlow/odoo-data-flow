"""This module contains the logic for installing languages in Odoo."""

from typing import Any

from ...lib import conf_lib
from ...logging_config import log


def run_language_installation(config: str, languages: list[str]) -> None:
    """Connects to Odoo and installs a list of languages.

    Args:
        config: Path to the connection configuration file.
        languages: A list of language codes to install (e.g., ['nl_BE', 'fr_FR']).
    """
    log.info(f"--- Starting Language Installation for: {', '.join(languages)} ---")
    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        wizard_obj = connection.get_model("base.language.install")
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return

    try:
        log.info(f"Preparing to install languages: {languages}")
        # The wizard expects a dictionary with a 'lang' key
        wizard_data = {"lang": ",".join(languages)}
        wizard_id = wizard_obj.create(wizard_data)
        # The wizard's method is confusingly named but this is correct
        wizard_obj.lang_install([wizard_id])
        log.info("Language installation process triggered successfully.")
    except Exception as e:
        log.error(f"An error occurred during language installation: {e}")

    log.info("--- Language Installation Finished ---")
