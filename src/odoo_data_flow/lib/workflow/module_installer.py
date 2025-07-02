"""This module contains the logic for (un)installing/upgrading Odoo modules."""

from typing import Any

from ...lib import conf_lib
from ...logging_config import log


def run_module_installation(
    config: str,
    modules: list[str],
) -> None:
    """Connects to Odoo and runs the module upgrade/install process.

    This function finds the specified modules, checks their current state,
    and then triggers the appropriate 'install' or 'upgrade' action.

    Args:
        config: Path to the connection configuration file.
        modules: A list of technical module names to install or upgrade.
    """
    log.info(f"--- Starting Module Installation Workflow for: {', '.join(modules)} ---")

    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        module_obj = connection.get_model("ir.module.module")
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return

    log.info(f"Searching for module(s): {modules}")
    module_ids = module_obj.search([("name", "in", modules)])

    if not module_ids:
        log.warning("No matching modules found in the database.")
        return

    # Read the state of each found module
    found_modules = module_obj.read(module_ids, ["name", "state"])
    log.info(f"Found modules: {found_modules}")

    modules_to_install = [m["id"] for m in found_modules if m["state"] == "uninstalled"]
    modules_to_upgrade = [m["id"] for m in found_modules if m["state"] == "installed"]

    try:
        if modules_to_install:
            log.info(
                f"Installing modules: "
                f"{[m['name'] for m in found_modules if m['id'] in modules_to_install]}"
            )
            module_obj.button_immediate_install(modules_to_install)
            log.info("Module installation process triggered successfully.")

        if modules_to_upgrade:
            log.info(
                f"Upgrading modules: "
                f"{[m['name'] for m in found_modules if m['id'] in modules_to_upgrade]}"
            )
            module_obj.button_immediate_upgrade(modules_to_upgrade)
            log.info("Module upgrade process triggered successfully.")

    except Exception as e:
        log.error(f"An error occurred during module operation: {e}")

    log.info("--- Module Installation Workflow Finished ---")


def run_module_uninstallation(
    config: str,
    modules: list[str],
) -> None:
    """Connects to Odoo and runs the module uninstallation process.

    Args:
        config: Path to the connection configuration file.
        modules: A list of technical module names to uninstall.
    """
    log.info(
        f"--- Starting Module Uninstallation Workflow for: {', '.join(modules)} ---"
    )

    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        module_obj = connection.get_model("ir.module.module")
    except Exception as e:
        log.error(f"Failed to connect to Odoo: {e}")
        return

    log.info(f"Searching for module(s) to uninstall: {modules}")
    module_ids = module_obj.search(
        [("name", "in", modules), ("state", "=", "installed")]
    )

    if not module_ids:
        log.warning("No matching installed modules found to uninstall.")
        return

    found_modules = module_obj.read(module_ids, ["name", "state"])
    log.info(f"Found modules to uninstall: {found_modules}")

    try:
        log.info("Triggering immediate uninstallation for found modules...")
        module_obj.button_immediate_uninstall(module_ids)
        log.info("Module uninstallation process triggered successfully.")
    except Exception as e:
        log.error(f"An error occurred during module uninstallation: {e}")

    log.info("--- Module Uninstallation Workflow Finished ---")
