"""This module provides a registry and functions for pre-flight checks.

These checks are run before the main import process to catch common,
systemic errors early (e.g., missing languages, incorrect configuration).
"""

from typing import Any, Callable

import polars as pl
from polars.exceptions import ColumnNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from odoo_data_flow.enums import PreflightMode

from ..logging_config import log
from . import conf_lib
from .actions import language_installer
from .internal.ui import _show_error_panel

# A registry to hold all pre-flight check functions
PREFLIGHT_CHECKS: list[Callable[..., bool]] = []


def register_check(func: Callable[..., bool]) -> Callable[..., bool]:
    """A decorator to register a new pre-flight check function."""
    PREFLIGHT_CHECKS.append(func)
    return func


@register_check
def connection_check(
    preflight_mode: "PreflightMode", config: str, **kwargs: Any
) -> bool:
    """Pre-flight check to verify connection to Odoo."""
    log.info("Running pre-flight check: Verifying Odoo connection...")
    try:
        # This line implicitly checks the connection
        conf_lib.get_connection_from_config(config_file=config)
        log.info("Connection to Odoo successful.")
        return True
    except Exception as e:
        _show_error_panel(
            "Odoo Connection Error",
            f"Could not establish connection to Odoo. "
            f"Please check your configuration.\nError: {e}",
        )
        return False


def _get_installed_languages(config_file: str) -> set[str]:
    """Connects to Odoo and returns the set of installed language codes."""
    try:
        connection = conf_lib.get_connection_from_config(config_file)
        lang_obj = connection.get_model("res.lang")
        installed_langs_data = lang_obj.search_read([("active", "=", True)], ["code"])
        return {lang["code"] for lang in installed_langs_data}
    except Exception as e:
        log.error(f"Could not fetch installed languages from Odoo. Error: {e}")
        return set()


@register_check
def language_check(
    preflight_mode: PreflightMode,
    model: str,
    filename: str,
    config: str,
    headless: bool,
    **kwargs: Any,
) -> bool:
    """Pre-flight check to verify that all required languages are installed.

    Scans the 'lang' column for `res.partner` and `res.users` imports.
    """
    if model not in ("res.partner", "res.users"):
        return True

    log.info("Running pre-flight check: Verifying required languages...")

    try:
        required_languages = (
            pl.read_csv(filename, separator=kwargs.get("separator", ";"))
            .get_column("lang")
            .unique()
            .drop_nulls()
            .to_list()
        )
    except ColumnNotFoundError:
        log.debug("No 'lang' column found in source file. Skipping language check.")
        return True
    except Exception as e:
        log.warning(
            f"Could not read languages from source file. Skipping check. Error: {e}"
        )
        return True

    if not required_languages:
        return True

    installed_languages = _get_installed_languages(config)
    missing_languages = set(required_languages) - installed_languages

    if not missing_languages:
        log.info("All required languages are installed.")
        return True
    if preflight_mode == PreflightMode.FAIL_MODE:
        log.warning(
            f"Fail mode: Missing languages detected "
            f"({', '.join(sorted(list(missing_languages)))}). "
            f"Language installation will be skipped. Proceeding with import, "
            f"but errors may occur."
        )
        return True  # Allow import to continue in fail mode
    else:  # NORMAL Mode
        console = Console(stderr=True, style="bold yellow")
        message = (
            "The following required languages are not installed in the target "
            f"database:\n\n"
            f"[bold red]{', '.join(sorted(list(missing_languages)))}[/bold red]"
            f"\n\nThis is likely to cause the import to fail."
        )
        console.print(
            Panel(
                message,
                title="Missing Languages Detected",
                border_style="yellow",
            )
        )

    if headless:
        log.info("--headless mode detected. Auto-confirming language installation.")
        return language_installer.run_language_installation(
            config, list(missing_languages)
        )

    proceed = Confirm.ask("Do you want to install them now?", default=True)
    if proceed:
        return language_installer.run_language_installation(
            config, list(missing_languages)
        )
    else:
        log.warning("Language installation cancelled by user. Aborting import.")
        return False


@register_check
def field_existence_check(
    preflight_mode: "PreflightMode",
    model: str,
    filename: str,
    config: str,
    **kwargs: Any,
) -> bool:
    """Preflight check.

    Pre-flight check to verify that all columns in the header exist as
    fields on the target Odoo model.
    """
    log.info(f"Running pre-flight check: Verifying fields for model '{model}'...")

    try:
        csv_header = pl.read_csv(
            filename, separator=kwargs.get("separator", ";"), n_rows=0
        ).columns
    except Exception as e:
        _show_error_panel("File Read Error", f"Could not read CSV header. Error: {e}")
        return False

    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        model_obj = connection.get_model(model)
        odoo_fields = model_obj.fields_get()
        odoo_field_names = set(odoo_fields.keys())

    except Exception as e:
        _show_error_panel(
            "Odoo Connection Error",
            f"Could not connect to Odoo to get model fields. Error: {e}",
        )
        return False

    # Add a check to specifically reject the export-only '/.id' syntax.
    missing_fields = [
        field
        for field in csv_header
        if (field.split("/")[0] not in odoo_field_names) or (field.endswith("/.id"))
    ]

    if missing_fields:
        error_message = (
            "The following columns in your import file do not exist "
            "on the Odoo model:\n"
        )
        for field in missing_fields:
            error_message += f"  - '{field}' is not a valid field on model '{model}'\n"
        _show_error_panel("Invalid Fields Found", error_message)
        return False

    log.info("Pre-flight Check Successful: All columns are valid fields on the model.")
    return True
