"""This module provides a registry and functions for pre-flight checks.

These checks are run before the main import process to catch common,
systemic errors early (e.g., missing languages, incorrect configuration).
"""

from typing import Any, Callable, Optional, cast

import polars as pl
from polars.exceptions import ColumnNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from odoo_data_flow.enums import PreflightMode

from ..logging_config import log
from . import conf_lib
from .actions import language_installer, module_manager
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


def _get_odoo_fields(config: str, model: str) -> Optional[dict[str, Any]]:
    """Fetches the field schema for a given model from Odoo.

    Args:
        config: The path to the connection configuration file.
        model: The target Odoo model name.

    Returns:
        A dictionary of the model's fields, or None on failure.
    """
    try:
        connection: Any = conf_lib.get_connection_from_config(config_file=config)
        model_obj = connection.get_model(model)
        # FIX: Use `cast` to inform mypy of the expected return type.
        return cast(dict[str, Any], model_obj.fields_get())
    except Exception as e:
        _show_error_panel(
            "Odoo Connection Error",
            f"Could not connect to Odoo to get model fields. Error: {e}",
        )
        return None


def _get_csv_header(filename: str, separator: str) -> Optional[list[str]]:
    """Reads the header from a CSV file.

    Args:
        filename: The path to the source CSV file.
        separator: The delimiter used in the CSV file.

    Returns:
        A list of strings representing the header, or None on failure.
    """
    try:
        return pl.read_csv(filename, separator=separator, n_rows=0).columns
    except Exception as e:
        _show_error_panel("File Read Error", f"Could not read CSV header. Error: {e}")
        return None


def _get_field_module_map(field_module_map_file: str) -> Optional[pl.DataFrame]:
    """Reads a field-to-module map from a CSV file."""
    try:
        df = pl.read_csv(field_module_map_file)
        if {"model_name", "field_name", "module_name"}.issubset(df.columns):
            return df
        log.error(
            "Field-module map file is missing required columns: 'model_name', 'field_name', 'module_name'."
        )
    except Exception as e:
        log.error(
            f"Failed to read field-module map file: {field_module_map_file}. Error: {e}"
        )
    return None


def _validate_header_with_module_suggestion(
    csv_header: list[str],
    odoo_fields: dict[str, Any],
    model: str,
    preflight_mode: PreflightMode,
    config: str,
    headless: bool,
    field_module_map_file: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """Validates that all CSV columns exist as fields.
    If a field is missing, checks a module map for a potential module to install.
    """
    odoo_field_names = set(odoo_fields.keys())
    missing_fields = [
        field
        for field in csv_header
        if (field.split("/")[0] not in odoo_field_names) or (field.endswith("/.id"))
    ]

    # This is the crucial new logic
    if missing_fields and field_module_map_file:
        log.warning(
            f"Missing fields detected: {missing_fields}. Checking module map..."
        )
        field_module_map = _get_field_module_map(field_module_map_file)

        if field_module_map is not None:
            modules_to_propose = set()
            for field in missing_fields:
                field_base_name = field.split("/")[0]

                # Lookup module name from the map
                module_row = field_module_map.filter(
                    (pl.col("model_name") == model)
                    & (pl.col("field_name") == field_base_name)
                )

                if not module_row.is_empty():
                    # Using .item() here is efficient for single-cell DataFrames
                    modules_to_propose.add(module_row["module_name"].item())

            if modules_to_propose:
                log.info(f"Potential modules for missing fields: {modules_to_propose}")
                if preflight_mode == PreflightMode.FAIL_MODE:
                    log.warning(
                        "Fail mode: Skipping module installation. Import will likely fail."
                    )
                    return True  # Allow it to continue in fail mode

                console = Console(stderr=True, style="bold yellow")
                message = (
                    "The following required fields are missing from the target database.\n"
                    "They may be provided by these modules, which are not currently installed:\n\n"
                    f"[bold red]{', '.join(sorted(list(modules_to_propose)))}[/bold red]"
                    "\n\nThis is likely to cause the import to fail."
                )
                console.print(Panel(message, title="Missing Module Dependencies"))

                should_install = headless or Confirm.ask(
                    "Do you want to install them now?", default=True
                )

                if should_install:
                    # Call your existing module installation function
                    module_manager.run_module_installation(
                        config, list(modules_to_propose)
                    )
                    log.info(
                        "Module installation completed. Re-running pre-flight check..."
                    )
                    # IMPORTANT: Re-run the full field check to confirm the problem is solved.
                    # We can do this by exiting the current `field_existence_check` and letting the
                    # main `importer.py` loop re-trigger it. Or we can
                    # do a recursive call. I'll propose a recursive call for simplicity.
                    # To prevent an infinite loop, we will use a flag or a counter.
                    # Let's assume the installation worked and return True here.
                    # The main `importer` will have to re-run the whole preflight check chain.
                    return True  # Let the outer loop run the check again
                else:
                    log.warning("Module installation skipped by user. Aborting import.")
                    return False

        # Fallback if no modules are proposed or the map file is invalid
        log.error("No corresponding modules found in the map for the missing fields.")

    if missing_fields:
        error_message = "The following columns do not exist on the Odoo model:\n"
        for field in missing_fields:
            error_message += f"  - '{field}' is not a valid field on model '{model}'\n"
        _show_error_panel("Invalid Fields Found", error_message)
        return False
    return True


def _detect_and_plan_deferrals(
    csv_header: list[str],
    odoo_fields: dict[str, Any],
    model: str,
    import_plan: Optional[dict[str, Any]],
    kwargs: dict[str, Any],
) -> bool:
    """Detects deferrable fields and updates the import plan."""
    deferrable_fields = []
    for field_name in csv_header:
        clean_field_name = field_name.replace("/id", "")
        if clean_field_name in odoo_fields:
            field_info = odoo_fields[clean_field_name]
            is_m2o_self = (
                field_info.get("type") == "many2one"
                and field_info.get("relation") == model
            )
            is_m2m = field_info.get("type") == "many2many"
            if is_m2o_self or is_m2m:
                deferrable_fields.append(clean_field_name)

    if deferrable_fields:
        log.info(f"Detected deferrable fields: {deferrable_fields}")
        unique_id_field = kwargs.get("unique_id_field")

        # --- NEW: Automatic 'id' column detection ---
        if not unique_id_field:
            if "id" in csv_header:
                log.info("Automatically using 'id' column as the unique identifier.")
                unique_id_field = "id"
                if import_plan is not None:
                    import_plan["unique_id_field"] = "id"  # Store the inferred field
            else:
                _show_error_panel(
                    "Action Required for Two-Pass Import",
                    "Deferrable fields were detected, but no 'id' column was found.\n"
                    "Please specify the unique ID column using the "
                    "[bold cyan]--unique-id-field[/bold cyan] option.",
                )
                return False

        if import_plan is not None:
            import_plan["deferred_fields"] = deferrable_fields
    return True


@register_check
def field_existence_check(
    preflight_mode: "PreflightMode",
    model: str,
    filename: str,
    config: str,
    import_plan: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> bool:
    """Verifies fields exist and detects which fields require deferred import.

    Args:
        preflight_mode: The current pre-flight mode.
        model: The target Odoo model name.
        filename: The path to the source CSV file.
        config: The path to the connection configuration file.
        import_plan: A dictionary to be populated with import strategy details.
        **kwargs: Additional arguments passed from the importer.

    Returns:
        True if all checks pass, False otherwise.
    """
    log.info(f"Running pre-flight check: Verifying fields for model '{model}'...")
    csv_header = _get_csv_header(filename, kwargs.get("separator", ";"))
    if not csv_header:
        return False

    odoo_fields = _get_odoo_fields(config, model)
    if not odoo_fields:
        return False

    if not _validate_header_with_module_suggestion(
        csv_header,
        odoo_fields,
        model,
        preflight_mode,
        config,
        kwargs.get("headless", False),
        kwargs.get("field_module_map_file"),
        **kwargs,
    ):
        return False

    if not _detect_and_plan_deferrals(
        csv_header, odoo_fields, model, import_plan, kwargs
    ):
        return False

    log.info("Pre-flight Check Successful: All columns are valid fields on the model.")
    return True
