"""This module provides a registry and functions for pre-flight checks.

These checks are run before the main import process to catch common,
systemic errors early (e.g., missing languages, incorrect configuration).
"""

from typing import Any, Callable, Optional, Union, cast

import polars as pl
from polars.exceptions import ColumnNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from odoo_data_flow.enums import PreflightMode

from ..logging_config import log
from . import cache, conf_lib, sort
from .actions import language_installer
from .internal.ui import _show_error_panel, _show_warning_panel

# A registry to hold all pre-flight check functions
PREFLIGHT_CHECKS: list[Callable[..., bool]] = []


def _handle_m2m_field(
    field_name: str,
    clean_field_name: str,
    field_info: dict[str, Any],
    df: pl.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    """Handle many2many field processing and strategy selection."""
    # Ensure the column is treated as string for splitting
    relation_count = (
        df.lazy()
        .select(pl.col(field_name).cast(pl.Utf8).str.split(","))
        .select(pl.col(field_name).list.len())
        .sum()
        .collect()
        .item()
    )
    # Check if required keys exist for many2many fields
    relation_table = field_info.get("relation_table")
    relation_field = field_info.get("relation_field")
    relation = field_info.get("relation")

    strategy_details = {}
    if relation_table and relation_field:
        if relation_count >= 500:
            strategy_details = {
                "strategy": "direct_relational_import",
                "relation_table": relation_table,
                "relation_field": relation_field,
                "relation": relation,
            }
        else:
            strategy_details = {
                "strategy": "write_tuple",
                "relation_table": relation_table,
                "relation_field": relation_field,
                "relation": relation,
            }
    else:
        # Log a warning when relation information is incomplete
        log.warning(
            f"Field '{clean_field_name}' is missing relation_table or relation_field "
            f"in Odoo metadata. This may cause issues with relational import."
        )
        # Fallback strategy when relation information is incomplete
        # Include whatever information we have, but don't set strategy to write_tuple
        # since we don't have the required fields for it
        strategy_details = {
            "strategy": "write_tuple",
            "relation_table": relation_table,
            "relation_field": relation_field,
            "relation": relation,
        }

    return True, strategy_details


def register_check(func: Callable[..., bool]) -> Callable[..., bool]:
    """A decorator to register a new pre-flight check function."""
    PREFLIGHT_CHECKS.append(func)
    return func


@register_check
def connection_check(
    preflight_mode: "PreflightMode", config: Union[str, dict[str, Any]], **kwargs: Any
) -> bool:
    """Pre-flight check to verify connection to Odoo."""
    log.info("Running pre-flight check: Verifying Odoo connection...")
    try:
        if isinstance(config, dict):
            conf_lib.get_connection_from_dict(config)
        else:
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


@register_check
def self_referencing_check(
    preflight_mode: "PreflightMode",
    filename: str,
    import_plan: dict[str, Any],
    **kwargs: Any,
) -> bool:
    """Detects self-referencing hierarchies and plans the sorting strategy."""
    if kwargs.get("o2m"):
        return True  # Skip this check if o2m is enabled

    log.info("Running pre-flight check: Detecting self-referencing hierarchy...")
    # We assume 'id' and 'parent_id' as conventional names.
    # This could be made configurable later if needed.
    result = sort.sort_for_self_referencing(
        filename,
        id_column="id",
        parent_column="parent_id",
        separator=kwargs.get("separator", ";"),
    )
    if result is False:
        # This means there was an error in sort_for_self_referencing
        # The error would have been displayed by the function itself
        return False
    elif result:
        # This means sorting was performed and we have a file path
        log.info(
            "Detected self-referencing hierarchy. Planning one-pass sort strategy."
        )
        import_plan["strategy"] = "sort_and_one_pass_load"
        import_plan["id_column"] = "id"
        import_plan["parent_column"] = "parent_id"
        return True
    else:
        # result is None, meaning no hierarchy detected
        log.info("No self-referencing hierarchy detected.")
        return True


def _get_installed_languages(config: Union[str, dict[str, Any]]) -> Optional[set[str]]:
    """Connects to Odoo and returns the set of installed language codes."""
    try:
        if isinstance(config, dict):
            connection = conf_lib.get_connection_from_dict(config)
        else:
            connection = conf_lib.get_connection_from_config(config)

        lang_obj = connection.get_model("res.lang")
        installed_langs_data = lang_obj.search_read([("active", "=", True)], ["code"])
        return {lang["code"] for lang in installed_langs_data}
    except Exception as e:
        error_message = str(e)
        title = "Odoo Connection Error"
        friendly_message = (
            "Could not fetch installed languages from Odoo. This usually means "
            "the connection details in your configuration file are incorrect.\n\n"
            "Please verify the following:\n"
            "  - [bold]hostname[/bold] is correct\n"
            "  - [bold]database[/bold] name is correct\n"
            "  - [bold]login[/bold] (username) is correct\n"
            "  - [bold]password[/bold] is correct\n\n"
            f"[bold]Original Error:[/bold] {error_message}"
        )
        _show_error_panel(title, friendly_message)
        return None


def _get_required_languages(filename: str, separator: str) -> Optional[list[str]]:
    """Extracts the list of required languages from the source file."""
    try:
        return (
            pl.read_csv(filename, separator=separator, truncate_ragged_lines=True)
            .get_column("lang")
            .unique()
            .drop_nulls()
            .to_list()
        )
    except ColumnNotFoundError:
        log.debug("No 'lang' column found in source file. Skipping language check.")
        return []
    except Exception as e:
        log.warning(
            f"Could not read languages from source file. Skipping check. Error: {e}"
        )
        return None


def _handle_missing_languages(
    config: Union[str, dict[str, Any]],
    missing_languages: set[str],
    headless: bool,
) -> bool:
    """Handles the process of installing missing languages."""
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
        if isinstance(config, dict):
            log.error("Language installation from a dict config is not supported.")
            return False
        return language_installer.run_language_installation(
            config, list(missing_languages)
        )

    if not Confirm.ask("Do you want to install them now?", default=True):
        log.warning("Language installation cancelled by user. Aborting import.")
        return False

    if isinstance(config, dict):
        log.error("Language installation from a dict config is not supported.")
        return False
    return language_installer.run_language_installation(config, list(missing_languages))


@register_check
def language_check(
    preflight_mode: PreflightMode,
    model: str,
    filename: str,
    config: Union[str, dict[str, Any]],
    headless: bool,
    **kwargs: Any,
) -> bool:
    """Pre-flight check to verify that all required languages are installed."""
    if preflight_mode == PreflightMode.FAIL_MODE or model not in (
        "res.partner",
        "res.users",
    ):
        log.debug("Skipping language pre-flight check.")
        return True

    log.info("Running pre-flight check: Verifying required languages...")

    required_languages = _get_required_languages(filename, kwargs.get("separator", ";"))
    if required_languages is None or not required_languages:
        return True

    installed_languages = _get_installed_languages(config)
    if installed_languages is None:
        return False

    missing_languages = set(required_languages) - installed_languages
    if not missing_languages:
        log.info("All required languages are installed.")
        return True

    return _handle_missing_languages(config, missing_languages, headless)


def _get_odoo_fields(
    config: Union[str, dict[str, Any]], model: str
) -> Optional[dict[str, Any]]:
    """Fetches the field schema for a given model from Odoo.

    Args:
        config: The path to the connection configuration file or a config dict.
        model: The target Odoo model name.

    Returns:
        A dictionary of the model's fields, or None on failure.
    """
    # 1. Try to load from cache first
    if isinstance(config, str):
        cached_fields = cache.load_fields_get_cache(config, model)
        if cached_fields:
            return cached_fields

    # 2. If cache miss, fetch from Odoo
    log.info(f"Cache miss for '{model}' fields, fetching from Odoo...")
    try:
        connection_obj: Any
        if isinstance(config, dict):
            connection_obj = conf_lib.get_connection_from_dict(config)
        else:
            connection_obj = conf_lib.get_connection_from_config(config_file=config)
        model_obj = connection_obj.get_model(model)
        odoo_fields = cast(dict[str, Any], model_obj.fields_get())

        # 3. Save the result to the cache for next time
        if isinstance(config, str):
            cache.save_fields_get_cache(config, model, odoo_fields)
        return odoo_fields
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


def _validate_header(
    csv_header: list[str], odoo_fields: dict[str, Any], model: str
) -> bool:
    """Validates that all CSV columns exist as fields on the Odoo model."""
    odoo_field_names = set(odoo_fields.keys())
    missing_fields = [
        field
        for field in csv_header
        if (field.split("/")[0] not in odoo_field_names) or (field.endswith("/.id"))
    ]

    if missing_fields:
        error_message = "The following columns do not exist on the Odoo model:\n"
        for field in missing_fields:
            error_message += f"  - '{field}' is not a valid field on model '{model}'\n"
        _show_error_panel("Invalid Fields Found", error_message)
        return False

    # Check for readonly fields that will be silently ignored
    readonly_fields = []
    for field in csv_header:
        clean_field = field.split("/")[
            0
        ]  # Handle external ID fields like 'parent_id/id'
        if clean_field in odoo_fields:
            field_info = odoo_fields[clean_field]
            is_readonly = field_info.get("readonly", False)
            is_stored = field_info.get(
                "store", True
            )  # Default to True for stored fields

            if is_readonly:
                readonly_fields.append(
                    {
                        "field": field,
                        "stored": is_stored,
                        "type": field_info.get("type", "unknown"),
                    }
                )

    # Warn about readonly fields, especially non-stored ones
    if readonly_fields:
        warning_message = (
            "The following readonly fields will be silently ignored during import:\n"
        )
        non_stored_count = 0
        for field_info in readonly_fields:
            storage_status = "non-stored" if not field_info["stored"] else "stored"
            if not field_info["stored"]:
                non_stored_count += 1
            warning_message += (
                f"  - '{field_info['field']}' "
                f"({storage_status} readonly {field_info['type']})\n"
            )

        if non_stored_count > 0:
            warning_message += (
                f"\n⚠️  {non_stored_count} non-stored readonly "
                f"fields will be completely ignored!\n"
            )
        warning_message += (
            "\nValues for these fields will be silently discarded during import."
        )
        _show_warning_panel("ReadOnly Fields Detected", warning_message)

    return True


def _plan_deferrals_and_strategies(
    header: list[str],
    odoo_fields: dict[str, Any],
    model: str,
    filename: str,
    separator: str,
    import_plan: dict[str, Any],
    **kwargs: Any,
) -> bool:
    """Analyzes fields to plan deferrals and select import strategies."""
    deferrable_fields = []
    strategies = {}
    df = pl.read_csv(filename, separator=separator, truncate_ragged_lines=True)

    for field_name in header:
        clean_field_name = field_name.replace("/id", "")
        if clean_field_name in odoo_fields:
            field_info = odoo_fields[clean_field_name]
            field_type = field_info.get("type")

            is_m2o_self = (
                field_type == "many2one" and field_info.get("relation") == model
            )
            is_m2m = field_type == "many2many"
            is_o2m = field_type == "one2many"

            if is_m2o_self:
                deferrable_fields.append(clean_field_name)
            elif is_m2m:
                deferrable_fields.append(clean_field_name)
                success, strategy_details = _handle_m2m_field(
                    field_name, clean_field_name, field_info, df
                )
                if success:
                    strategies[clean_field_name] = strategy_details
            elif is_o2m:
                deferrable_fields.append(clean_field_name)
                strategies[clean_field_name] = {"strategy": "write_o2m_tuple"}

    if deferrable_fields:
        log.info(f"Detected deferrable fields: {deferrable_fields}")
        unique_id_field = kwargs.get("unique_id_field")
        if not unique_id_field and "id" in header:
            log.info("Automatically using 'id' column as the unique identifier.")
            import_plan["unique_id_field"] = "id"
        elif not unique_id_field:
            _show_error_panel(
                "Action Required for Two-Pass Import",
                "Deferrable fields were detected, but no 'id' column was found.\n"
                "Please specify the unique ID column using the "
                "[bold cyan]--unique-id-field[/bold cyan] option.",
            )
            return False

        import_plan["deferred_fields"] = deferrable_fields
        import_plan["strategies"] = strategies
    return True


@register_check
def deferral_and_strategy_check(
    preflight_mode: "PreflightMode",
    model: str,
    filename: str,
    config: Union[str, dict[str, Any]],
    import_plan: dict[str, Any],
    **kwargs: Any,
) -> bool:
    """Verifies fields, detects deferrals, and plans import strategies."""
    log.info(f"Running pre-flight check: Verifying fields for model '{model}'...")
    separator = kwargs.get("separator", ";")
    csv_header = _get_csv_header(filename, separator)
    if not csv_header:
        return False

    ignore_list = kwargs.get("ignore", [])
    header_to_validate = [h for h in csv_header if h not in ignore_list]

    odoo_fields = _get_odoo_fields(config, model)
    if not odoo_fields:
        return False

    if not _validate_header(header_to_validate, odoo_fields, model):
        return False

    if preflight_mode == PreflightMode.FAIL_MODE:
        log.debug("Skipping deferral and strategy check in fail mode.")
        return True

    kwargs.pop("separator", None)
    if not _plan_deferrals_and_strategies(
        header_to_validate,
        odoo_fields,
        model,
        filename,
        separator,
        import_plan,
        **kwargs,
    ):
        return False

    log.info("Pre-flight Check Successful: All columns are valid fields on the model.")
    return True
