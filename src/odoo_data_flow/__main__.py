"""Command-line interface for odoo-data-flow."""

import ast
from typing import Any

import click

from .converter import run_path_to_image, run_url_to_image
from .exporter import run_export
from .importer import run_import
from .logging_config import setup_logging
from .migrator import run_migration
from .workflow_runner import run_invoice_v9_workflow


@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.version_option()
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable verbose, debug-level logging."
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Odoo Data Flow: A tool for importing, exporting, and processing data."""
    setup_logging(verbose)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# --- Workflow Command ---
@cli.command(name="workflow-invoice-v9")
@click.option("-c", "--config", required=True, help="Path to the connection.conf file.")
@click.option(
    "--action",
    "actions",
    multiple=True,
    type=click.Choice(
        ["tax", "validate", "pay", "proforma", "rename", "all"],
        case_sensitive=False,
    ),
    default=["all"],
    help="Workflow action to run. Can be specified multiple times. Defaults to 'all'.",
)
@click.option(
    "--field",
    required=True,
    help="The source field containing the legacy invoice status.",
)
@click.option(
    "--status-map",
    "status_map_str",
    required=True,
    help="Dictionary string mapping Odoo states to legacy states. "
    "e.g., \"{'open': ['OP']}\"",
)
@click.option(
    "--paid-date-field",
    required=True,
    help="The source field containing the payment date.",
)
@click.option(
    "--payment-journal",
    required=True,
    type=int,
    help="The database ID of the payment journal.",
)
@click.option(
    "--max-connection", default=4, type=int, help="Number of parallel threads."
)
def invoice_v9_cmd(**kwargs: Any) -> None:
    """Runs the legacy Odoo v9 invoice processing workflow."""
    run_invoice_v9_workflow(**kwargs)


# --- Import Command ---
@cli.command(name="import")
@click.option(
    "-c",
    "--config",
    required=True,
    help="Configuration file for connection parameters.",
)
@click.option("--file", "filename", required=True, help="File to import.")
@click.option("--model", required=True, help="Odoo model to import into.")
@click.option(
    "--worker", default=1, type=int, help="Number of simultaneous connections."
)
@click.option(
    "--size",
    "batch_size",
    default=10,
    type=int,
    help="Number of lines to import per connection.",
)
@click.option("--skip", default=0, type=int, help="Number of initial lines to skip.")
@click.option(
    "--fail",
    is_flag=True,
    default=False,
    help="Run in fail mode, retrying records from the .fail file.",
)
@click.option("-s", "--sep", "separator", default=";", help="CSV separator character.")
@click.option(
    "--groupby",
    "split",
    default=None,
    help="Column to group data by to avoid concurrent updates.",
)
@click.option(
    "--ignore", default=None, help="Comma-separated list of columns to ignore."
)
@click.option(
    "--check",
    is_flag=True,
    default=False,
    help="Check if records are imported after each batch.",
)
@click.option(
    "--context",
    default="{'tracking_disable': True}",
    help="Odoo context as a dictionary string.",
)
@click.option(
    "--o2m",
    is_flag=True,
    default=False,
    help="Special handling for one-to-many imports.",
)
@click.option("--encoding", default="utf-8", help="Encoding of the data file.")
def import_cmd(**kwargs: Any) -> None:
    """Runs the data import process."""
    run_import(**kwargs)


# --- Export Command ---
@cli.command(name="export")
@click.option(
    "-c",
    "--config",
    required=True,
    help="Configuration file for connection parameters.",
)
@click.option("--file", "filename", required=True, help="Output file path.")
@click.option("--model", required=True, help="Odoo model to export from.")
@click.option(
    "--fields", required=True, help="Comma-separated list of fields to export."
)
@click.option("--domain", default="[]", help="Odoo domain filter as a list string.")
@click.option(
    "--worker", default=1, type=int, help="Number of simultaneous connections."
)
@click.option(
    "--size",
    "batch_size",
    default=10,
    type=int,
    help="Number of records to process per batch.",
)
@click.option("-s", "--sep", "separator", default=";", help="CSV separator character.")
@click.option(
    "--context",
    default="{'tracking_disable': True}",
    help="Odoo context as a dictionary string.",
)
@click.option("--encoding", default="utf-8", help="Encoding of the data file.")
def export_cmd(**kwargs: Any) -> None:
    """Runs the data export process."""
    run_export(**kwargs)


# --- Path-to-Image Command ---
@cli.command(name="path-to-image")
@click.argument("file")
@click.option(
    "-f",
    "--fields",
    required=True,
    help="Comma-separated list of fields to convert from path to base64.",
)
@click.option(
    "--path",
    default=None,
    help="Image path prefix. Defaults to the current working directory.",
)
@click.option("--out", default="out.csv", help="Name of the resulting output file.")
def path_to_image_cmd(**kwargs: Any) -> None:
    """Converts columns with local file paths into base64 strings."""
    run_path_to_image(**kwargs)


# --- URL-to-Image Command ---
@cli.command(name="url-to-image")
@click.argument("file")
@click.option(
    "-f",
    "--fields",
    required=True,
    help="Comma-separated list of fields with URLs to convert to base64.",
)
@click.option("--out", default="out.csv", help="Name of the resulting output file.")
def url_to_image_cmd(**kwargs: Any) -> None:
    """Downloads content from URLs in columns and converts to base64."""
    run_url_to_image(**kwargs)


# --- Migrate Command ---
@cli.command(name="migrate")
@click.option(
    "--config-export",
    required=True,
    help="Path to the source Odoo connection config.",
)
@click.option(
    "--config-import",
    required=True,
    help="Path to the destination Odoo connection config.",
)
@click.option("--model", required=True, help="The Odoo model to migrate.")
@click.option(
    "--domain", default="[]", help="Domain filter to select records for export."
)
@click.option(
    "--fields", required=True, help="Comma-separated list of fields to migrate."
)
@click.option(
    "--mapping",
    default=None,
    help="A dictionary string defining the transformation mapping.",
)
@click.option(
    "--export-worker",
    default=1,
    type=int,
    help="Number of workers for the export phase.",
)
@click.option(
    "--export-batch-size",
    default=100,
    type=int,
    help="Batch size for the export phase.",
)
@click.option(
    "--import-worker",
    default=1,
    type=int,
    help="Number of workers for the import phase.",
)
@click.option(
    "--import-batch-size",
    default=10,
    type=int,
    help="Batch size for the import phase.",
)
def migrate_cmd(**kwargs: Any) -> None:
    """Performs a direct server-to-server data migration."""
    if kwargs.get("mapping"):
        try:
            kwargs["mapping"] = ast.literal_eval(kwargs["mapping"])
        except Exception as e:
            print(
                "Error: Invalid mapping provided. "
                f"Must be a valid Python dictionary string. Error: {e}"
            )
            return
    run_migration(**kwargs)


if __name__ == "__main__":
    cli()
