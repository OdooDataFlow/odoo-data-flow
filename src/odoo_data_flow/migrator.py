"""Migrate data between two odoo databases.

This module contains the logic for performing a direct, in-memory
migration of data from one Odoo instance to another.
"""

from .exporter import run_export_for_migration
from .importer import run_import_for_migration
from .lib.transform import Processor
from .logging_config import log


def run_migration(
    config_export,
    config_import,
    model,
    domain="[]",
    fields=None,
    mapping=None,
    export_worker=1,
    export_batch_size=100,
    import_worker=1,
    import_batch_size=10,
):
    """Performs a server-to-server data migration.

    This function chains together the export, transform, and import processes
    without creating intermediate files.
    """
    log.info("--- Starting Server-to-Server Migration ---")

    # Step 1: Export data from the source database
    log.info(f"Exporting data from model '{model}'...")
    header, data = run_export_for_migration(
        config=config_export,
        model=model,
        domain=domain,
        fields=fields,
        worker=export_worker,
        batch_size=export_batch_size,
    )

    if not data:
        log.warning("No data exported. Migration finished.")
        return

    log.info(f"Successfully exported {len(data)} records.")

    # Step 2: Transform the data in memory
    log.info("Transforming data in memory...")
    processor = Processor(header=header, data=data)

    if not mapping:
        log.info("No mapping provided, using 1-to-1 mapping.")
        mapping = processor.get_o2o_mapping()

    # The process method returns the transformed header and data
    to_import_header, to_import_data = processor.process(mapping, filename=None)

    # Step 3: Import the transformed data into the destination database
    log.info(f"Importing {len(to_import_data)} records into destination...")
    run_import_for_migration(
        config=config_import,
        model=model,
        header=to_import_header,
        data=to_import_data,
        worker=import_worker,
        batch_size=import_batch_size,
    )

    log.info("--- Migration Finished Successfully ---")
