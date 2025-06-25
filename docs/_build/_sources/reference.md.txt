# API Reference

This section provides an auto-generated API reference for the core components of the `odoo-data-flow` library.

## Command-Line Interface (`__main__`)

This module contains the main `click`-based command-line interface.

```{eval-rst}
.. click:: odoo_data_flow.__main__:cli
  :prog: odoo-data-flow
  :nested: full
```

## Transformation Processor (`lib.transform`)

This module contains the main `Processor` class used for data transformation.

```{eval-rst}
.. automodule:: odoo_data_flow.lib.transform
   :members: Processor
   :member-order: bysource
```

## Mapper Functions (`lib.mapper`)

This module contains all the built-in `mapper` functions for data transformation.

```{eval-rst}
.. automodule:: odoo_data_flow.lib.mapper
   :members:
   :undoc-members:
```

## High-Level Runners

These modules contain the high-level functions that are called by the CLI commands.

### Importer (`importer`)

```{eval-rst}
.. automodule:: odoo_data_flow.importer
   :members: run_import
```

### Exporter (`exporter`)

```{eval-rst}
.. automodule:: odoo_data_flow.exporter
   :members: run_export
```

### Migrator (`migrator`)

```{eval-rst}
.. automodule:: odoo_data_flow.migrator
   :members: run_migration
```
