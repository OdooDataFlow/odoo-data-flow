<p align="center">
  <img src="https://raw.githubusercontent.com/OdooDataFlow/odoo-data-flow/master/docs/_static/icon.png" width="200">
</p>

# Odoo Data Flow

[![PyPI](https://img.shields.io/pypi/v/odoo-data-flow.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/odoo-data-flow.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/odoo-data-flow)][pypi status]
[![License](https://img.shields.io/pypi/l/odoo-data-flow)][license]

[![Read the documentation at https://odoodataflow.readthedocs.io/](https://img.shields.io/readthedocs/odoodataflow/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/OdooDataFlow/odoo-data-flow/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/OdooDataFlow/odoo-data-flow/branch/master/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff codestyle][ruff badge]][ruff project]

[pypi status]: https://pypi.org/project/odoo-data-flow/
[read the docs]: https://odoodataflow.readthedocs.io/
[tests]: https://github.com/OdooDataFlow/odoo-data-flow/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/OdooDataFlow/odoo-data-flow
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff project]: https://github.com/charliermarsh/ruff

A powerful Python library for defining robust, repeatable, and high-performance data import/export workflows for Odoo. It replaces complex, manual data preparation with a clean, "configuration-as-code" approach.

---

## Key Features

- **Declarative Transformations:** Use simple Python scripts and a rich set of `mapper` functions to transform any source CSV or XML data into an Odoo-ready format.
- **Two-Phase Workflow:** Cleanly separates data **transformation** from data **loading**, making complex migrations easier to manage, reuse, and debug.
- **High-Performance CLI:** Import and export data with a clean, modern command-line interface, featuring high performance parallel processing (`--worker`), batching (`--size`), and robust error handling.
- **Automatic Scripting:** Automatically generate shell scripts for the loading phase, ensuring a repeatable and reliable process every time.
- **Robust Error Handling and Recovery:** Verify the number of records processed in a batch against the number successfully imported, helping to quickly identify issues.
- **Direct Server-to-Server Migration:** Perform a complete export, transform, and import from one Odoo instance to another in a single, in-memory step with the `migrate` command.
- **Post-Import Workflows:** Run automated actions on your data _after_ it has been imported (e.g., validating invoices, registering payments) using the powerful `workflow` command.
- **High-Performance Streaming Exports:** Export massive datasets from Odoo with confidence. The export process uses a streaming pipeline to keep memory usage low, and leverages the high-performance Polars engine for multi-threaded data cleaning and CSV writing. It intelligently handles data type issues, such as converting Odoo's `False` values to empty strings for non-boolean fields, ensuring your exported data is clean and accurate.
- **Multiple Data Sources**: Natively supports CSV and XML files. Easily extendable to support other sources like databases or APIs.
- **Data Validation:** Ensure data integrity before it even reaches Odoo.


## Installation

You can install _Odoo Data Flow_ via `uv` or `pip` from [PyPI]:

```console
$ uv pip install odoo-data-flow
```

## Quick Usage Example

The core workflow involves two simple steps:

**1. Transform your source data with a Python script.**
Create a `transform.py` file to define the mapping from your source file to Odoo's format.

```python
# transform.py
from odoo_data_flow.lib.transform import Processor
from odoo_data_flow.lib import mapper

my_mapping = {
    'id': mapper.concat('prod_', 'SKU'),
    'name': mapper.val('ProductName'),
    'list_price': mapper.num('Price'),
}

processor = Processor('origin/products.csv')
processor.process(my_mapping, 'data/products_clean.csv', {'model': 'product.product'})
processor.write_to_file("load.sh")
```
...
```console
$ python transform.py
```

**2. Load the clean data into Odoo using the CLI.**
The `transform.py` script generates a `load.sh` file containing the correct CLI command.

```bash
# Contents of the generated load.sh
odoo-data-flow import --config conf/connection.conf --file data/products_clean.csv --model product.product ...
```
Then execute the script.
```console
$ bash load.sh
```

## Documentation

For a complete user guide, tutorials, and API reference, please see the **[full documentation on Read the Docs][read the docs]**.
Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [LGPL 3.0 license][license],
_Odoo Data Flow_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This development of project is financially supported by [stefcy.com].
This project was generated from [@bosd]'s [uv hypermodern python cookiecutter] template.

[stefcy.com]: https://stefcy.com
[@bosd]: https://github.com/bosd
[pypi]: https://pypi.org/
[uv hypermodern python cookiecutter]: https://github.com/bosd/cookiecutter-uv-hypermodern-python
[file an issue]: https://github.com/OdooDataFlow/odoo-data-flow/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/OdooDataFlow/odoo-data-flow/blob/main/LICENSE
[contributor guide]: https://github.com/OdooDataFlow/odoo-data-flow/blob/main/CONTRIBUTING.md
[command-line reference]: https://odoo-data-flow.readthedocs.io/en/latest/usage.html
