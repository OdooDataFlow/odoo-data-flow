"""This file handles the mypyc compilation settings."""

import os

from mypyc.build import mypycify
from setuptools import setup


def get_ext_modules():
    """Conditionally builds mypyc extensions."""
    # If the environment variable is set, compile import_threaded.py
    if os.environ.get("ODF_COMPILE_MYPYC") == "1":
        print("Compiling 'import_threaded.py' and 'importer.py' with mypyc...")
        return mypycify(
            [
                "src/odoo_data_flow/import_threaded.py",
                "src/odoo_data_flow/importer.py",
                "src/odoo_data_flow/lib/mapper.py",
                "src/odoo_data_flow/export_threaded.py",
            ]
        )

    # Otherwise, return an empty list to build a pure Python package
    print("Skipping mypyc compilation. Building pure Python package...")
    return []


# This minimalist setup.py provides ONLY the C extension build instructions
# and the entry point for the command-line script.
setup(
    # Your other setup arguments like name, version, packages, etc., go here.
    # setuptools will automatically find most of this from pyproject.toml
    # if you have it configured.
    ext_modules=get_ext_modules(),
)
