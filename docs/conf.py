"""Sphinx configuration."""

project = "Odoo Data Flow"
author = "bosd"
copyright = "2025, bosd"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxmermaid",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
