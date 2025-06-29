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
    "sphinx_copybutton",
]
autodoc_typehints = "description"
html_theme = "shibuya"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = "_static/icon.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# -- Logic to disable mermaid for LaTeX output -----------------------------
# The 'tags' object is automatically provided by Sphinx.
tags = globals().get("tags")
if tags and tags.has("latex"):
    # The sphinx-mermaid extension is not compatible with the LaTeX builder
    extensions.remove("sphinx_mermaid")
