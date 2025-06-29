"""Sphinx configuration."""

from sphinx.application import Sphinx  # type: ignore[import-not-found]

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
html_static_path = ["_static"]


def on_builder_inited(app: Sphinx) -> None:
    """This function is connected to the 'builder-inited' event.

    It removes the sphinx-mermaid extension if the builder is LaTeX, as it is
    not compatible with PDF output.
    """
    if app.builder.name == "latex":
        if "sphinx_mermaid" in extensions:
            extensions.remove("sphinx_mermaid")


# -- Setup function for builder-specific configuration ----------------------
def setup(app: Sphinx) -> None:
    """Called by Sphinx during the build process.

    We use this to disable extensions that are not compatible with certain
    builders, like LaTeX/PDF.
    """
    # The sphinx-mermaid extension is not compatible with the LaTeX builder,
    # so we remove it from the extensions list only when building for PDF.
    app.connect("builder-inited", on_builder_inited)
