"""This module provides shared user interface components, like panels."""

from rich.console import Console
from rich.panel import Panel


def _show_error_panel(title: str, message: str) -> None:
    """Displays a formatted error panel to the console."""
    console = Console(stderr=True, style="bold red")
    console.print(Panel(message, title=title, border_style="red"))


def _show_warning_panel(title: str, message: str) -> None:
    """Displays a formatted warning panel to the console."""
    console = Console(stderr=True, style="bold yellow")
    console.print(Panel(message, title=title, border_style="yellow"))
