"""Test the shared user interface components."""

from unittest.mock import MagicMock, patch

from rich.panel import Panel

from odoo_data_flow.lib.internal.ui import _show_error_panel


@patch("odoo_data_flow.lib.internal.ui.Console")
def test_show_error_panel(mock_console_class: MagicMock) -> None:
    """Tests that the _show_error_panel function.

    Tests that the _show_error_panel function correctly calls rich.Console
    and rich.Panel with the expected arguments.
    """
    # 1. Setup
    mock_console_instance = mock_console_class.return_value
    test_title = "Test Error"
    test_message = "This is a test error message."

    # 2. Action
    _show_error_panel(test_title, test_message)

    # 3. Assertions
    # Check that Console was initialized correctly for stderr output
    mock_console_class.assert_called_once_with(stderr=True, style="bold red")

    # Check that console.print was called once
    mock_console_instance.print.assert_called_once()

    # Check the properties of the Panel object that was passed to print
    call_args = mock_console_instance.print.call_args
    panel_instance = call_args[0][0]
    assert isinstance(panel_instance, Panel)
    assert panel_instance.title == test_title
    assert panel_instance.renderable == test_message
    assert panel_instance.border_style == "red"
