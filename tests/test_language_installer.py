"""Test the language installation workflow."""

from unittest.mock import MagicMock, patch

from odoo_data_flow.lib.actions.language_installer import run_language_installation


@patch(
    "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
)
def test_run_language_installation_success(mock_get_connection: MagicMock) -> None:
    """Test Succesfull language installation.

    Tests that the language installation workflow calls the correct
    Odoo wizard and methods with the correct parameters.
    """
    # 1. Setup
    # Mock the Odoo wizard object and its methods
    mock_wizard_obj = MagicMock()
    # The create method returns a dummy wizard ID
    mock_wizard_obj.create.return_value = 123

    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_wizard_obj
    mock_get_connection.return_value = mock_connection

    languages_to_install = ["nl_BE", "fr_FR"]

    # 2. Action
    run_language_installation(config="dummy.conf", languages=languages_to_install)

    # 3. Assertions
    mock_get_connection.assert_called_once_with(config_file="dummy.conf")
    mock_connection.get_model.assert_called_once_with("base.language.install")

    # Verify that the wizard was created with the correct language string
    expected_create_data = {"lang": "nl_BE,fr_FR"}
    mock_wizard_obj.create.assert_called_once_with(expected_create_data)

    # Verify that the install method was called with the wizard's ID
    mock_wizard_obj.lang_install.assert_called_once_with([123])


@patch("odoo_data_flow.lib.actions.language_installer.log.error")
@patch(
    "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
)
def test_run_language_installation_connection_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that an error is logged if the Odoo connection fails."""
    # 1. Setup
    mock_get_connection.side_effect = Exception("Connection Refused")

    # 2. Action
    run_language_installation(config="bad.conf", languages=["en_US"])

    # 3. Assertions
    mock_log_error.assert_called_once()
    assert "Failed to connect to Odoo" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.actions.language_installer.log.error")
@patch(
    "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
)
def test_run_language_installation_api_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests error handling when the Odoo API call for installation fails."""
    # 1. Setup
    mock_wizard_obj = MagicMock()
    # Simulate an error when trying to run the installation
    mock_wizard_obj.lang_install.side_effect = Exception("Odoo API Error")
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_wizard_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    run_language_installation(config="dummy.conf", languages=["en_US"])

    # 3. Assertions
    mock_log_error.assert_called_once()
    assert (
        "An error occurred during language installation"
        in mock_log_error.call_args[0][0]
    )
