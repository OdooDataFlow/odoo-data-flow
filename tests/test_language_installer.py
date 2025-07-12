"""Test the language installation workflow."""

from unittest.mock import MagicMock, call, patch

from odoo_data_flow.lib.actions.language_installer import (
    run_language_installation,
)


@patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
@patch(
    "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
)
def test_run_language_installation_success(
    mock_get_connection: MagicMock, mock_get_odoo_version: MagicMock
) -> None:
    """Test successful language installation."""
    # 1. Setup
    # Mock the version detection to simulate a legacy environment
    mock_get_odoo_version.return_value = 14

    mock_wizard_obj = MagicMock()
    mock_wizard_obj.create.return_value = 123  # Dummy wizard ID

    mock_connection = MagicMock()
    # IMPORTANT: We only mock the return value for the 'base.language.install' call
    mock_connection.get_model.return_value = mock_wizard_obj
    mock_get_connection.return_value = mock_connection

    languages_to_install = ["nl_BE", "fr_FR"]

    # 2. Action
    run_language_installation(config="dummy.conf", languages=languages_to_install)

    # 3. Assertions
    # Check that we connected and detected the version
    mock_get_connection.assert_called_once_with(config_file="dummy.conf")
    mock_get_odoo_version.assert_called_once_with(mock_connection)

    # Check that we retrieved the correct model for the wizard
    mock_connection.get_model.assert_called_once_with("base.language.install")

    # Check that the wizard was created and executed for each language
    expected_create_calls = [call({"lang": "nl_BE"}), call({"lang": "fr_FR"})]
    mock_wizard_obj.create.assert_has_calls(expected_create_calls, any_order=True)
    assert mock_wizard_obj.create.call_count == 2
    mock_wizard_obj.lang_install.assert_has_calls(
        [call([123]), call([123])], any_order=True
    )


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
@patch("odoo_data_flow.lib.actions.language_installer.odoo_lib.get_odoo_version")
@patch(
    "odoo_data_flow.lib.actions.language_installer.conf_lib.get_connection_from_config"
)
def test_run_language_installation_api_error(
    mock_get_connection: MagicMock,
    mock_get_odoo_version: MagicMock,
    mock_log_error: MagicMock,
) -> None:
    """Tests error handling when the Odoo API call for installation fails."""
    # 1. Setup
    # Mock the version detection to simulate a legacy environment
    mock_get_odoo_version.return_value = 14

    mock_wizard_obj = MagicMock()
    # Simulate an error when trying to run the installation
    mock_wizard_obj.lang_install.side_effect = Exception("Odoo API Error")

    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_wizard_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    run_language_installation(config="dummy.conf", languages=["en_US"])

    # 3. Assertions
    # Check that the new, more specific error message was logged
    mock_log_error.assert_called_once()
    logged_message = mock_log_error.call_args[0][0]
    assert "Failed to install language 'en_US'" in logged_message
    assert "Odoo API Error" in logged_message
