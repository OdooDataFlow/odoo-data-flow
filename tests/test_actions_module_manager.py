"""Test the module management workflow."""

from unittest.mock import MagicMock, patch

from odoo_data_flow.lib.actions.module_manager import (
    run_module_installation,
    run_module_uninstallation,
    run_update_module_list,
)


@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_update_module_list(mock_get_connection: MagicMock) -> None:
    """Test the update module list action.

    Tests that the 'Update Apps List' workflow calls the correct methods
    in the correct order to ensure a synchronous update.
    """
    # 1. Setup
    mock_module_obj = MagicMock()
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    result = run_update_module_list(config="dummy.conf")

    # 3. Assertions
    assert result is True
    mock_module_obj.update_list.assert_called_once()
    mock_module_obj.clear_caches.assert_called_once()
    mock_module_obj.search_count.assert_called_once_with([])


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_update_module_list_connection_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that a connection error is handled gracefully."""
    mock_get_connection.side_effect = Exception("Connection Failed")
    result = run_update_module_list(config="bad.conf")
    assert result is False
    mock_log_error.assert_called_once()
    assert "Failed to connect to Odoo" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_update_module_list_api_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that an API error during update_list is handled."""
    mock_module_obj = MagicMock()
    mock_module_obj.update_list.side_effect = Exception("Odoo API Error")
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    result = run_update_module_list(config="dummy.conf")
    assert result is False
    mock_log_error.assert_called_once()
    assert (
        "An error occurred during 'Update Apps List'" in mock_log_error.call_args[0][0]
    )


@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_install_and_upgrade(
    mock_get_connection: MagicMock,
) -> None:
    """Test Install and upgrade mix.

    Tests that the workflow correctly identifies modules
    to install vs. upgrade and calls the appropriate methods.
    """
    # 1. Setup
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [1, 2, 3]
    mock_module_obj.read.return_value = [
        {"id": 1, "name": "module_to_install", "state": "uninstalled"},
        {"id": 2, "name": "module_to_upgrade", "state": "installed"},
        {"id": 3, "name": "another_to_upgrade", "state": "installed"},
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    # 2. Action
    run_module_installation(
        config="dummy.conf",
        modules=["module_to_install", "module_to_upgrade", "another_to_upgrade"],
    )

    # 3. Assertions
    mock_module_obj.search.assert_called_once()
    mock_module_obj.read.assert_called_once()
    mock_module_obj.button_immediate_install.assert_called_once_with([1])
    mock_module_obj.button_immediate_upgrade.assert_called_once_with([2, 3])


@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_install_only(mock_get_connection: MagicMock) -> None:
    """Tests the workflow when only installations are needed."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [1]
    mock_module_obj.read.return_value = [
        {"id": 1, "name": "module_to_install", "state": "uninstalled"}
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_installation(config="dummy.conf", modules=["module_to_install"])

    mock_module_obj.button_immediate_install.assert_called_once_with([1])
    mock_module_obj.button_immediate_upgrade.assert_not_called()


@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_upgrade_only(mock_get_connection: MagicMock) -> None:
    """Tests the workflow when only upgrades are needed."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [2, 3]
    mock_module_obj.read.return_value = [
        {"id": 2, "name": "module_to_upgrade", "state": "installed"},
        {"id": 3, "name": "another_to_upgrade", "state": "installed"},
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_installation(
        config="dummy.conf", modules=["module_to_upgrade", "another_to_upgrade"]
    )

    mock_module_obj.button_immediate_install.assert_not_called()
    mock_module_obj.button_immediate_upgrade.assert_called_once_with([2, 3])


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_api_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests error handling when the Odoo API call for installation fails."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [1]
    mock_module_obj.read.return_value = [
        {"id": 1, "name": "module_a", "state": "uninstalled"}
    ]
    mock_module_obj.button_immediate_install.side_effect = Exception("API Error")
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_installation(config="dummy.conf", modules=["module_a"])

    mock_log_error.assert_called_once()
    assert "An error occurred during module operation" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_upgrade_api_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests error handling when the Odoo API call for upgrading fails."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [1]
    mock_module_obj.read.return_value = [
        {"id": 1, "name": "module_a", "state": "installed"}
    ]
    mock_module_obj.button_immediate_upgrade.side_effect = Exception("API Error")
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_installation(config="dummy.conf", modules=["module_a"])

    mock_log_error.assert_called_once()
    assert "An error occurred during module operation" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.actions.module_manager.log.warning")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_not_found(
    mock_get_connection: MagicMock, mock_log_warning: MagicMock
) -> None:
    """Tests that a warning is logged if no modules are found to install."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = []
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_installation(config="dummy.conf", modules=["non_existent_module"])

    mock_log_warning.assert_called_once_with(
        "No matching modules found in the database."
    )


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_installation_connection_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests that an error is logged if the Odoo connection fails."""
    mock_get_connection.side_effect = Exception("Connection Refused")
    run_module_installation(config="bad.conf", modules=["any_module"])
    mock_log_error.assert_called_once()
    assert "Failed to connect to Odoo" in mock_log_error.call_args[0][0]


@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_uninstallation(mock_get_connection: MagicMock) -> None:
    """Tests that the uninstallation workflow correctly finds and uninstalls modules."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [10, 20]
    mock_module_obj.read.return_value = [
        {"id": 10, "name": "module_a", "state": "installed"},
        {"id": 20, "name": "module_b", "state": "installed"},
    ]
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_uninstallation(config="dummy.conf", modules=["module_a", "module_b"])

    mock_module_obj.search.assert_called_once()
    mock_module_obj.button_immediate_uninstall.assert_called_once_with([10, 20])


@patch("odoo_data_flow.lib.actions.module_manager.log.warning")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_uninstallation_not_found(
    mock_get_connection: MagicMock, mock_log_warning: MagicMock
) -> None:
    """Tests that a warning is logged if no installed modules are found to uninstall."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = []
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_uninstallation(config="dummy.conf", modules=["uninstalled_module"])

    mock_log_warning.assert_called_once_with(
        "No matching installed modules found to uninstall."
    )
    mock_module_obj.button_immediate_uninstall.assert_not_called()


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_uninstallation_api_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests error handling when the Odoo API call for uninstallation fails."""
    mock_module_obj = MagicMock()
    mock_module_obj.search.return_value = [1]
    mock_module_obj.read.return_value = [
        {"id": 1, "name": "module_a", "state": "installed"}
    ]
    mock_module_obj.button_immediate_uninstall.side_effect = Exception("API Error")
    mock_connection = MagicMock()
    mock_connection.get_model.return_value = mock_module_obj
    mock_get_connection.return_value = mock_connection

    run_module_uninstallation(config="dummy.conf", modules=["module_a"])

    mock_log_error.assert_called_once()
    assert (
        "An error occurred during module uninstallation"
        in mock_log_error.call_args[0][0]
    )


@patch("odoo_data_flow.lib.actions.module_manager.log.error")
@patch("odoo_data_flow.lib.actions.module_manager.conf_lib.get_connection_from_config")
def test_run_module_uninstallation_connection_error(
    mock_get_connection: MagicMock, mock_log_error: MagicMock
) -> None:
    """Tests error handling for connection failure during uninstallation."""
    mock_get_connection.side_effect = Exception("Connection Refused")
    run_module_uninstallation(config="bad.conf", modules=["any_module"])
    mock_log_error.assert_called_once()
    assert "Failed to connect to Odoo" in mock_log_error.call_args[0][0]
