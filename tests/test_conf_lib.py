"""Test the converter.

This test script generates data for the image converter functions
to be used in the main test suite.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from odoo_data_flow.lib.conf_lib import get_connection_from_config


@patch("odoo_data_flow.lib.conf_lib.odoolib.get_connection")
def test_get_connection_from_config_success(
    mock_get_connection: MagicMock, tmp_path: Path
) -> None:
    """Tests successful connection configuration parsing.

    Verifies that it reads a valid config file and calls the underlying
    connection library with correctly parsed and typed parameters.
    """
    # 1. Setup: Create a valid temporary config file
    config_file = tmp_path / "connection.conf"
    config_content = """
[Connection]
hostname = test-server
port = 8070
database = test-db
login = test-user
password = test-pass
uid = 2
"""
    config_file.write_text(config_content)

    # 2. Action: Call the function we are testing
    get_connection_from_config(str(config_file))

    # 3. Assertions: Check that the connection function was called correctly
    mock_get_connection.assert_called_once()
    call_kwargs = mock_get_connection.call_args.kwargs

    assert call_kwargs.get("hostname") == "test-server"
    assert call_kwargs.get("port") == 8070  # Should be converted to int
    assert call_kwargs.get("database") == "test-db"
    assert call_kwargs.get("login") == "test-user"
    assert call_kwargs.get("password") == "test-pass"
    # 'uid' should be popped and renamed to 'user_id'
    assert "uid" not in call_kwargs
    assert call_kwargs.get("user_id") == 2  # Should be converted to int


def test_get_connection_file_not_found() -> None:
    """Tests that a FileNotFoundError is raised if the config file does not exist."""
    with pytest.raises(FileNotFoundError):
        get_connection_from_config("non_existent_file.conf")


def test_get_connection_missing_key(tmp_path: Path) -> None:
    """Tests that a KeyError is raised if a required key is missing."""
    config_file = tmp_path / "missing_key.conf"
    # This config is missing the 'database' key
    config_content = """
[Connection]
hostname = test-server
port = 8069
login = admin
password = admin
"""
    config_file.write_text(config_content)

    with pytest.raises(KeyError):
        get_connection_from_config(str(config_file))


def test_get_connection_malformed_value(tmp_path: Path) -> None:
    """Tests that a ValueError is raised if a value cannot be converted to int."""
    config_file = tmp_path / "malformed.conf"
    # 'port' is not a valid integer
    config_content = """
[Connection]
hostname = test-server
port = not-a-number
database = test-db
login = admin
password = admin
uid = 2
"""
    config_file.write_text(config_content)

    with pytest.raises(ValueError):
        get_connection_from_config(str(config_file))
