"""Config File Handler.

This module handles reading the connection configuration file and
establishing a connection to the Odoo server using odoo-client-lib.
"""

import configparser
from typing import Any

import odoolib

from ..logging_config import log

_connection_cache: dict[str, Any] = {}


def get_connection_from_config(config_file: str) -> Any:
    """Get connection from config.

    Reads an Odoo connection configuration file and returns an
    initialized OdooClient object. It caches connections to reuse them.

    Args:
        config_file: The path to the connection.conf file.

    Returns:
        An initialized and connected Odoo client object,
        (returned by odoolib.get_connection)
        or raises an exception on failure.
    """
    if config_file in _connection_cache:
        log.debug(f"Reusing cached connection for {config_file}")
        return _connection_cache[config_file]

    config = configparser.ConfigParser()
    if not config.read(config_file):
        log.error(f"Configuration file not found or is empty: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        conn_details: dict[str, Any] = dict(config["Connection"])

        # Explicitly check for required keys before proceeding.
        # This loop is the crucial fix.
        required_keys = ["hostname", "database", "login", "password"]
        for key in required_keys:
            if key not in conn_details:
                raise KeyError(f"Required key '{key}' not found in config file.")

        # Ensure port and uid are integers if they exist
        if "port" in conn_details:
            conn_details["port"] = int(conn_details["port"])
        if "uid" in conn_details:
            # The OdooClient expects the user ID as 'user_id'
            conn_details["user_id"] = int(conn_details.pop("uid"))

        log.info(f"Connecting to Odoo server at {conn_details.get('hostname')}...")

        # Use odoo-client-lib to establish the connection
        connection = odoolib.get_connection(**conn_details)
        _connection_cache[config_file] = connection
        return connection

    except (KeyError, ValueError) as e:
        log.error(
            f"Configuration file '{config_file}' is missing a required key "
            f"or has a malformed value: {e}"
        )
        raise
    except Exception as e:
        log.error(f"An unexpected error occurred while connecting to Odoo: {e}")
        raise
