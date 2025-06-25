"""Config File Handler.

This module handles reading the connection configuration file and
establishing a connection to the Odoo server using odoo-client-lib.
"""

import configparser
from typing import Any

import odoolib

from ..logging_config import log


def get_connection_from_config(config_file: str) -> Any:
    """Get connection from config.

    Reads an Odoo connection configuration file and returns an
    initialized OdooClient object.

    Args:
        config_file (str): The path to the connection.conf file.

    Returns:
        Any: An initialized and connected Odoo client object,
             returned by odoolib.get_connection)
                    or raises an exception on failure.
    """
    config = configparser.ConfigParser()
    if not config.read(config_file):
        log.error(f"Configuration file not found or is empty: {config_file}")
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        conn_details = dict(config["Connection"])

        # Ensure port and uid are integers
        if "port" in conn_details:
            conn_details["port"] = int(conn_details["port"])
        if "uid" in conn_details:
            # The OdooClient expects the user ID as 'user_id'
            conn_details["user_id"] = int(conn_details.pop("uid"))

        log.info(f"Connecting to Odoo server at {conn_details.get('hostname')}...")

        # Use odoo-client-lib to establish the connection
        connection = odoolib.get_connection(**conn_details)

        log.info("Connection successful.")
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
