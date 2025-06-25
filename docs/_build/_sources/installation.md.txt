# Installation with uv

This guide uses `uv`, a high-performance Python package installer and resolver, to set up your environment. It's a modern, fast alternative to `pip` and `venv`.

## 1. Install `uv` (if you haven't already)

First, ensure `uv` is installed on your system. If not, run the appropriate command for your operating system:

```bash
# macOS / Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (in PowerShell)
irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex
```

For other installation options, please refer to the [official `uv` documentation](https://astral.sh/uv#installation).

## 2. Prerequisites

- **Python 3.10 or newer:** `uv` will automatically find and use a compatible Python version on your system.
- **Access to an Odoo instance:** To import or export data, you will need the URL, database name, and login credentials for an Odoo instance.

## 3. The Connection Configuration File

Before you can use the tool, you must create a configuration file to store your Odoo connection details.

Create a folder named `conf/` in your project directory, and inside it, create a file named `connection.conf`.

**File: `conf/connection.conf`**

```ini
[Connection]
hostname = my-odoo-instance.odoo.com
database = my_odoo_db
login = admin
password = <your_odoo_password>
protocol = jsonrpcs
port = 443
uid = 2
```

### Configuration Keys Explained

| Key        | Description                                                                                                                               |
| :--------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `hostname` | The domain or IP address of your Odoo server.                                                                                             |
| `database` | The name of the Odoo database you want to connect to.                                                                                     |
| `login`    | The login username for the Odoo user that will perform the operations.                                                                    |
| `password` | The password for the specified Odoo user.                                                                                                 |
| `protocol` | The protocol to use for the connection. For Odoo.sh or a standard HTTPS setup, use `jsonrpcs`. For a local, non-SSL setup, use `jsonrpc`. |
| `port`     | The port for the connection. Standard ports are `443` for HTTPS (`jsonrpcs`) and `8069` for HTTP (`jsonrpc`).                             |
| `uid`      | The database ID of the Odoo user. `2` is often the default administrator user in a new database.                                          |

## 4. Standard Installation

1.  **Create and activate a virtual environment:**

    This command creates a standard virtual environment in a `.venv` folder.

    ```bash
    uv venv
    ```

    Next, activate the environment:

    ```bash
    # For Unix/macOS
    source .venv/bin/activate

    # For Windows
    .venv\Scripts\activate
    ```

    Your terminal prompt should now indicate that you are in the `.venv` environment.

2.  **Install `odoo-data-flow`:**

    With the environment active, use `uv` to install the package from PyPI.

    ```bash
    uv pip install odoo-data-flow
    ```

## 5. Installing for Development

If you want to contribute to the project or test the latest unreleased changes, you can install the library directly from the source code.

1.  **Clone the GitHub repository:**

    ```bash
    git clone [https://github.com/OdooDataFlow/odoo-data-flow.git](https://github.com/OdooDataFlow/odoo-data-flow.git)
    cd odoo-data-flow
    ```

2.  **Create and activate an environment:**

    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install in editable mode:**
    This command links the installed package to the source code in your directory. Any edits you make to the code will be immediately available.
    ```bash
    uv pip install -e .
    ```

You are now set up and ready to create your first data flow.
