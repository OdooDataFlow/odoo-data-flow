# Configuration Guide

This guide provides a detailed reference for the `connection.conf` file, which is essential for connecting the `odoo-data-flow` tool to your Odoo instance.

## The Connection File

All commands that need to communicate with an Odoo server (e.g., `import`, `export`, `migrate`) require connection details. These are stored in a standard INI-formatted configuration file.

By default, the tool looks for this file at `conf/connection.conf`, but you can specify a different path using the `--config` command-line option.

### File Format and Example

The configuration file must contain a `[Connection]` section with the necessary key-value pairs.


```{code-block} ini
:caption: conf/connection.conf
[Connection]
hostname = localhost
port = 8069
database = my_odoo_db
login = admin
password = my_admin_password
uid = 2
protocol = xmlrpc
```

### Configuration Keys

#### `hostname`
* **Required**: Yes
* **Description**: The IP address or domain name of your Odoo server.
* **Example**: `hostname = odoo.mycompany.com`

#### `port`
* **Required**: Yes
* **Description**: The port your Odoo server is running on. This is typically `8069` for standard Odoo instances.
* **Example**: `port = 8069`

#### `database`
* **Required**: Yes
* **Description**: The name of the Odoo database you want to connect to.
* **Example**: `database = my_production_db`

#### `login`
* **Required**: Yes
* **Description**: The username (login email) of the Odoo user that the tool will use to connect.
* **Example**: `login = admin`

#### `password`
* **Required**: Yes
* **Description**: The password for the specified Odoo user.
* **Example**: `password = my_secret_password`

#### `uid`
* **Required**: Yes
* **Description**: The database ID of the Odoo user identified by the `login` parameter. This is required for making RPC calls.
* **Well-known IDs**:
  * `1`: The default administrator user in Odoo versions prior to 12.0.
  * `2`: The default administrator user in Odoo versions 12.0 and newer.
* **Example**: `uid = 2`

#### `protocol`
* **Required**: No
* **Description**: The connection protocol to use for XML-RPC calls. `xmlrpc` uses HTTP, while `xmlrpcs` uses HTTPS for a secure connection. While modern Odoo uses JSON-RPC for its web interface, the external API for this type of integration typically uses XML-RPC.
* **Default**: `xmlrpc`
* **Example**: `protocol = xmlrpcs`

---


```{admonition} Tip
:class: note

On premise, it's advised to use a dedicated API user with the minimal access rights required for the models related to the import, rather than using the main administrator account.
```

### Real world Example

Below is a real world example of connection to a cloud hosted odoo instance on [opaas](https://www.opaas.cloud/).

```{code-block} ini
:caption: conf/connection.conf
[Connection]
hostname = test.yourinstance.opa.as
database = bvnem-test
login = admin
password = secret_password
protocol = jsonrpcs
port = 443
uid = 2
```
