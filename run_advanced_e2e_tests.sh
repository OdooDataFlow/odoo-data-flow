#!/bin/bash
set -e

ODOO_VERSION=$1
if [ -z "$ODOO_VERSION" ]; then
    echo "Usage: $0 <odoo_version>"
    exit 1
fi

COMPOSE_FILE="docker-compose.advanced.yml"

# Function to handle failures
handle_failure() {
    echo "--- An error occurred. Dumping container logs. ---"
    docker-compose -f $COMPOSE_FILE logs
    echo "--- Tearing down containers ---"
    docker-compose -f $COMPOSE_FILE down --volumes
    exit 1
}

# Trap errors
trap 'handle_failure' ERR


# Function to replace Odoo version in compose file
replace_odoo_version() {
    sed -i "s/image: odoo:.*/image: odoo:$ODOO_VERSION/g" $COMPOSE_FILE
}

# Clean up previous run
echo "--- Cleaning up previous run ---"
docker-compose -f $COMPOSE_FILE down --volumes || true
rm -f testdata/res_partner_advanced.csv
rm -f testdata/res_partner_category_advanced.csv
rm -rf .odf_cache
mkdir -p conf conf_target testdata

# Replace Odoo version in compose file
replace_odoo_version

# Start Odoo containers
echo "--- Starting containers for Odoo $ODOO_VERSION ---"
docker-compose -f $COMPOSE_FILE up -d --build

# Wait for databases to be ready
echo "--- Waiting for databases to be ready ---"
sleep 15 # Initial wait

echo "Waiting for Odoo Source to be ready..."
timeout 600 bash -c 'until curl -s http://localhost:8069/web/login > /dev/null; do echo -n "."; sleep 5; done'
echo "Odoo Source is ready!"

echo "Waiting for Odoo Target to be ready..."
timeout 600 bash -c 'until curl -s http://localhost:8070/web/login > /dev/null; do echo -n "."; sleep 5; done'
echo "Odoo Target is ready!"

# Create and initialize databases
echo "--- Creating and initializing databases ---"
docker-compose -f $COMPOSE_FILE exec -T odoo-source odoo -d odoo -i base --stop-after-init --db_host=db-source --db_user=odoo --db_password=odoo
docker-compose -f $COMPOSE_FILE exec -T odoo-target odoo -d odoo -i base --stop-after-init --db_host=db-target --db_user=odoo --db_password=odoo


# Install dependencies in containers
echo "--- Installing dependencies ---"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-source bash -c "apt-get update && apt-get install -y git && python3 -m pip install --upgrade pip setuptools && pip install /odoo-data-flow"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-target bash -c "apt-get update && apt-get install -y git && python3 -m pip install --upgrade pip setuptools && pip install /odoo-data-flow"

# Seed the source database
echo "--- Seeding source database ---"
docker-compose -f $COMPOSE_FILE exec -T odoo-source python3 /odoo-data-flow/tests/e2e/seed_advanced_database.py odoo

# Create connection configs
cat << EOF > conf/connection.conf
[Connection]
hostname = localhost
port = 8069
login = admin
password = admin
database = odoo
protocol = jsonrpc
EOF

cat << EOF > conf_target/connection.conf
[Connection]
hostname = localhost
port = 8070
login = admin
password = admin
database = odoo
protocol = jsonrpc
EOF

# Run the export for categories
echo "--- Exporting categories ---"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-source bash -c "chown -R odoo:odoo /odoo-data-flow && cd /odoo-data-flow && su odoo -c 'odoo-data-flow export --config conf/connection.conf --model res.partner.category --domain \"[('name', 'like', 'Test Category%')]\" --fields \"id,name\" --output testdata/res_partner_category_advanced.csv'"

# Run the export for partners
echo "--- Exporting partners ---"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-source bash -c "chown -R odoo:odoo /odoo-data-flow && cd /odoo-data-flow && su odoo -c 'odoo-data-flow export --config conf/connection.conf --model res.partner --domain \"[('name', 'like', 'Advanced Test Partner%')]\" --fields \"id,name,category_id/.id\" --output testdata/res_partner_advanced.csv'"

# Modify the partner export header for import
sed -i 's/category_id\/.id/category_id/g' testdata/res_partner_advanced.csv

# Run the import for categories into the target
echo "--- Importing categories into target ---"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-target bash -c "chown -R odoo:odoo /odoo-data-flow && cd /odoo-data-flow && su odoo -c 'odoo-data-flow import --config conf_target/connection.conf --file testdata/res_partner_category_advanced.csv'"

# Run the import for partners into the target
echo "--- Importing partners into target ---"
docker-compose -f $COMPOSE_FILE exec -T --user root odoo-target bash -c "chown -R odoo:odoo /odoo-data-flow && cd /odoo-data-flow && su odoo -c 'odoo-data-flow import --config conf_target/connection.conf --file testdata/res_partner_advanced.csv --strategy relational'"}

# Verify the data in the target database
echo "--- Verifying data in target database ---"
docker-compose -f $COMPOSE_FILE exec -T odoo-target python3 /odoo-data-flow/tests/e2e/verify_advanced_data.py odoo

# Tear down the containers
echo "--- Tearing down containers ---"
docker-compose -f $COMPOSE_FILE down --volumes

echo "--- Advanced e2e tests completed successfully! ---"