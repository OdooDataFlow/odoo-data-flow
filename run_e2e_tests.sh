#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Argument 1 is the Odoo version
ODOO_VERSION=$1

if [ -z "$ODOO_VERSION" ]; then
    echo "Usage: $0 <odoo_version>"
    exit 1
fi

echo "--- Starting e2e tests for Odoo version $ODOO_VERSION ---"

# Cleanup any previous runs
podman compose down -v || true

# Use the Odoo version to create a dynamic docker-compose.yml
cat << EOF > docker-compose.yml

services:
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=odoo
      - POSTGRES_USER=odoo
    ports:
      - "5432:5432"

  odoo:
    image: odoo:$ODOO_VERSION
    depends_on:
      - db
    ports:
      - "8069:8069"
    environment:
      - PGHOST=db
      - PGPORT=5432
      - PGUSER=odoo
      - PGPASSWORD=odoo

    volumes:
      - .:/odoo-data-flow

EOF

cat docker-compose.yml

echo "--- Starting containers... ---"
podman compose up -d

# 2. Wait for Odoo to be ready
echo "Waiting for Odoo to be ready..."
TIMEOUT=300 # 5 minutes timeout
START_TIME=$(date +%s)
until podman compose logs odoo | grep -q "HTTP service (werkzeug) running"; do
  CURRENT_TIME=$(date +%s)
  ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
  if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
    echo "Timeout: Odoo did not become ready within $TIMEOUT seconds."
    echo "Odoo container logs:"
    podman compose logs odoo
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo "Odoo is ready!"


echo "Checking Odoo accessibility from host..."
curl -v http://localhost:8069/web/login

# 3. Create the source and target databases
# Use odoo to create the databases. You'll need to run this within the odoo container.
docker-compose exec -T odoo odoo -d odoo_data_flow_source_db -i base  --without-demo=True --stop-after-init

# Wait for source database to be ready
echo "Waiting for source database to be ready..."
TIMEOUT=300 # 5 minutes timeout
START_TIME=$(date +%s)
until docker-compose exec -T odoo psql -h db -U odoo -d odoo_data_flow_source_db -c "SELECT 1" > /dev/null 2>&1; do
  CURRENT_TIME=$(date +%s)
  ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
  if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
    echo "Timeout: Source database did not become ready within $TIMEOUT seconds."
    exit 1
  fi
  echo -n "."
  sleep 2
done
echo "Source database is ready!"

docker-compose exec -T odoo odoo -d odoo_data_flow_target_db --without-demo=True -i base --stop-after-init

# Wait for target database to be ready
echo "Waiting for target database to be ready..."
TIMEOUT=300 # 5 minutes timeout
START_TIME=$(date +%s)
until docker-compose exec -T odoo psql -h db -U odoo -d odoo_data_flow_target_db -c "SELECT 1" > /dev/null 2>&1; do
  CURRENT_TIME=$(date +%s)
  ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
  if [ $ELAPSED_TIME -ge $TIMEOUT ]; then
    echo "Timeout: Target database did not become ready within $TIMEOUT seconds."
    exit 1
  fi
  echo -n "."
  sleep 2
done
echo "Target database is ready!"

# 4. Seed the source database
echo "--- Seeding the source database... ---"
docker-compose exec -T odoo python3 /odoo-data-flow/tests/e2e/seed_database.py odoo_data_flow_source_db

# Create connection.conf
mkdir -p conf
cat << EOF > conf/connection.conf
[Connection]
hostname = localhost
port = 8069
login = admin
password = admin
database = odoo_data_flow_source_db
uid = 2
protocol = jsonrpc
EOF

# Install odoo-data-flow
uv pip install -e .

# 5. Run odoo-data-flow export
# (Call your tool's CLI)
python3 -m odoo_data_flow export --config conf/connection.conf --output testdata/res_partner.csv --model res.partner --fields id,name,email

# Update connection.conf for import
cat << EOF > conf/connection.conf
[Connection]
hostname = localhost
port = 8069
login = admin
password = admin
database = odoo_data_flow_target_db
uid = 2
protocol = jsonrpc
EOF

# 6. Run odoo-data-flow import
python3 -m odoo_data_flow import --config conf/connection.conf --file testdata/res_partner.csv

# 7. Verify the data
echo "--- Verifying the data... ---"
docker-compose exec -T odoo python3 /odoo-data-flow/tests/e2e/verify_data.py odoo_data_flow_target_db

# Cleanup containers
podman compose down -v

echo "--- e2e tests completed for Odoo version $ODOO_VERSION ---"
