# Install act if you haven't already
# (e.g., brew install act, or other methods)

# If using Podman, set up the Docker compatibility layer
# Note: This is a common way, but consult Podman docs for the latest recommendations.
podman system service --time=0 tcp:0.0.0.0:2375 &
export DOCKER_HOST=tcp://localhost:2375

# Run the e2e tests locally for a specific Odoo version
# The `-j` flag specifies the job, and we can simulate the `workflow_dispatch` input
act workflow_dispatch -j e2e --input odoo_version=16.0

# To run all versions defined in the matrix:
act workflow_dispatch -j e2e --input odoo_version=all
