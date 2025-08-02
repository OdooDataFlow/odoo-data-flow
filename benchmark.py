import shutil
import subprocess
import sys
import time

# Path to a realistically large sample CSV file
SAMPLE_CSV = "/home/bosd/doodba/sps_12_18/data/res_partner_transformed.csv"
CONFIG_FILE = "/home/bosd/doodba/sps_12_18/conf/local_connection.conf"  # Make sure this path is correct
MODEL = "res.partner"

# --- Script Logic ---
# Find the 'odoo-data-flow' command in the current environment's PATH.
# This is a more robust way to find the executable.
command_path = shutil.which("odoo-data-flow")

start_time = time.time()

if not command_path:
    print("ERROR: Could not find the 'odoo-data-flow' command in your PATH.")
    print(
        "Please ensure your virtual environment is active and the package is installed."
    )
    sys.exit(1)

print(f"--- Running Benchmark using command: {command_path} ---")

# Use subprocess to call the command-line tool directly
# This ensures you're running the installed, potentially compiled version
subprocess.run(
    [
        command_path,
        "import",
        "--config",
        CONFIG_FILE,
        "--file",
        SAMPLE_CSV,
        "--model",
        MODEL,
        "--no-preflight-checks",  # Skip checks for a purer performance test
    ],
    check=True,
)

end_time = time.time()
duration = end_time - start_time

print("\n--- Benchmark Complete ---")
print(f"Total import time: {duration:.4f} seconds")
