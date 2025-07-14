"""Test the converter.

This test script generates data for the image converter functions
to be used in the main test suite.
"""

import base64
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests  # type: ignore[import-untyped]

from odoo_data_flow.converter import run_path_to_image, run_url_to_image


def test_run_path_to_image(tmp_path: Path) -> None:
    """Tests the run_path_to_image function.

    This test verifies that:
    1. It correctly reads a source CSV.
    2. It finds local image files and converts them to base64.
    3. It writes the correct data to the output CSV.
    4. It handles cases where image files are not found.
    """
    # 1. Setup: Create source CSV and dummy image files
    source_dir = tmp_path
    image_dir = source_dir / "images"
    image_dir.mkdir()

    source_csv = source_dir / "source.csv"
    output_csv = source_dir / "output.csv"

    # Create a dummy image file
    image_file_path = image_dir / "test_image.png"
    image_content = b"fake-image-data"
    image_file_path.write_bytes(image_content)
    expected_base64 = base64.b64encode(image_content).decode("utf-8")

    source_header = ["id", "name", "image_path"]
    source_data = [
        ["1", "Product A", "images/test_image.png"],
        ["2", "Product B", "images/not_found.png"],  # This file does not exist
        ["3", "Product C", ""],  # Empty path
    ]

    with open(source_csv, "w", newline="", encoding="utf-8") as f:
        # Use semicolon as the delimiter to match the Processor's default
        writer = csv.writer(f, delimiter=";")
        writer.writerow(source_header)
        writer.writerows(source_data)

    # 2. Action: Run the converter function
    run_path_to_image(
        file=str(source_csv),
        fields="image_path",
        out=str(output_csv),
        path=str(source_dir),
        delimiter=";",
    )

    # 3. Assertions
    assert output_csv.exists()
    with open(output_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        result_data = list(reader)

    assert len(result_data) == 3
    assert result_data[0]["name"] == "Product A"
    assert result_data[0]["image_path"] == expected_base64

    assert result_data[1]["name"] == "Product B"
    assert result_data[1]["image_path"] == "", "Path for missing file should be empty"

    assert result_data[2]["name"] == "Product C"
    assert result_data[2]["image_path"] == "", "Empty path should result in empty"


# Patch the target where it is looked up: in the mapper module
@patch("odoo_data_flow.lib.mapper.requests.get")
def test_run_url_to_image(mock_requests_get: MagicMock, tmp_path: Path) -> None:
    """Tests the run_url_to_image function.

    This test verifies that:
    1. It correctly reads a source CSV.
    2. It "downloads" content from a URL and converts it to base64.
    3. It handles cases where a URL download fails.
    """
    # 1. Setup: Mock the requests library and create a source file
    source_csv = tmp_path / "source_urls.csv"
    output_csv = tmp_path / "output_urls.csv"

    # Configure the mock to simulate a successful and a failed request
    mock_response_success = MagicMock()
    mock_response_success.content = b"fake-url-image-data"
    mock_response_success.raise_for_status.return_value = None

    mock_response_fail = MagicMock()
    # Raise the correct exception type that the code expects to catch
    mock_response_fail.raise_for_status.side_effect = (
        requests.exceptions.RequestException("404 Not Found")
    )

    # The side_effect will return these values in order for each call to get()
    mock_requests_get.side_effect = [
        mock_response_success,
        mock_response_fail,
    ]
    expected_base64 = base64.b64encode(b"fake-url-image-data").decode("utf-8")

    source_header = ["id", "name", "image_url"]
    source_data = [
        ["10", "Product D", "http://example.com/image.png"],
        ["20", "Product E", "http://example.com/not_found.png"],
    ]

    with open(source_csv, "w", newline="", encoding="utf-8") as f:
        # Use semicolon as the delimiter
        writer = csv.writer(f, delimiter=";")
        writer.writerow(source_header)
        writer.writerows(source_data)

    # 2. Action: Run the converter function
    run_url_to_image(
        file=str(source_csv),
        fields="image_url",
        out=str(output_csv),
        delimiter=";",
    )

    # 3. Assertions
    assert output_csv.exists()
    with open(output_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        result_data = list(reader)

    assert len(result_data) == 2
    assert result_data[0]["name"] == "Product D"
    assert result_data[0]["image_url"] == expected_base64

    assert result_data[1]["name"] == "Product E"
    assert result_data[1]["image_url"] == "", "URL for failed download should be empty"
