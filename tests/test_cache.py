"""Tests for the caching logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from polars.testing import assert_frame_equal

from odoo_data_flow.lib import cache


@patch("configparser.ConfigParser")
def test_get_cache_dir_creates_unique_directory(
    mock_config_parser: MagicMock, tmp_path: Path
) -> None:
    """Verify that a unique, hashed directory is created."""
    # Arrange
    mock_instance = mock_config_parser.return_value
    mock_instance.get.side_effect = ["localhost", 8069, "test_db"]
    expected_hash = "a1b2c3d4e5f6..."  # A known hash for the test data
    with patch("hashlib.sha256") as mock_sha256:
        mock_sha256.return_value.hexdigest.return_value = expected_hash
        with patch.object(Path, "cwd", return_value=tmp_path):
            # Act
            cache_dir = cache.get_cache_dir("dummy.conf")

            # Assert
            assert cache_dir is not None
            assert cache_dir.name == expected_hash
            assert cache_dir.exists()


@patch("odoo_data_flow.lib.cache.get_cache_dir")
def test_save_and_load_id_map(mock_get_cache_dir: "MagicMock", tmp_path: Path) -> None:
    """Verify that an id_map can be saved and loaded correctly."""
    # Arrange
    mock_get_cache_dir.return_value = tmp_path
    model = "res.partner"
    id_map = {"partner_a": 101, "partner_b": 102}

    # Act
    cache.save_id_map("dummy.conf", model, id_map)
    loaded_df = cache.load_id_map("dummy.conf", model)

    # Assert
    assert loaded_df is not None
    expected_df = pl.DataFrame(
        {"external_id": ["partner_a", "partner_b"], "db_id": [101, 102]}
    )
    assert_frame_equal(loaded_df, expected_df)


def test_load_id_map_returns_none_if_not_found(tmp_path: Path) -> None:
    """Verify that loading a non-existent map returns None."""
    with patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=tmp_path):
        loaded_df = cache.load_id_map("dummy.conf", "non.existent.model")
        assert loaded_df is None
