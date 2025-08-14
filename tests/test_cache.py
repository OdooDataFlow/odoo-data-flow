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


@patch("configparser.ConfigParser")
def test_get_cache_dir_handles_exception(
    mock_config_parser: MagicMock, caplog: "MagicMock"
) -> None:
    """Verify that get_cache_dir handles exceptions gracefully."""
    mock_instance = mock_config_parser.return_value
    mock_instance.get.side_effect = Exception("Test exception")
    cache_dir = cache.get_cache_dir("dummy.conf")
    assert cache_dir is None
    assert "Could not create or access cache directory" in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=None)
def test_save_id_map_handles_no_cache_dir(
    mock_get_cache_dir: MagicMock, caplog: "MagicMock"
) -> None:
    """Verify save_id_map handles no cache directory."""
    cache.save_id_map("dummy.conf", "res.partner", {"a": 1})
    assert "Saved id_map for model" not in caplog.text


def test_save_id_map_handles_empty_id_map(tmp_path: Path, caplog: "MagicMock") -> None:
    """Verify save_id_map handles an empty id_map."""
    with patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=tmp_path):
        cache.save_id_map("dummy.conf", "res.partner", {})
        assert "Saved id_map for model" not in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir")
@patch("polars.DataFrame.write_parquet")
def test_save_id_map_handles_write_error(
    mock_write_parquet: MagicMock, mock_get_cache_dir: MagicMock, tmp_path: Path, caplog: "MagicMock"
) -> None:
    """Verify save_id_map handles write errors."""
    mock_get_cache_dir.return_value = tmp_path
    mock_write_parquet.side_effect = Exception("Write error")
    cache.save_id_map("dummy.conf", "res.partner", {"a": 1})
    assert "Failed to save id_map for model 'res.partner'" in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=None)
def test_load_id_map_handles_no_cache_dir(mock_get_cache_dir: MagicMock) -> None:
    """Verify load_id_map handles no cache directory."""
    result = cache.load_id_map("dummy.conf", "res.partner")
    assert result is None


@patch("odoo_data_flow.lib.cache.get_cache_dir")
@patch("polars.read_parquet")
def test_load_id_map_handles_read_error(
    mock_read_parquet: MagicMock, mock_get_cache_dir: MagicMock, tmp_path: Path, caplog: "MagicMock"
) -> None:
    """Verify load_id_map handles read errors."""
    mock_get_cache_dir.return_value = tmp_path
    (tmp_path / "res.partner.id_map.parquet").touch()
    mock_read_parquet.side_effect = Exception("Read error")
    result = cache.load_id_map("dummy.conf", "res.partner")
    assert result is None
    assert "Failed to load id_map for model 'res.partner'" in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir")
def test_save_and_load_fields_get_cache(
    mock_get_cache_dir: MagicMock, tmp_path: Path
) -> None:
    """Verify that a fields_get result can be saved and loaded."""
    # Arrange
    mock_get_cache_dir.return_value = tmp_path
    model = "res.users"
    fields_data = {
        "name": {"type": "char", "string": "Name"},
        "email": {"type": "char", "string": "Email"},
    }

    # Act
    cache.save_fields_get_cache("dummy.conf", model, fields_data)
    loaded_data = cache.load_fields_get_cache("dummy.conf", model)

    # Assert
    assert loaded_data == fields_data


def test_load_fields_get_cache_returns_none_if_not_found(tmp_path: Path) -> None:
    """Verify that loading a non-existent fields_get cache returns None."""
    with patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=tmp_path):
        loaded_data = cache.load_fields_get_cache("dummy.conf", "non.existent.model")
        assert loaded_data is None


@patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=None)
def test_save_fields_get_cache_handles_no_cache_dir(
    mock_get_cache_dir: MagicMock, caplog: "MagicMock"
) -> None:
    """Verify save_fields_get_cache handles no cache directory."""
    cache.save_fields_get_cache("dummy.conf", "res.partner", {"field": "data"})
    assert "Saved fields_get cache for model" not in caplog.text


def test_save_fields_get_cache_handles_empty_data(
    tmp_path: Path, caplog: "MagicMock"
) -> None:
    """Verify save_fields_get_cache handles empty data."""
    with patch("odoo_data_flow.lib.cache.get_cache_dir", return_value=tmp_path):
        cache.save_fields_get_cache("dummy.conf", "res.partner", {})
        assert "Saved fields_get cache for model" not in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir")
@patch("json.dump")
def test_save_fields_get_cache_handles_write_error(
    mock_json_dump: MagicMock, mock_get_cache_dir: MagicMock, tmp_path: Path, caplog: "MagicMock"
) -> None:
    """Verify save_fields_get_cache handles write errors."""
    mock_get_cache_dir.return_value = tmp_path
    mock_json_dump.side_effect = Exception("Write error")
    cache.save_fields_get_cache("dummy.conf", "res.partner", {"field": "data"})
    assert "Failed to save fields_get cache for model 'res.partner'" in caplog.text


@patch("odoo_data_flow.lib.cache.get_cache_dir")
@patch("json.load")
def test_load_fields_get_cache_handles_read_error(
    mock_json_load: MagicMock, mock_get_cache_dir: MagicMock, tmp_path: Path, caplog: "MagicMock"
) -> None:
    """Verify load_fields_get_cache handles read errors."""
    mock_get_cache_dir.return_value = tmp_path
    (tmp_path / "res.partner.fields.json").touch()
    mock_json_load.side_effect = Exception("Read error")
    result = cache.load_fields_get_cache("dummy.conf", "res.partner")
    assert result is None
    assert "Failed to load fields_get cache for model 'res.partner'" in caplog.text


def test_generate_session_id_is_consistent() -> None:
    """Verify that the session ID is consistent for the same inputs."""
    # Arrange
    model = "res.partner"
    domain = [("is_company", "=", True), ("customer", "=", True)]
    fields = ["name", "email", "phone"]

    # Act
    session_id1 = cache.generate_session_id(model, domain, fields)
    session_id2 = cache.generate_session_id(model, domain, fields)

    # Assert
    assert session_id1 == session_id2


def test_generate_session_id_is_sensitive_to_model() -> None:
    """Verify that the session ID changes with the model."""
    # Arrange
    domain = [("is_company", "=", True)]
    fields = ["name"]

    # Act
    session_id1 = cache.generate_session_id("res.partner", domain, fields)
    session_id2 = cache.generate_session_id("res.users", domain, fields)

    # Assert
    assert session_id1 != session_id2


def test_generate_session_id_is_sensitive_to_domain() -> None:
    """Verify that the session ID changes with the domain."""
    # Arrange
    model = "res.partner"
    fields = ["name"]

    # Act
    session_id1 = cache.generate_session_id(model, [("is_company", "=", True)], fields)
    session_id2 = cache.generate_session_id(model, [("is_company", "=", False)], fields)

    # Assert
    assert session_id1 != session_id2


def test_generate_session_id_is_sensitive_to_fields() -> None:
    """Verify that the session ID changes with the fields."""
    # Arrange
    model = "res.partner"
    domain = [("is_company", "=", True)]

    # Act
    session_id1 = cache.generate_session_id(model, domain, ["name", "email"])
    session_id2 = cache.generate_session_id(model, domain, ["name", "phone"])

    # Assert
    assert session_id1 != session_id2


def test_generate_session_id_is_order_agnostic() -> None:
    """Verify that the session ID is not sensitive to the order of items."""
    # Arrange
    model = "res.partner"
    domain1 = [("is_company", "=", True), ("customer", "=", True)]
    domain2 = [("customer", "=", True), ("is_company", "=", True)]
    fields1 = ["name", "email", "phone"]
    fields2 = ["phone", "name", "email"]

    # Act
    session_id1 = cache.generate_session_id(model, domain1, fields1)
    session_id2 = cache.generate_session_id(model, domain2, fields2)

    # Assert
    assert session_id1 == session_id2


def test_generate_session_id_handles_unsortable_domain() -> None:
    """Verify session ID generation with unsortable domain falls back."""
    # Arrange
    model = "res.partner"
    # Domain with mixed types that can't be sorted
    domain = [("name", "=", "test"), ("id", "in", [1, 2]), 1]
    fields = ["name"]

    # Act
    session_id = cache.generate_session_id(model, domain, fields)

    # Assert
    assert isinstance(session_id, str)
    assert len(session_id) == 16


def test_get_session_dir_creates_directory(tmp_path: Path) -> None:
    """Verify that a session directory is created correctly."""
    # Arrange
    session_id = "test_session_123"
    with patch.object(Path, "cwd", return_value=tmp_path):
        # Act
        session_dir = cache.get_session_dir(session_id)

        # Assert
        assert session_dir is not None
        assert session_dir.name == session_id
        assert session_dir.parent.name == "sessions"
        assert session_dir.exists()


@patch("pathlib.Path.mkdir")
def test_get_session_dir_handles_exception(
    mock_mkdir: MagicMock, caplog: "MagicMock"
) -> None:
    """Verify get_session_dir handles exceptions gracefully."""
    mock_mkdir.side_effect = Exception("Test exception")
    session_dir = cache.get_session_dir("test_session")
    assert session_dir is None
    assert "Could not create or access session directory" in caplog.text

