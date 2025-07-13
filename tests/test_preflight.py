"""Test the pre-flight checker functions."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from odoo_data_flow.lib import preflight


@pytest.fixture
def mock_polars_read_csv() -> Generator[MagicMock, None, None]:
    """Fixture to mock polars.read_csv."""
    with patch("odoo_data_flow.lib.preflight.pl.read_csv") as mock_read:
        yield mock_read


@pytest.fixture
def mock_conf_lib() -> Generator[MagicMock, None, None]:
    """Fixture to mock conf_lib.get_connection_from_config."""
    with patch(
        "odoo_data_flow.lib.preflight.conf_lib.get_connection_from_config"
    ) as mock_conn:
        yield mock_conn


@pytest.fixture
def mock_show_error_panel() -> Generator[MagicMock, None, None]:
    """Fixture to mock _show_error_panel."""
    with patch("odoo_data_flow.lib.preflight._show_error_panel") as mock_panel:
        yield mock_panel


class TestInternalHelpers:
    """Tests for internal helper functions in the preflight module."""

    @patch("odoo_data_flow.lib.preflight.log.error")
    def test_get_installed_languages_connection_fails(
        self, mock_log_error: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests that _get_installed_languages handles a connection error."""
        mock_conf_lib.side_effect = Exception("Connection Error")
        result = preflight._get_installed_languages("dummy.conf")
        assert result == set()
        mock_log_error.assert_called_once()
        assert "Could not fetch installed languages" in mock_log_error.call_args[0][0]

    def test_install_languages_success(self, mock_conf_lib: MagicMock) -> None:
        """Tests the success path of the _install_languages helper."""
        mock_lang_obj = MagicMock()
        mock_conf_lib.return_value.get_model.return_value = mock_lang_obj

        result = preflight._install_languages("dummy.conf", ["fr_FR", "de_DE"])
        assert result is True
        mock_lang_obj.load_lang.assert_called_once_with(["fr_FR", "de_DE"])

    @patch("odoo_data_flow.lib.preflight.log.error")
    def test_install_languages_failure(
        self, mock_log_error: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests that _install_languages handles an installation failure."""
        mock_lang_obj = MagicMock()
        mock_lang_obj.load_lang.side_effect = Exception("Install Failed")
        mock_conf_lib.return_value.get_model.return_value = mock_lang_obj

        result = preflight._install_languages("dummy.conf", ["fr_FR"])
        assert result is False
        mock_log_error.assert_called_once()
        assert "Failed to install languages" in mock_log_error.call_args[0][0]


class TestLanguageCheck:
    """Tests for the language_check pre-flight checker."""

    def test_language_check_skips_for_other_models(self) -> None:
        """Tests that the check is skipped for models other than partner/users."""
        result = preflight.language_check(
            model="product.product", filename="", config="", headless=False
        )
        assert result is True

    def test_language_check_skips_if_lang_column_missing(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests that the check is skipped if the 'lang' column is not present."""
        mock_polars_read_csv.return_value.get_column.side_effect = (
            pl.ColumnNotFoundError
        )
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is True

    def test_language_check_handles_file_read_error(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests that the check handles an error when reading the CSV."""
        mock_polars_read_csv.side_effect = Exception("Read Error")
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is True  # Should gracefully skip the check

    def test_language_check_no_required_languages(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests the case where the source file contains no languages."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = []  # noqa
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is True

    def test_all_languages_installed(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests the success case where all required languages are installed."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa
            "en_US",
            "fr_FR",
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"},
            {"code": "fr_FR"},
            {"code": "de_DE"},
        ]
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is True

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch("odoo_data_flow.lib.preflight._install_languages", return_value=True)
    def test_missing_languages_user_confirms_install_success(
        self,
        mock_install: MagicMock,
        mock_confirm: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Tests missing languages where user confirms and install succeeds."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa
            "en_US",
            "fr_FR",
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is True
        mock_confirm.assert_called_once()
        mock_install.assert_called_once_with("", ["fr_FR"])

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch("odoo_data_flow.lib.preflight._install_languages", return_value=False)
    def test_missing_languages_user_confirms_install_fails(
        self,
        mock_install: MagicMock,
        mock_confirm: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Tests missing languages where user confirms but install fails."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa
            "fr_FR"
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is False
        mock_confirm.assert_called_once()
        mock_install.assert_called_once_with("", ["fr_FR"])

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=False)
    def test_missing_languages_user_cancels(
        self,
        mock_confirm: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Tests missing languages where user cancels the installation."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa
            "fr_FR"
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=False
        )
        assert result is False
        mock_confirm.assert_called_once()

    @patch("odoo_data_flow.lib.preflight._install_languages", return_value=True)
    def test_missing_languages_headless_mode(
        self,
        mock_install: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Tests that languages are auto-installed in headless mode."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa
            "fr_FR"
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]
        result = preflight.language_check(
            model="res.partner", filename="file.csv", config="", headless=True
        )
        assert result is True
        mock_install.assert_called_once_with("", ["fr_FR"])


class TestFieldExistenceCheck:
    """Tests for the field_existence_check pre-flight checker."""

    def test_field_check_success(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests the success case where all CSV columns exist on the model."""
        mock_polars_read_csv.return_value.columns = ["id", "name", "email"]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"name": "id"},
            {"name": "name"},
            {"name": "email"},
            {"name": "phone"},
        ]
        result = preflight.field_existence_check(
            model="res.partner", filename="file.csv", config=""
        )
        assert result is True

    def test_field_check_failure(
        self,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Tests the failure case where a CSV column is missing from the model."""
        mock_polars_read_csv.return_value.columns = [
            "id",
            "name",
            "x_legacy_field",
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"name": "id"},
            {"name": "name"},
        ]
        result = preflight.field_existence_check(
            model="res.partner", filename="file.csv", config=""
        )
        assert result is False
        mock_show_error_panel.assert_called_once()
        assert "Invalid Fields Found" in mock_show_error_panel.call_args[0][0]
        assert "x_legacy_field" in mock_show_error_panel.call_args[0][1]

    def test_field_check_read_csv_fails(
        self, mock_polars_read_csv: MagicMock, mock_show_error_panel: MagicMock
    ) -> None:
        """Tests that the check handles an error when reading the CSV header."""
        mock_polars_read_csv.side_effect = Exception("Cannot read file")
        result = preflight.field_existence_check(
            model="res.partner", filename="file.csv", config=""
        )
        assert result is False
        mock_show_error_panel.assert_called_once()
        assert "File Read Error" in mock_show_error_panel.call_args[0][0]

    def test_field_check_odoo_connection_fails(
        self,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Tests that the check handles an Odoo connection failure."""
        mock_polars_read_csv.return_value.columns = ["id", "name"]
        mock_conf_lib.side_effect = Exception("Connection Refused")
        result = preflight.field_existence_check(
            model="res.partner", filename="file.csv", config=""
        )
        assert result is False
        mock_show_error_panel.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error_panel.call_args[0][0]
