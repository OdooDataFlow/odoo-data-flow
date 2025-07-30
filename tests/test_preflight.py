"""Test the pre-flight checker functions."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from polars.exceptions import ColumnNotFoundError

from odoo_data_flow.enums import PreflightMode
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


class TestLanguageCheck:
    """Tests for the language_check pre-flight checker."""

    def test_language_check_skips_for_other_models(self) -> None:
        """Tests that the check is skipped for models other than partner/users."""
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="product.product",
            filename="",
            config="",
            headless=False,
        )
        assert result is True

    def test_language_check_skips_if_lang_column_missing(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests that the check is skipped if the 'lang' column is not present."""
        mock_polars_read_csv.return_value.get_column.side_effect = ColumnNotFoundError
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is True

    def test_language_check_handles_file_read_error(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests that the check handles an error when reading the CSV."""
        mock_polars_read_csv.side_effect = Exception("Read Error")
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is True

    def test_language_check_no_required_languages(
        self, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests the case where the source file contains no languages."""
        mock_df = MagicMock()
        mock_df.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = []  # noqa: E501
        mock_polars_read_csv.return_value = mock_df
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is True

    def test_all_languages_installed(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests the success case where all required languages are installed."""
        mock_df = MagicMock()
        mock_df.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa: E501
            "en_US",
            "fr_FR",
        ]
        mock_polars_read_csv.return_value = mock_df

        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"},
            {"code": "fr_FR"},
            {"code": "de_DE"},
        ]
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is True

    @patch("odoo_data_flow.lib.preflight.language_installer.run_language_installation")
    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch(
        "odoo_data_flow.lib.preflight._get_installed_languages",
        return_value={"en_US"},
    )
    def test_missing_languages_user_confirms_install_success(
        self,
        mock_get_langs: MagicMock,
        mock_confirm: MagicMock,
        mock_installer: MagicMock,
        mock_polars_read_csv: MagicMock,
    ) -> None:
        """Tests missing languages where user confirms and install succeeds."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa: E501
            "fr_FR"
        ]
        mock_installer.return_value = True

        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is True
        mock_confirm.assert_called_once()
        mock_installer.assert_called_once_with("", ["fr_FR"])

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch(
        "odoo_data_flow.lib.actions.language_installer.run_language_installation",
        return_value=False,
    )
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
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is False
        mock_confirm.assert_called_once()
        mock_install.assert_called_once_with("", ["fr_FR"])

    @patch("odoo_data_flow.lib.preflight.language_installer.run_language_installation")
    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=False)
    @patch(
        "odoo_data_flow.lib.preflight._get_installed_languages",
        return_value={"en_US"},
    )
    def test_missing_languages_user_cancels(
        self,
        mock_get_langs: MagicMock,
        mock_confirm: MagicMock,
        mock_installer: MagicMock,
        mock_polars_read_csv: MagicMock,
    ) -> None:
        """Tests that the check fails if the user cancels the installation."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa: E501
            "fr_FR"
        ]

        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )
        assert result is False
        mock_confirm.assert_called_once()
        mock_installer.assert_not_called()

    @patch("odoo_data_flow.lib.preflight.language_installer.run_language_installation")
    @patch("odoo_data_flow.lib.preflight.Confirm.ask")
    @patch(
        "odoo_data_flow.lib.preflight._get_installed_languages",
        return_value={"en_US"},
    )
    def test_missing_languages_headless_mode(
        self,
        mock_get_langs: MagicMock,
        mock_confirm: MagicMock,
        mock_installer: MagicMock,
        mock_polars_read_csv: MagicMock,
    ) -> None:
        """Tests that languages are auto-installed in headless mode."""
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa: E501
            "fr_FR"
        ]
        mock_installer.return_value = True

        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="dummy.conf",
            headless=True,
        )
        assert result is True
        mock_confirm.assert_not_called()
        mock_installer.assert_called_once_with("dummy.conf", ["fr_FR"])

    @patch("odoo_data_flow.lib.preflight.log.warning")
    @patch("odoo_data_flow.lib.preflight.Confirm.ask")  # Ensure this is not called
    @patch(
        "odoo_data_flow.lib.actions.language_installer.run_language_installation"
    )  # Ensure this is not called
    def test_language_check_fail_mode_skips_install(
        self,
        mock_install: MagicMock,
        mock_confirm: MagicMock,
        mock_log_warning: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Test no language check in fail mode.

        Tests that in FAIL_MODE, language installation is skipped
        and the import is allowed to continue.
        """
        mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value = [  # noqa E501
            "fr_FR"
        ]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]

        result = preflight.language_check(
            preflight_mode=PreflightMode.FAIL_MODE,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )

        assert result is True  # Should allow import to continue
        mock_log_warning.assert_called_once()
        assert (
            "Fail mode: Missing languages detected" in mock_log_warning.call_args[0][0]
        )
        mock_confirm.assert_not_called()
        mock_install.assert_not_called()


class TestFieldExistenceCheck:
    """Tests for the field_existence_check pre-flight checker."""

    def test_field_check_success(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests the success case where all CSV columns exist on the model."""
        mock_polars_read_csv.return_value.columns = ["id", "name", "email"]

        # Mock model.fields_get() which is now used instead of search_read()
        # It returns a dictionary where keys are the field names.
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "email": {"type": "char"},
            "phone": {"type": "char"},
        }

        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
        )
        assert result is True

    def test_field_check_success_with_relational_id(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Test Relational Id's passes tests.

        Tests that the check PASSES when using Odoo's standard relational
        field syntax (field/id).
        """
        # Arrange
        mock_polars_read_csv.return_value.columns = ["name", "parent_id/id"]
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "name": {"type": "char"},
            "parent_id": {"type": "many2one"},
        }

        # Act
        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner.category",
            filename="file.csv",
            config="",
        )

        # Assert
        assert result is True

    def test_field_check_failure_with_invalid_base_field(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Test invalid base field.

        Tests that the check FAILS if the base field name is invalid,
        even when using relational syntax.
        """
        # Arrange
        mock_polars_read_csv.return_value.columns = [
            "name",
            "invalid_parent/id",
        ]
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "name": {"type": "char"},
            "parent_id": {"type": "many2one"},
        }

        # Act
        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner.category",
            filename="file.csv",
            config="",
        )

        # Assert
        assert result is False

    def test_field_check_rejects_export_only_syntax(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Test Reject export only syntax.

        Tests that the check FAILS when using the export-only '/.id' syntax,
        as this is not valid for importing.
        """
        # Arrange
        mock_polars_read_csv.return_value.columns = ["name", "parent_id/.id"]
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "name": {"type": "char"},
            "parent_id": {"type": "many2one"},
        }

        # Act
        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner.category",
            filename="file.csv",
            config="",
        )

        # Assert
        assert result is False

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
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {"id": {}, "name": {}}

        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
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
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
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
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
        )
        assert result is False
        mock_show_error_panel.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error_panel.call_args[0][0]

    def test_field_check_success_with_slash_notation_for_display_name(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Test the slash notation.

        Tests that the pre-flight check PASSES when using Odoo's standard
        slash notation to get a display name from a related field
        (e.g., 'field/name').
        """
        # --- Arrange ---
        # Simulate a CSV header requesting a related field's name
        mock_polars_read_csv.return_value.columns = ["name", "parent_id/name"]

        # Mock the Odoo model to confirm that the base field 'parent_id' exists
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "name": {"type": "char"},
            "parent_id": {
                "type": "many2one",
                "relation": "res.partner.category",
            },
        }

        # --- Act ---
        result = preflight.field_existence_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner.category",
            filename="file.csv",
            config="",
        )

        # --- Assert ---
        # The check should pass because 'parent_id' is a valid field
        assert result is True
