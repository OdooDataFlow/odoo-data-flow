"""Test the pre-flight checker functions."""

import csv
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
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


@pytest.fixture
def mock_cache() -> Generator[MagicMock, None, None]:
    """Fixture to mock the cache module."""
    with patch("odoo_data_flow.lib.preflight.cache") as mock_cache_module:
        yield mock_cache_module


@pytest.fixture
def mock_show_warning_panel() -> Generator[MagicMock, None, None]:
    """Fixture to mock _show_warning_panel."""
    with patch("odoo_data_flow.lib.preflight._show_warning_panel") as mock_panel:
        yield mock_panel


class TestConnectionCheck:
    """Tests for the connection_check pre-flight checker."""

    @patch("odoo_data_flow.lib.preflight.conf_lib.get_connection_from_dict")
    def test_connection_check_success_with_dict_config(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Verify connection check succeeds with dict configuration."""
        config = {"hostname": "localhost", "database": "test_db"}
        result = preflight.connection_check(
            preflight_mode=PreflightMode.NORMAL,
            config=config,
            model="res.partner",
            filename="test.csv",
        )

        assert result is True
        mock_get_connection.assert_called_once_with(config)

    @patch("odoo_data_flow.lib.preflight.conf_lib.get_connection_from_config")
    def test_connection_check_success_with_file_config(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Verify connection check succeeds with file configuration."""
        config_file = "/path/to/config.conf"
        result = preflight.connection_check(
            preflight_mode=PreflightMode.NORMAL,
            config=config_file,
            model="res.partner",
            filename="test.csv",
        )

        assert result is True
        mock_get_connection.assert_called_once_with(config_file=config_file)

    @patch("odoo_data_flow.lib.preflight.conf_lib.get_connection_from_dict")
    @patch("odoo_data_flow.lib.preflight._show_error_panel")
    def test_connection_check_failure_with_dict_config(
        self, mock_show_error: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """Verify connection check fails gracefully with dict configuration error."""
        mock_get_connection.side_effect = Exception("Connection failed")
        config = {"hostname": "invalid.host", "database": "test_db"}
        result = preflight.connection_check(
            preflight_mode=PreflightMode.NORMAL,
            config=config,
            model="res.partner",
            filename="test.csv",
        )

        assert result is False
        mock_get_connection.assert_called_once_with(config)
        mock_show_error.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error.call_args[0][0]

    @patch("odoo_data_flow.lib.preflight.conf_lib.get_connection_from_config")
    @patch("odoo_data_flow.lib.preflight._show_error_panel")
    def test_connection_check_failure_with_file_config(
        self, mock_show_error: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """Verify connection check fails gracefully with file configuration error."""
        mock_get_connection.side_effect = Exception("Config file not found")
        config_file = "/invalid/path/to/config.conf"
        result = preflight.connection_check(
            preflight_mode=PreflightMode.NORMAL,
            config=config_file,
            model="res.partner",
            filename="test.csv",
        )

        assert result is False
        mock_get_connection.assert_called_once_with(config_file=config_file)
        mock_show_error.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error.call_args[0][0]


class TestSelfReferencingCheck:
    """Tests for the self_referencing_check."""

    @patch("odoo_data_flow.lib.preflight.sort.sort_for_self_referencing")
    def test_check_plans_strategy_when_hierarchy_detected(
        self, mock_sort: MagicMock, tmp_path: "Path"
    ) -> None:
        """Verify the import plan is updated when a hierarchy is found."""
        sorted_file = tmp_path / "sorted.csv"
        mock_sort.return_value = str(sorted_file)
        import_plan: dict[str, Any] = {}
        result = preflight.self_referencing_check(
            preflight_mode=PreflightMode.NORMAL,
            filename="file.csv",
            import_plan=import_plan,
        )
        assert result is True
        assert import_plan["strategy"] == "sort_and_one_pass_load"
        assert import_plan["id_column"] == "id"
        assert import_plan["parent_column"] == "parent_id"
        mock_sort.assert_called_once_with(
            "file.csv", id_column="id", parent_column="parent_id", separator=";"
        )

    @patch("odoo_data_flow.lib.preflight.sort.sort_for_self_referencing")
    def test_check_does_nothing_when_no_hierarchy(self, mock_sort: MagicMock) -> None:
        """Verify the import plan is unchanged when no hierarchy is found."""
        mock_sort.return_value = None
        import_plan: dict[str, Any] = {}
        result = preflight.self_referencing_check(
            preflight_mode=PreflightMode.NORMAL,
            filename="file.csv",
            import_plan=import_plan,
        )
        assert result is True
        assert "strategy" not in import_plan

    @patch("odoo_data_flow.lib.preflight.sort.sort_for_self_referencing")
    def test_check_is_skipped_for_o2m(self, mock_sort: MagicMock) -> None:
        """Verify the check is skipped when o2m flag is True."""
        import_plan: dict[str, Any] = {}
        result = preflight.self_referencing_check(
            preflight_mode=PreflightMode.NORMAL,
            filename="file.csv",
            import_plan=import_plan,
            o2m=True,
        )
        assert result is True
        assert "strategy" not in import_plan
        mock_sort.assert_not_called()

    @patch("odoo_data_flow.lib.preflight.sort.sort_for_self_referencing")
    def test_check_handles_sort_function_error(self, mock_sort: MagicMock) -> None:
        """Verify the check handles errors from sort_for_self_referencing gracefully."""
        # Mock sort function to return False indicating an error
        mock_sort.return_value = False
        import_plan: dict[str, Any] = {}

        result = preflight.self_referencing_check(
            preflight_mode=PreflightMode.NORMAL,
            filename="file.csv",
            import_plan=import_plan,
        )

        # Should return False when sort function encounters an error
        assert result is False
        # Should not modify import plan when there's an error
        assert "strategy" not in import_plan
        mock_sort.assert_called_once()


class TestInternalHelpers:
    """Tests for internal helper functions in the preflight module."""

    @patch("odoo_data_flow.lib.preflight._show_error_panel")
    def test_get_installed_languages_connection_fails(
        self, mock_show_error_panel: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Tests that _get_installed_languages handles a connection error."""
        mock_conf_lib.side_effect = Exception("Connection Error")
        result = preflight._get_installed_languages("dummy.conf")
        assert result is None
        mock_show_error_panel.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error_panel.call_args[0][0]


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
        (
            mock_df.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = []
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
        (
            mock_df.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = [
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

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch(
        "odoo_data_flow.lib.preflight._get_installed_languages",
        return_value={"en_US"},
    )
    def test_language_check_dict_config_installation_not_supported(
        self,
        mock_get_langs: MagicMock,
        mock_confirm: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Tests that language installation fails gracefully with dict config."""
        # Setup data with missing languages
        mock_df = MagicMock()
        (
            mock_df.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = [
            "fr_FR",
        ]
        mock_polars_read_csv.return_value = mock_df

        # Use dict config (not supported for installation)
        config = {"hostname": "localhost", "database": "test_db"}
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config=config,
            headless=False,
        )

        # Should fail when installation is attempted with dict config
        assert result is False
        mock_confirm.assert_called_once()
        mock_show_error_panel.assert_called_once()
        assert (
            "Language installation from a dict config is not supported"
            in mock_show_error_panel.call_args[0][0]
        )

    @patch("odoo_data_flow.lib.preflight._get_installed_languages", return_value=None)
    def test_language_check_handles_get_installed_languages_failure(
        self, mock_get_langs: MagicMock, mock_polars_read_csv: MagicMock
    ) -> None:
        """Tests that language_check handles when _get_installed_languages fails."""
        # Setup CSV data with languages that would require checking
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]

        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )

        # Should return False when _get_installed_languages fails
        assert result is False
        mock_get_langs.assert_called_once_with("")

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
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]
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
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]

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
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]
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
        # In tests/test_preflight.py

    # Replace the old test_language_check_fail_mode_skips_install with this one.
    @patch("odoo_data_flow.lib.preflight.log.debug")  # Note: patching log.debug now
    @patch("odoo_data_flow.lib.preflight.Confirm.ask")
    @patch("odoo_data_flow.lib.actions.language_installer.run_language_installation")
    def test_language_check_fail_mode_skips_entire_check(
        self,
        mock_install: MagicMock,
        mock_confirm: MagicMock,
        mock_log_debug: MagicMock,  # Renamed from mock_log_warning
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
    ) -> None:
        """Test the skipped language check in fail mode.

        Tests that in FAIL_MODE, the language check is skipped entirely,
        preventing file reads or Odoo calls.
        """
        # ACT: Run the check in fail mode.
        result = preflight.language_check(
            preflight_mode=PreflightMode.FAIL_MODE,
            model="res.partner",
            filename="file.csv",
            config="",
            headless=False,
        )

        # ASSERT: Check for the new, correct behavior.
        assert result is True, "The check should return True in fail mode"

        # 1. Assert that the correct debug message was logged.
        mock_log_debug.assert_called_once_with("Skipping language pre-flight check.")

        # 2. Assert that the function exited before doing any real work.
        mock_polars_read_csv.assert_not_called()
        mock_conf_lib.assert_not_called()
        mock_install.assert_not_called()
        mock_confirm.assert_not_called()

    @patch("odoo_data_flow.lib.preflight.Confirm.ask", return_value=True)
    @patch(
        "odoo_data_flow.lib.preflight._get_installed_languages",
        return_value={"en_US"},
    )
    def test_language_check_dict_config_installation_not_supported_v2(
        self,
        mock_get_langs: MagicMock,
        mock_confirm: MagicMock,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Tests that language installation fails gracefully with dict config."""
        # Setup data with missing languages
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]
        mock_conf_lib.return_value.get_model.return_value.search_read.return_value = [
            {"code": "en_US"}
        ]

        # Use dict config (not supported for installation)
        config = {"hostname": "localhost", "database": "test_db"}
        result = preflight.language_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config=config,
            headless=False,
        )

        # Should fail when installation is attempted with dict config
        assert result is False
        mock_confirm.assert_called_once()
        mock_show_error_panel.assert_called_once()
        assert (
            "Language installation from a dict config is not supported"
            in mock_show_error_panel.call_args[0][0]
        )


class TestDeferralAndStrategyCheck:
    """Tests for the deferral_and_strategy_check pre-flight checker."""

    def test_direct_relational_import_strategy_for_large_volumes(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify 'direct_relational_import' is chosen for many m2m links."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "category_id"]

        # Setup a more robust mock for the chained Polars calls
        mock_df_data = MagicMock()
        (
            mock_df_data.lazy.return_value.select.return_value.select.return_value.sum.return_value.collect.return_value.item.return_value
        ) = 500
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "category_id": {
                "type": "many2many",
                "relation": "res.partner.category",
                "relation_table": "res_partner_res_partner_category_rel",
                "relation_field": "partner_id",
            },
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        assert "category_id" in import_plan["deferred_fields"]
        assert (
            import_plan["strategies"]["category_id"]["strategy"]
            == "direct_relational_import"
        )

    def test_write_tuple_strategy_when_missing_relation_info(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify 'write_tuple' is chosen when relation info is missing."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "category_id"]

        # Setup a more robust mock for the chained Polars calls
        mock_df_data = MagicMock()
        (
            mock_df_data.lazy.return_value.select.return_value.select.return_value.sum.return_value.collect.return_value.item.return_value
        ) = 100
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "category_id": {
                "type": "many2many",
                "relation": "res.partner.category",
                # Missing relation_table and relation_field
            },
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        assert "category_id" in import_plan["deferred_fields"]
        assert import_plan["strategies"]["category_id"]["strategy"] == "write_tuple"
        # Should not have relation_table or relation_field in strategy
        assert "relation" in import_plan["strategies"]["category_id"]

    def test_write_tuple_strategy_for_small_volumes(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify 'write_tuple' is chosen for fewer m2m links."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "category_id"]

        # Setup a more robust mock for the chained Polars calls
        mock_df_data = MagicMock()
        (
            mock_df_data.lazy.return_value.select.return_value.select.return_value.sum.return_value.collect.return_value.item.return_value
        ) = 499
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "category_id": {
                "type": "many2many",
                "relation": "res.partner.category",
                "relation_table": "res_partner_res_partner_category_rel",
                "relation_field": "partner_id",
            },
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        assert "category_id" in import_plan["deferred_fields"]
        assert import_plan["strategies"]["category_id"]["strategy"] == "write_tuple"

    def test_self_referencing_m2o_is_deferred(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify self-referencing many2one fields are deferred."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "parent_id"]
        mock_df_data = MagicMock()
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "parent_id": {"type": "many2one", "relation": "res.partner"},
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        assert "parent_id" in import_plan["deferred_fields"]

    def test_auto_detects_unique_id_field(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify 'id' is automatically chosen as the unique id field."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "parent_id"]
        mock_df_data = MagicMock()
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "parent_id": {"type": "many2one", "relation": "res.partner"},
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        assert import_plan["unique_id_field"] == "id"

    def test_error_if_no_unique_id_field_for_deferrals(
        self,
        mock_polars_read_csv: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Verify an error is shown if deferrals exist but no 'id' column."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["name", "parent_id"]
        mock_df_data = MagicMock()
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "name": {"type": "char"},
            "parent_id": {"type": "many2one", "relation": "res.partner"},
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is False
        mock_show_error_panel.assert_called_once()
        assert "Action Required" in mock_show_error_panel.call_args[0][0]

    def test_product_template_attribute_value_ids_not_deferred_in_product_product_model(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify product_template_attribute_value_ids is not deferred."""
        mock_df_header = MagicMock()
        mock_df_header.columns = [
            "id",
            "name",
            "categ_id",
            "product_template_attribute_value_ids",
        ]
        mock_df_data = MagicMock()
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "categ_id": {"type": "many2one", "relation": "product.category"},
            "product_template_attribute_value_ids": {
                "type": "many2many",
                "relation": "product.template.attribute.value",
            },
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="product.product",
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        # product_template_attribute_value_ids should NOT be in
        # deferred_fields for product.product model
        # But other relational fields like categ_id should still be deferred
        if "deferred_fields" in import_plan:
            assert (
                "product_template_attribute_value_ids"
                not in import_plan["deferred_fields"]
            )
            # categ_id should still be deferred as it's not the special case
            assert "categ_id" in import_plan["deferred_fields"]
        else:
            # If no fields are deferred, it means only the
            # product_template_attribute_value_ids was in the list
            # but since it's skipped, there are no deferred fields at all
            assert "product_template_attribute_value_ids" not in import_plan

    def test_product_template_attribute_value_ids_deferred_in_other_models(
        self, mock_polars_read_csv: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify product_template_attribute_value_ids is deferred."""
        mock_df_header = MagicMock()
        mock_df_header.columns = ["id", "name", "product_template_attribute_value_ids"]
        mock_df_data = MagicMock()
        mock_polars_read_csv.side_effect = [mock_df_header, mock_df_data]

        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "product_template_attribute_value_ids": {
                "type": "many2many",
                "relation": "product.template.attribute.value",
            },
        }
        import_plan: dict[str, Any] = {}
        result = preflight.deferral_and_strategy_check(
            preflight_mode=PreflightMode.NORMAL,
            model="res.partner",  # Different model
            filename="file.csv",
            config="",
            import_plan=import_plan,
        )
        assert result is True
        # product_template_attribute_value_ids SHOULD be in
        # deferred_fields for other models
        assert "product_template_attribute_value_ids" in import_plan["deferred_fields"]


class TestGetOdooFields:
    """Tests for the _get_odoo_fields helper function."""

    def test_get_odoo_fields_cache_hit(
        self, mock_cache: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify fields are returned from cache and Odoo is not called."""
        mock_cache.load_fields_get_cache.return_value = {"name": {"type": "char"}}
        result = preflight._get_odoo_fields("dummy.conf", "res.partner")

        assert result == {"name": {"type": "char"}}
        mock_cache.load_fields_get_cache.assert_called_once_with(
            "dummy.conf", "res.partner"
        )
        mock_conf_lib.assert_not_called()

    def test_get_odoo_fields_cache_miss(
        self, mock_cache: MagicMock, mock_conf_lib: MagicMock
    ) -> None:
        """Verify fields are fetched from Odoo and cached on a cache miss."""
        mock_cache.load_fields_get_cache.return_value = None
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.fields_get.return_value = {"name": {"type": "char"}}

        result = preflight._get_odoo_fields("dummy.conf", "res.partner")

        assert result == {"name": {"type": "char"}}
        mock_cache.load_fields_get_cache.assert_called_once_with(
            "dummy.conf", "res.partner"
        )
        mock_conf_lib.return_value.get_model.assert_called_once_with("res.partner")
        mock_model.fields_get.assert_called_once()
        mock_cache.save_fields_get_cache.assert_called_once_with(
            "dummy.conf", "res.partner", {"name": {"type": "char"}}
        )

    def test_get_odoo_fields_odoo_error(
        self,
        mock_cache: MagicMock,
        mock_conf_lib: MagicMock,
        mock_show_error_panel: MagicMock,
    ) -> None:
        """Verify None is returned and error is shown when Odoo call fails."""
        mock_cache.load_fields_get_cache.return_value = None
        mock_conf_lib.side_effect = Exception("Odoo Error")

        result = preflight._get_odoo_fields("dummy.conf", "res.partner")

        assert result is None
        mock_show_error_panel.assert_called_once()
        assert "Odoo Connection Error" in mock_show_error_panel.call_args[0][0]
        mock_cache.save_fields_get_cache.assert_not_called()


class TestGetCSVHeader:
    """Tests for the _get_csv_header helper function."""

    def test_get_csv_header_success(self, tmp_path: Path) -> None:
        """Verify _get_csv_header successfully reads a CSV file header."""
        # Create a sample CSV file
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id", "name", "email", "age"])
            writer.writerow(["1", "Alice", "alice@test.com", "25"])
            writer.writerow(["2", "Bob", "bob@test.com", "30"])

        from odoo_data_flow.lib.preflight import _get_csv_header

        result = _get_csv_header(str(csv_file), ";")

        assert result == ["id", "name", "email", "age"]

    def test_get_csv_header_file_not_found(self) -> None:
        """Verify _get_csv_header returns None when file does not exist."""
        from odoo_data_flow.lib.preflight import _get_csv_header

        result = _get_csv_header("/nonexistent.csv", ";")

        assert result is None

    def test_get_csv_header_empty_file(self, tmp_path: Path) -> None:
        """Verify _get_csv_header handles empty file gracefully."""
        csv_file = tmp_path / "empty.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("")  # Empty file

        from odoo_data_flow.lib.preflight import _get_csv_header

        result = _get_csv_header(str(csv_file), ";")

        assert result is None


class TestValidateHeader:
    """Tests for the _validate_header function."""

    def test_validate_header_passes_with_valid_fields(self) -> None:
        """Verify _validate_header passes with all valid fields."""
        csv_header = ["id", "name", "email"]
        odoo_fields = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "email": {"type": "char"},
        }

        result = preflight._validate_header(csv_header, odoo_fields, "res.partner")
        assert result is True

    def test_validate_header_fails_with_invalid_fields(
        self, mock_show_error_panel: MagicMock
    ) -> None:
        """Verify _validate_header fails and shows error for invalid fields."""
        csv_header = ["id", "name", "invalid_field"]
        odoo_fields = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        result = preflight._validate_header(csv_header, odoo_fields, "res.partner")
        assert result is False
        mock_show_error_panel.assert_called_once()
        call_args = mock_show_error_panel.call_args
        assert call_args[0][0] == "Invalid Fields Found"
        assert "invalid_field" in call_args[0][1]

    def test_validate_header_passes_with_external_id_fields(self) -> None:
        """Verify _validate_header passes with external ID fields."""
        csv_header = ["id", "name", "parent_id/id", "category_id/id"]
        odoo_fields = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "parent_id": {"type": "many2one", "relation": "res.partner"},
            "category_id": {"type": "many2many", "relation": "res.partner.category"},
        }

        result = preflight._validate_header(csv_header, odoo_fields, "res.partner")
        assert result is True

    def test_validate_header_warns_about_readonly_fields(
        self, mock_show_warning_panel: MagicMock
    ) -> None:
        """Verify _validate_header warns about readonly fields."""
        csv_header = ["id", "name", "display_name"]
        odoo_fields = {
            "id": {"type": "integer", "readonly": True, "store": True},
            "name": {"type": "char", "readonly": False, "store": True},
            "display_name": {"type": "char", "readonly": True, "store": False},
        }

        result = preflight._validate_header(csv_header, odoo_fields, "res.partner")
        assert result is True
        mock_show_warning_panel.assert_called_once()
        call_args = mock_show_warning_panel.call_args
        assert call_args[0][0] == "ReadOnly Fields Detected"
        assert "display_name" in call_args[0][1]
        assert "non-stored" in call_args[0][1]

    def test_validate_header_warns_about_multiple_readonly_fields(
        self, mock_show_warning_panel: MagicMock
    ) -> None:
        """Verify _validate_header warns about multiple readonly fields."""
        csv_header = ["id", "name", "display_name", "commercial_company_name"]
        odoo_fields = {
            "id": {"type": "integer", "readonly": True, "store": True},
            "name": {"type": "char", "readonly": False, "store": True},
            "display_name": {"type": "char", "readonly": True, "store": False},
            "commercial_company_name": {
                "type": "char",
                "readonly": True,
                "store": True,
            },
        }

        result = preflight._validate_header(csv_header, odoo_fields, "res.partner")
        assert result is True
        mock_show_warning_panel.assert_called_once()
        call_args = mock_show_warning_panel.call_args
        assert call_args[0][0] == "ReadOnly Fields Detected"
        assert "display_name" in call_args[0][1]
        assert "commercial_company_name" in call_args[0][1]
        assert "non-stored" in call_args[0][1]
        assert "1 non-stored readonly" in call_args[0][1]


def test_type_correction_check_no_corrections_needed(tmp_path: Path) -> None:
    """Test type correction check when no corrections are needed."""
    # Create a simple CSV file with clean integer data
    csv_file = tmp_path / "clean_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "name", "quantity"])
        writer.writerow(["rec1", "Product A", "100"])
        writer.writerow(["rec2", "Product B", "200"])

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock Odoo fields to simulate integer field type
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "name": {"type": "char"},
            "quantity": {"type": "integer"},
        }

        from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

        result = type_correction_check(
            PreflightMode.NORMAL,
            "test.model",
            str(csv_file),
            config,
            import_plan,
            separator=";",
            encoding="utf-8",
        )

        # Should pass when no corrections needed
        assert result is True
        # Should not create corrected file when no corrections needed
        assert "_corrected_file" not in import_plan


def test_type_correction_check_with_corrections(tmp_path: Path) -> None:
    """Test type correction check when corrections are needed and applied."""
    # Create a CSV file with float-like integer strings that need correction
    csv_file = tmp_path / "dirty_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "name", "price"])
        writer.writerow(
            ["rec1", "Product A", "99.0"]
        )  # Float string that's really integer
        writer.writerow(
            ["rec2", "Product B", "149.00"]
        )  # Float string that's really integer

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock Odoo fields to simulate integer field type
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "name": {"type": "char"},
            "price": {"type": "integer"},
        }

        from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

        result = type_correction_check(
            PreflightMode.NORMAL,
            "test.model",
            str(csv_file),
            config,
            import_plan,
            separator=";",
            encoding="utf-8",
        )

        # Should pass after applying corrections
        assert result is True
        # Should create corrected file when corrections are applied
        assert "_corrected_file" in import_plan
        assert import_plan["_corrected_file"].endswith(".csv")


def test_type_correction_check_odoo_error_handling(tmp_path: Path) -> None:
    """Test type correction check gracefully handles Odoo connection errors."""
    csv_file = tmp_path / "test_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "value"])
        writer.writerow(["rec1", "100.0"])

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock Odoo fields to raise an exception
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.side_effect = Exception("Connection failed")

        from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

        result = type_correction_check(
            PreflightMode.NORMAL,
            "test.model",
            str(csv_file),
            config,
            import_plan,
            separator=";",
            encoding="utf-8",
        )

        # Should still pass even when Odoo connection fails
        assert result is True
        # Should proceed with original file when Odoo connection fails
        assert "_corrected_file" not in import_plan


def test_type_correction_check_empty_file(tmp_path: Path) -> None:
    """Test type correction check handles empty file gracefully."""
    csv_file = tmp_path / "empty.csv"
    with open(csv_file, "w", newline="", encoding="utf-8"):
        pass  # Create empty file

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

    result = type_correction_check(
        PreflightMode.NORMAL,
        "test.model",
        str(csv_file),
        config,
        import_plan,
        separator=";",
        encoding="utf-8",
    )

    # Should pass even with empty file
    assert result is True
    # Should not create corrected file for empty file
    assert "_corrected_file" not in import_plan


def test_type_correction_check_no_header(tmp_path: Path) -> None:
    """Test type correction check handles file with no header gracefully."""
    csv_file = tmp_path / "no_header.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        f.write("")  # Empty file with no header

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

    result = type_correction_check(
        PreflightMode.NORMAL,
        "test.model",
        str(csv_file),
        config,
        import_plan,
        separator=";",
        encoding="utf-8",
    )

    # Should pass even with no header
    assert result is True
    # Should not create corrected file when no header
    assert "_corrected_file" not in import_plan


def test_type_correction_check_odoo_field_retrieval_failure(tmp_path: Path) -> None:
    """Test type correction check handles Odoo field retrieval failure gracefully."""
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "name"])
        writer.writerow(["rec1", "Test"])

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock to simulate Odoo connection failure
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields", return_value=None):
        from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

        result = type_correction_check(
            PreflightMode.NORMAL,
            "test.model",
            str(csv_file),
            config,
            import_plan,
            separator=";",
            encoding="utf-8",
        )

        # Should still pass even when Odoo field retrieval fails
        assert result is True
        # Should proceed with original file when Odoo connection fails
        assert "_corrected_file" not in import_plan


def test_type_correction_check_file_read_error(tmp_path: Path) -> None:
    """Test type correction check handles file read errors gracefully."""
    # Create a file that will be deleted to simulate read error
    csv_file = tmp_path / "unreadable.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "price"])
        writer.writerow(["rec1", "100.0"])

    # Delete the file to simulate read error
    csv_file.unlink()

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

    result = type_correction_check(
        PreflightMode.NORMAL,
        "test.model",
        str(csv_file),
        config,
        import_plan,
        separator=";",
        encoding="utf-8",
    )

    # Should still pass even when file read fails
    assert result is True
    # Should proceed with original file when read fails
    assert "_corrected_file" not in import_plan


def test_type_correction_check_polars_casting_failure(tmp_path: Path) -> None:
    """Test type correction check handles Polars casting failures gracefully."""
    csv_file = tmp_path / "casting_failure.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "price"])
        writer.writerow(["rec1", "invalid_price"])  # Invalid value that can't be cast

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock to return integer field to trigger correction logic
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "price": {"type": "integer"},
        }

        from odoo_data_flow.lib.preflight import PreflightMode, type_correction_check

        result = type_correction_check(
            PreflightMode.NORMAL,
            "test.model",
            str(csv_file),
            config,
            import_plan,
            separator=";",
            encoding="utf-8",
        )

        # Should still pass even when Polars casting fails
        assert result is True
        # May or may not create corrected file depending on logic
        # The key is it should not crash


def test_type_correction_check_main_exception_handler(tmp_path: Path) -> None:
    """Test that type correction check gracefully handles main function exceptions."""
    # Create a valid CSV file
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "price"])
        writer.writerow(["rec1", "100.0"])

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock _get_odoo_fields to return integer fields to trigger correction logic
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "price": {"type": "integer"},
        }

        # Mock polars.read_csv to raise an exception to trigger the main
        # exception handler
        with patch("odoo_data_flow.lib.preflight.pl.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = Exception("Simulated Polars read error")

            from odoo_data_flow.lib.preflight import (
                PreflightMode,
                type_correction_check,
            )

            result = type_correction_check(
                PreflightMode.NORMAL,
                "test.model",
                str(csv_file),
                config,
                import_plan,
                separator=";",
                encoding="utf-8",
            )

            # Should still return True even when main function raises exception
            assert result is True
            # Should warn about the error but proceed with original file
            assert "_corrected_file" not in import_plan


def test_type_correction_check_casting_exception_handler(tmp_path: Path) -> None:
    """Test that type correction check gracefully handles Polars casting exceptions."""
    # Create a valid CSV file with integer-like float strings
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "price"])
        writer.writerow(["rec1", "100.0"])

    config = {"test": "config"}
    import_plan: dict[str, Any] = {}

    # Mock _get_odoo_fields to return integer fields to trigger correction logic
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "price": {"type": "integer"},
        }

        # Mock write_csv to raise an exception to trigger the casting exception handler
        with patch(
            "odoo_data_flow.lib.preflight.pl.DataFrame.write_csv"
        ) as mock_write_csv:
            mock_write_csv.side_effect = Exception("Simulated Polars write error")

            from odoo_data_flow.lib.preflight import (
                PreflightMode,
                type_correction_check,
            )

            result = type_correction_check(
                PreflightMode.NORMAL,
                "test.model",
                str(csv_file),
                config,
                import_plan,
                separator=";",
                encoding="utf-8",
            )

            # Should still return True even when casting raises exception
            assert result is True
            # Should still proceed and may or may not create corrected file depending
            # on flow
    # Mock _get_odoo_fields to return integer fields to trigger correction logic
    with patch("odoo_data_flow.lib.preflight._get_odoo_fields") as mock_get_fields:
        mock_get_fields.return_value = {
            "id": {"type": "char"},
            "price": {"type": "integer"},
        }

        # Mock the DataFrame.with_columns operation to raise an exception
        # This will trigger the casting exception handler
        with patch.object(pl.DataFrame, "with_columns") as mock_with_columns:
            mock_with_columns.side_effect = Exception("Simulated Polars cast error")

            from odoo_data_flow.lib.preflight import (
                PreflightMode,
                type_correction_check,
            )

            result = type_correction_check(
                PreflightMode.NORMAL,
                "test.model",
                str(csv_file),
                config,
                import_plan,
                separator=";",
                encoding="utf-8",
            )

            # Should still return True even when casting raises exception
            assert result is True
            # Should still proceed and may or may not create corrected file depending
            # on flow
