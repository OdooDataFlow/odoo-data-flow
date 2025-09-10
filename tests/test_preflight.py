"""Test the pre-flight checker functions."""

from collections.abc import Generator
from pathlib import Path
from typing import Any
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


@pytest.fixture
def mock_cache() -> Generator[MagicMock, None, None]:
    """Fixture to mock the cache module."""
    with patch("odoo_data_flow.lib.preflight.cache") as mock_cache_module:
        yield mock_cache_module


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
        (
            mock_polars_read_csv.return_value.get_column.return_value.unique.return_value.drop_nulls.return_value.to_list.return_value
        ) = ["fr_FR"]
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
