"""Test the main importer orchestrator."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from odoo_data_flow.importer import (
    _count_lines,
    _get_fail_filename,
    _infer_model_from_filename,
    run_import,
    run_import_for_migration,
)


class TestFilenameUtils:
    """Tests for filename and path utility functions."""

    def test_count_lines(self, tmp_path: Path) -> None:
        """Test that line counting works correctly."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line1\nline2\nline3")
        assert _count_lines(str(file_path)) == 3

    def test_infer_model_from_filename(self) -> None:
        """Test model name inference from various filename formats."""
        assert _infer_model_from_filename("res_partner.csv") == "res.partner"
        assert _infer_model_from_filename("sale_order_line.csv") == "sale.order.line"
        assert _infer_model_from_filename("x_custom_model.csv") == "x.custom.model"
        assert _infer_model_from_filename("res_partner_fail.csv") == "res.partner"
        assert _infer_model_from_filename("res_users_123.csv") == "res.users"

    def test_get_fail_filename_recovery_mode(self) -> None:
        """Tests that _get_fail_filename creates a timestamped name in fail mode."""
        filename = _get_fail_filename("res.partner", is_fail_run=True)
        assert "res_partner" in filename
        assert "failed" in filename
        assert any(char.isdigit() for char in filename)


class TestRunImport:
    """Tests for the main run_import orchestrator function."""

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._run_preflight_checks")
    def test_run_import_success_path(
        self,
        mock_preflight: MagicMock,
        mock_import_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test the successful execution path of run_import."""
        # Arrange
        source_file = tmp_path / "source.csv"
        source_file.touch()
        mock_preflight.return_value = True
        mock_import_data.return_value = (True, {"total_records": 1})

        # Act
        run_import(
            config="dummy.conf",
            filename=str(source_file),
            model="res.partner",
            deferred_fields=None,
            unique_id_field=None,
            no_preflight_checks=False,
            headless=True,
            worker=1,
            batch_size=100,
            skip=0,
            fail=False,
            separator=";",
            ignore=None,
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )

        # Assert
        mock_preflight.assert_called_once()
        mock_import_data.assert_called_once()

    @patch("odoo_data_flow.importer._infer_model_from_filename")
    @patch("odoo_data_flow.importer._show_error_panel")
    def test_run_import_fails_if_model_not_found(
        self,
        mock_show_error: MagicMock,
        mock_infer_model: MagicMock,
    ) -> None:
        """Test that the import aborts if no model can be determined."""
        # Arrange
        mock_infer_model.return_value = None

        # Act
        run_import(
            config="dummy.conf",
            filename="no_model.csv",
            model=None,  # No model provided
            deferred_fields=None,
            unique_id_field=None,
            no_preflight_checks=False,
            headless=True,
            worker=1,
            batch_size=100,
            skip=0,
            fail=False,
            separator=";",
            ignore=None,
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )

        # Assert
        mock_show_error.assert_called_once()
        assert "Model Not Found" in mock_show_error.call_args[0]

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_import_data_simple_success(
        self,
        mock_import_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests a simple, successful import with no failures."""
        source_file = tmp_path / "source.csv"
        source_file.touch()
        mock_import_data.return_value = (True, {"created_records": 2})

        run_import(
            config=str(source_file),
            filename=str(source_file),
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
            no_preflight_checks=True,
            headless=True,
            worker=1,
            batch_size=100,
            skip=0,
            fail=False,
            separator=";",
            ignore=[],
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )
        mock_import_data.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_import_data_two_pass_success(
        self,
        mock_import_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests a successful two-pass import with deferred fields."""
        source_file = tmp_path / "source.csv"
        source_file.touch()
        mock_import_data.return_value = (True, {"created_records": 2})

        run_import(
            config=str(source_file),
            filename=str(source_file),
            model="res.partner",
            deferred_fields=["parent_id"],
            unique_id_field="id",
            no_preflight_checks=True,
            headless=True,
            worker=1,
            batch_size=100,
            skip=0,
            fail=False,
            separator=";",
            ignore=[],
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )
        mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks", return_value=False)
def test_run_import_preflight_fails(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that the import aborts if preflight checks fail."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_import_data.assert_not_called()


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
def test_run_import_fail_mode(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the fail mode logic."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    fail_file = tmp_path / "res_partner_fail.csv"
    fail_file.write_text("id,name\n1,test")
    mock_import_data.return_value = (True, {"total_records": 1})

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        fail=True,
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    assert mock_import_data.call_args.kwargs["file_csv"] == str(fail_file)


@patch("odoo_data_flow.importer.sort.sort_for_self_referencing")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_sort_strategy(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_sort: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the sort and one pass load strategy."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    sorted_file = tmp_path / "sorted.csv"
    mock_sort.return_value = str(sorted_file)

    def preflight_side_effect(*args: Any, **kwargs: Any) -> bool:
        kwargs["import_plan"]["strategy"] = "sort_and_one_pass_load"
        kwargs["import_plan"]["id_column"] = "id"
        kwargs["import_plan"]["parent_column"] = "parent_id"
        return True

    mock_preflight.side_effect = preflight_side_effect
    mock_import_data.return_value = (True, {"total_records": 1})

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_sort.assert_called_once()
    assert mock_import_data.call_args.kwargs["file_csv"] == str(sorted_file)


@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_for_migration(mock_import_data: MagicMock) -> None:
    """Test the run_import_for_migration function."""
    mock_import_data.return_value = (True, {})
    run_import_for_migration(
        config="dummy.conf",
        model="res.partner",
        header=["id", "name"],
        data=[[1, "test"]],
    )
    mock_import_data.assert_called_once()


@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_invalid_context(mock_show_error: MagicMock) -> None:
    """Test that run_import handles invalid context."""
    run_import(
        config="dummy.conf",
        filename="dummy.csv",
        model="res.partner",
        context="not a dict",
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=True,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        fail=False,
        separator=";",
        ignore=None,
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_show_error.assert_called_once()


@patch("odoo_data_flow.importer.relational_import.run_direct_relational_import")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_fail_mode_with_strategies(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_relational_import: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that relational strategies are skipped in fail mode."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    fail_file = tmp_path / "res_partner_fail.csv"
    fail_file.write_text("id,name\n1,test")

    def preflight_side_effect(*_args: Any, **kwargs: Any) -> bool:
        kwargs["import_plan"]["strategies"] = {
            "field": {"strategy": "direct_relational_import"}
        }
        return True

    mock_preflight.side_effect = preflight_side_effect
    mock_import_data.return_value = (True, {"total_records": 1, "id_map": {"1": 1}})

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        fail=True,
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=False,
        headless=True,
        worker=1,
        batch_size=100,
        skip=0,
        separator=";",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )
    mock_import_data.assert_called_once()
    mock_relational_import.assert_not_called()
