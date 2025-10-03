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
    mock_import_data.return_value = (
        True,
        {"total_records": 1, "id_map": {"1": 1}},
    )

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


@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer.Console")
def test_run_import_fail_mode_no_records(
    mock_console: MagicMock, mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Test fail mode when the fail file has no records to retry."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    fail_file = tmp_path / "res_partner_fail.csv"
    fail_file.write_text("id,name\n")  # Only a header

    run_import(
        config="dummy.conf",
        filename=str(source_file),
        model="res.partner",
        fail=True,
        deferred_fields=None,
        unique_id_field=None,
        no_preflight_checks=True,
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
    mock_import_data.assert_not_called()
    mock_console.return_value.print.assert_called_once()
    assert (
        "No records to retry"
        in mock_console.return_value.print.call_args[0][0].renderable
    )


@patch("odoo_data_flow.importer.sort.sort_for_self_referencing")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_sort_strategy_already_sorted(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_sort: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the sort strategy when the file is already sorted."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    mock_sort.return_value = True  # Indicates file is already sorted

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
    assert mock_import_data.call_args.kwargs["file_csv"] == str(source_file)


@patch("odoo_data_flow.importer._show_error_panel")
def test_run_import_invalid_json_type_context(mock_show_error: MagicMock) -> None:
    """Test that run_import handles context that is not a JSON dict."""
    # Using Any type to bypass type checking for this specific test case
    # This tests the runtime error handling for invalid context types
    context: Any = '["not", "a", "dict"]'  # Valid JSON, but not a dict

    run_import(
        config="dummy.conf",
        filename="dummy.csv",
        model="res.partner",
        context=context,
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
    assert "must be a valid JSON dictionary" in mock_show_error.call_args[0][1]


@patch("odoo_data_flow.importer.cache.save_id_map")
@patch("odoo_data_flow.importer.relational_import.run_direct_relational_import")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_with_relational_strategy(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_run_direct_relational: MagicMock,
    mock_save_cache: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that relational import strategies are called in Pass 2."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name,tags\np1,Partner 1,tag1,tag2")

    def preflight_side_effect(*args: Any, **kwargs: Any) -> bool:
        kwargs["import_plan"]["strategies"] = {
            "tags": {"strategy": "direct_relational_import"}
        }
        return True

    mock_preflight.side_effect = preflight_side_effect
    # Pass 1 successful, returns an id_map
    mock_import_data.return_value = (True, {"id_map": {"p1": 1}})
    # Pass 2 (from relational) returns None, so no third import call
    mock_run_direct_relational.return_value = None

    run_import(
        config=str(tmp_path / "dummy.conf"),
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
        separator=",",
        ignore=None,
        context={},
        encoding="utf-8",
        o2m=False,
        groupby=None,
    )

    assert mock_import_data.call_count == 1  # Only the first pass
    mock_run_direct_relational.assert_called_once()
    mock_save_cache.assert_called_once()


"""Additional tests for importer.py to improve coverage."""


def test_count_lines_file_not_found() -> None:
    """Test _count_lines with non-existent file."""
    result = _count_lines("/path/that/does/not/exist.csv")
    assert result == 0


def test_count_lines_with_content() -> None:
    """Test _count_lines with actual content."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        result = _count_lines(temp_path)
        assert result == 3
    finally:
        Path(temp_path).unlink()


def test_infer_model_from_filename() -> None:
    """Test _infer_model_from_filename with various patterns."""
    # Test with standard patterns (should contain dots after conversion)
    assert _infer_model_from_filename("res_partner.csv") == "res.partner"
    assert _infer_model_from_filename("account_move_line.csv") == "account.move.line"
    assert _infer_model_from_filename("product_product.csv") == "product.product"

    # Test with path
    assert _infer_model_from_filename("/some/path/res_partner.csv") == "res.partner"

    # Test with no match (no underscore to convert to dot)
    assert _infer_model_from_filename("unknown.csv") is None  # "unknown" has no dot

    # Test with suffixes that should be removed
    assert _infer_model_from_filename("res_partner_fail.csv") == "res.partner"
    assert _infer_model_from_filename("res_partner_transformed.csv") == "res.partner"
    assert _infer_model_from_filename("res_partner_123.csv") == "res.partner"
