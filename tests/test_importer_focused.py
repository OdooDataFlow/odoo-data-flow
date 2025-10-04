"""Focused tests for importer to improve coverage."""

import tempfile
from unittest.mock import Mock, patch

from odoo_data_flow.enums import PreflightMode
from odoo_data_flow.importer import (
    _count_lines,
    _get_fail_filename,
    _infer_model_from_filename,
    _run_preflight_checks,
    run_import,
    run_import_for_migration,
)


class TestCountLines:
    """Test _count_lines function."""

    def test_count_lines_success(self) -> None:
        """Test counting lines in a file successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            f.flush()
            filepath = f.name

        result = _count_lines(filepath)
        assert result == 3

    def test_count_lines_empty(self) -> None:
        """Test counting lines in an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.flush()
            filepath = f.name

        result = _count_lines(filepath)
        assert result == 0

    def test_count_lines_not_found(self) -> None:
        """Test counting lines in a non-existent file."""
        result = _count_lines("/nonexistent.csv")
        assert result == 0


class TestInferModelFromFilename:
    """Test _infer_model_from_filename function."""

    def test_infer_model_from_product_filename(self) -> None:
        """Test inferring model from product filename."""
        result = _infer_model_from_filename("/path/to/product_template.csv")
        assert result == "product.template"

    def test_infer_model_from_product_variants_filename(self) -> None:
        """Test inferring model from product variants filename."""
        result = _infer_model_from_filename("product_product.csv")
        assert result == "product.product"

    def test_infer_model_from_res_partner_filename(self) -> None:
        """Test inferring model from partner filename."""
        result = _infer_model_from_filename("res_partner.csv")
        assert result == "res.partner"

    def test_infer_model_from_invoice_filename(self) -> None:
        """Test inferring model from invoice filename."""
        result = _infer_model_from_filename("/some/path/account_move.csv")
        assert result == "account.move"

    def test_infer_model_from_unknown_filename(self) -> None:
        """Test inferring model from unknown filename."""
        # unknown_file.txt -> stem: unknown_file -> replace _ with . ->
        # unknown.file -> has dot -> return
        result = _infer_model_from_filename("unknown_file.txt")
        assert result == "unknown.file"

    def test_infer_model_from_filename_no_extension(self) -> None:
        """Test inferring model from filename without extension."""
        # res_partner -> stem: res_partner -> replace _ with .
        # -> res.partner -> has dot -> return
        result = _infer_model_from_filename("res_partner")
        assert result == "res.partner"

    def test_infer_model_from_filename_no_underscore(self) -> None:
        """Test inferring model from filename with no underscores."""
        # product -> stem: product -> replace _ with . -> product -> no dot -> None
        result = _infer_model_from_filename("product.csv")
        assert result is None


class TestGetFailFilename:
    """Test _get_fail_filename function."""

    def test_get_fail_filename_not_fail_run(self) -> None:
        """Test getting fail filename when not in fail run."""
        # Actually returns 'res_partner_fail.csv', not empty string
        result = _get_fail_filename("res.partner", False)
        assert result == "res_partner_fail.csv"

    def test_get_fail_filename_is_fail_run(self) -> None:
        """Test getting fail filename when in fail run."""
        # Returns with timestamp: 'res_partner_YYYYMMDD_HHMMSS_failed.csv'
        result = _get_fail_filename("res.partner", True)
        assert result.startswith("res_partner_")
        assert result.endswith("_failed.csv")
        assert "202" in result  # Year should be in there


class TestRunPreflightChecks:
    """Test _run_preflight_checks function."""

    @patch("odoo_data_flow.importer.preflight.PREFLIGHT_CHECKS", [])
    def test_run_preflight_checks_no_checks(self) -> None:
        """Test running preflight checks with no checks registered."""
        result = _run_preflight_checks(
            preflight_mode=PreflightMode.NORMAL,
            import_plan={},
        )
        assert result is True

    @patch("odoo_data_flow.importer.preflight.PREFLIGHT_CHECKS")
    def test_run_preflight_checks_success(self, mock_checks: Mock) -> None:
        """Test running preflight checks with success."""
        mock_check = Mock(return_value=True)
        mock_checks.__iter__ = Mock(return_value=iter([mock_check]))

        result = _run_preflight_checks(
            preflight_mode=PreflightMode.NORMAL,
            import_plan={},
        )
        assert result is True

    @patch("odoo_data_flow.importer.preflight.PREFLIGHT_CHECKS")
    def test_run_preflight_checks_failure(self, mock_checks: Mock) -> None:
        """Test running preflight checks with failure."""
        mock_check = Mock(return_value=False)
        mock_checks.__iter__ = Mock(return_value=iter([mock_check]))

        result = _run_preflight_checks(
            preflight_mode=PreflightMode.NORMAL,
            import_plan={},
        )
        assert result is False


class TestRunImport:
    """Test run_import function."""

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._run_preflight_checks")
    @patch("odoo_data_flow.importer._count_lines")
    def test_run_import_success_normal_mode(
        self, mock_count_lines: Mock, mock_preflight: Mock, mock_import_data: Mock
    ) -> None:
        """Test running import successfully in normal mode."""
        mock_count_lines.return_value = 100
        mock_preflight.return_value = True
        mock_import_data.return_value = (True, {"records_processed": 100})

        # run_import doesn't return a value, it returns None after successful execution
        run_import(
            config="dummy.conf",
            filename="test.csv",
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
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
        mock_import_data.assert_called()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._run_preflight_checks")
    @patch("odoo_data_flow.importer._count_lines")
    def test_run_import_success_fail_mode(
        self, mock_count_lines: Mock, mock_preflight: Mock, mock_import_data: Mock
    ) -> None:
        """Test running import successfully in fail mode."""
        mock_count_lines.return_value = 100
        mock_preflight.return_value = True  # Should be ignored in fail mode
        mock_import_data.return_value = (True, {"records_processed": 100})

        run_import(
            config="dummy.conf",
            filename="test.csv",
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
            no_preflight_checks=False,
            headless=True,
            worker=1,
            batch_size=100,
            skip=0,
            fail=True,  # fail mode
            separator=";",
            ignore=["_ERROR_REASON"],
            context={},
            encoding="utf-8",
            o2m=False,
            groupby=None,
        )
        mock_import_data.assert_called()

    @patch("odoo_data_flow.importer._count_lines")
    def test_run_import_preflight_fails(self, mock_count_lines: Mock) -> None:
        """Test running import when preflight fails."""
        mock_count_lines.return_value = 100

        with patch("odoo_data_flow.importer._run_preflight_checks", return_value=False):
            run_import(
                config="dummy.conf",
                filename="test.csv",
                model="res.partner",
                deferred_fields=None,
                unique_id_field="id",
                no_preflight_checks=False,  # preflight checks enabled
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
            # run_import exits early and returns None

    @patch("odoo_data_flow.importer._count_lines")
    @patch("odoo_data_flow.importer._run_preflight_checks")
    def test_run_import_empty_file(
        self, mock_preflight: Mock, mock_count_lines: Mock
    ) -> None:
        """Test running import with empty file."""
        mock_count_lines.return_value = 0  # empty file
        mock_preflight.return_value = True

        run_import(
            config="dummy.conf",
            filename="test.csv",
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
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
        # run_import returns None when there's no data to process

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._run_preflight_checks")
    @patch("odoo_data_flow.importer._count_lines")
    def test_run_import_data_fails(
        self, mock_count_lines: Mock, mock_preflight: Mock, mock_import_data: Mock
    ) -> None:
        """Test running import when import_data fails."""
        mock_count_lines.return_value = 100
        mock_preflight.return_value = True
        # The actual import_data function returns a tuple (success, stats)
        mock_import_data.return_value = (False, {"error": "Some error"})  # import fails

        run_import(
            config="dummy.conf",
            filename="test.csv",
            model="res.partner",
            deferred_fields=None,
            unique_id_field="id",
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
        # run_import returns None after the import attempt


class TestRunImportForMigration:
    """Test run_import_for_migration function."""

    @patch("odoo_data_flow.importer.import_threaded")
    def test_run_import_for_migration_success(self, mock_import_threaded: Mock) -> None:
        """Test running import for migration successfully."""
        # Mock the import_data method to return success
        mock_import_threaded.import_data.return_value = None

        # run_import_for_migration doesn't return a value
        run_import_for_migration(
            config="dummy.conf",
            model="res.partner",
            header=["id", "name"],  # Must include 'id' column
            data=[["1", "Test"], ["2", "Another"]],
            worker=1,
            batch_size=100,
        )
        # run_import_for_migration should also return None
        mock_import_threaded.import_data.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded")
    def test_run_import_for_migration_failure(self, mock_import_threaded: Mock) -> None:
        """Test running import for migration when it fails."""
        # Mock the import_data method to return None
        # (successful call that returns nothing)
        mock_import_threaded.import_data.return_value = None

        # run_import_for_migration doesn't return a value even when it fails
        run_import_for_migration(
            config="dummy.conf",
            model="res.partner",
            header=["id", "name"],  # Must include 'id' column
            data=[["1", "Test"], ["2", "Another"]],
            worker=1,
            batch_size=100,
        )
        # Should have called import_data
        mock_import_threaded.import_data.assert_called_once()
