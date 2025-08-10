"""Test the high-level import orchestrator, including pre-flight checks."""

import csv
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

from odoo_data_flow import import_threaded
from odoo_data_flow.importer import (
    _get_fail_filename,
    run_import,
    run_import_for_migration,
)


class TestRunImport:
    """Tests for the main run_import orchestrator function."""

    DEFAULT_ARGS: ClassVar[dict[str, Any]] = {
        "config": "dummy.conf",
        "filename": "res.partner.csv",
        "model": "res.partner",
        "deferred_fields": [],
        "unique_id_field": "id",
        "no_preflight_checks": True,
        "headless": True,
        "worker": 1,
        "batch_size": 100,
        "skip": 0,
        "fail": False,
        "separator": ",",
        "ignore": [],
        "context": {},
        "encoding": "utf-8",
        "o2m": False,
        "groupby": [],
    }

    @patch("odoo_data_flow.importer._infer_model_from_filename", return_value=None)
    @patch("odoo_data_flow.importer._show_error_panel")
    def test_run_import_no_model_fails(
        self, mock_show_error: MagicMock, mock_infer: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that the import fails if no model can be inferred."""
        test_args = self.DEFAULT_ARGS.copy()
        test_args["model"] = None
        run_import(**test_args)
        mock_show_error.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_run_import_routes_to_single_pass(
        self, mock_import_data: MagicMock
    ) -> None:
        """Tests that a non-deferred call routes to import_data."""
        mock_import_data.return_value = (True, {"total_records": 123})
        test_args = self.DEFAULT_ARGS.copy()
        test_args["context"] = "{}"
        run_import(**test_args)
        mock_import_data.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._count_lines", return_value=0)
    def test_run_import_fail_mode_no_records_to_retry(
        self, mock_count: MagicMock, mock_import_data: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that fail mode is skipped if the fail file is empty."""
        fail_file = tmp_path / "res_partner_fail.csv"
        fail_file.touch()

        test_args = self.DEFAULT_ARGS.copy()
        test_args.update({"filename": str(tmp_path / "res.partner.csv"), "fail": True})

        run_import(**test_args)
        mock_import_data.assert_not_called()

    @patch("odoo_data_flow.importer.log")
    @patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_fail_mode_with_records_logs_count_and_proceeds(
        self,
        mock_import_data: MagicMock,
        mock_preflight_checks: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests that fail mode with records logs the count and continues."""
        source_file = tmp_path / "res_partner.csv"
        source_file.touch()
        fail_file = tmp_path / "res_partner_fail.csv"
        fail_file.write_text("id,name\n1,a\n2,b\n3,c\n4,d\n5,e\n")
        record_count = 5

        mock_import_data.return_value = (True, {"total_records": record_count})

        test_args = self.DEFAULT_ARGS.copy()
        test_args["context"] = "{}"
        test_args.update(
            {
                "filename": str(source_file),
                "fail": True,
            }
        )

        run_import(**test_args)
        mock_import_data.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_run_import_routes_correctly(self, mock_import_data: MagicMock) -> None:
        """Tests that a standard call correctly delegates to the core engine."""
        mock_import_data.return_value = (True, {"total_records": 123})
        test_args = self.DEFAULT_ARGS.copy()
        test_args["context"] = "{}"
        test_args["deferred_fields"] = ["parent_id"]
        run_import(**test_args)
        mock_import_data.assert_called_once()
        assert mock_import_data.call_args.kwargs["deferred_fields"] == ["parent_id"]

    @patch(
        "odoo_data_flow.importer.import_threaded.conf_lib.get_connection_from_config"
    )
    @patch("odoo_data_flow.importer._show_error_panel")
    @patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
    def test_pass_1_create_fallback_failure_creates_fail_file(
        self,
        mock_preflight: MagicMock,
        mock_show_error: MagicMock,
        mock_get_conn: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests that a `create` failure during Pass 1 fallback creates a fail file."""
        source_file = tmp_path / "res.partner.csv"
        fail_file = tmp_path / "res_partner_fail.csv"
        header = ["id", "name"]
        source_data = [["new_partner", "Partner Name"]]
        with open(source_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(source_data)

        mock_model = MagicMock()
        mock_model.with_context.return_value = mock_model
        mock_model.load.return_value = {
            "ids": [],
            "messages": [{"type": "error", "message": "Batch load failed"}],
        }
        mock_model.browse.return_value.env.ref.return_value = None
        mock_model.create.side_effect = Exception("Validation Error on Create")
        mock_get_conn.return_value.get_model.return_value = mock_model

        test_args = self.DEFAULT_ARGS.copy()
        test_args.update(
            {
                "filename": str(source_file),
                "deferred_fields": [],
                "no_preflight_checks": False,
            }
        )
        run_import(**test_args)
        mock_show_error.assert_called_once()
        assert fail_file.exists()
        with open(fail_file) as f:
            reader = csv.reader(f)
            content = list(reader)
            assert len(content) == 2
            assert content[0] == [*header, "_ERROR_REASON"]
            assert content[1] == source_data[0] + ["Validation Error on Create"]

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer.Console")
    def test_fail_mode_aborts_if_fail_file_is_empty(
        self,
        mock_console: MagicMock,
        mock_import_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test fail mode aborts if fail file is empty."""
        source_file = tmp_path / "res.partner.csv"
        source_file.touch()
        fail_file = tmp_path / "res.partner_fail.csv"
        fail_file.write_text("id,name,_ERROR_REASON\n")

        test_args = self.DEFAULT_ARGS.copy()
        test_args.update(
            {
                "filename": str(source_file),
                "fail": True,
            }
        )
        run_import(**test_args)
        mock_import_data.assert_not_called()
        mock_console.return_value.print.assert_called_once()

    @patch(
        "odoo_data_flow.importer.import_threaded.conf_lib.get_connection_from_config"
    )
    def test_import_data_simple_success(
        self, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Tests a simple, successful import with no failures."""
        source_file = tmp_path / "source.csv"
        fail_file = tmp_path / "source_fail.csv"
        header = ["id", "name"]
        source_data = [["rec1", "Record 1"], ["rec2", "Record 2"]]
        with open(source_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            writer.writerows(source_data)

        mock_model = MagicMock()
        mock_model.load.return_value = {"ids": [101, 102], "messages": []}
        mock_get_conn.return_value.get_model.return_value = mock_model

        success, stats = import_threaded.import_data(
            config_file="dummy.conf",
            model="res.partner",
            unique_id_field="id",
            file_csv=str(source_file),
            fail_file=str(fail_file),
        )
        assert success is True
        assert stats["total_records"] == 2
        assert fail_file.exists()
        with open(fail_file, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1

    @patch(
        "odoo_data_flow.importer.import_threaded.conf_lib.get_connection_from_config"
    )
    def test_import_data_two_pass_success(
        self, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Tests a successful two-pass import with deferred fields."""
        source_file = tmp_path / "source.csv"
        header = ["id", "name", "parent_id/id"]
        source_data = [
            ["parent1", "Parent One", ""],
            ["child1", "Child One", "parent1"],
        ]
        with open(source_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            writer.writerows(source_data)

        mock_model = MagicMock()
        mock_model.with_context.return_value = mock_model
        mock_model.load.return_value = {"ids": [10, 20], "messages": []}
        mock_get_conn.return_value.get_model.return_value = mock_model

        success, _ = import_threaded.import_data(
            config_file="dummy.conf",
            model="res.partner",
            unique_id_field="id",
            file_csv=str(source_file),
            deferred_fields=["parent_id"],
            separator=";",
        )
        assert success is True
        mock_model.write.assert_called_once_with(
            [20], {"parent_id": 10}, context={"tracking_disable": True}
        )


class TestRunImportEdgeCases:
    """Tests for edge cases and error handling in the importer."""

    def test_get_fail_filename_recovery_mode(self) -> None:
        """Tests that _get_fail_filename creates a timestamped name in fail mode."""
        filename = _get_fail_filename("res.partner", is_fail_run=True)
        assert "res_partner" in filename
        assert "failed" in filename
        assert any(char.isdigit() for char in filename)

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_run_import_fail_mode_ignore_is_none(
        self, mock_import_data: MagicMock, tmp_path: Path
    ) -> None:
        """Test fail mode with no ignore list."""
        fail_file = tmp_path / "res_partner_fail.csv"
        fail_file.write_text("id,name,_ERROR_REASON\n1,a,error")

        args = TestRunImport.DEFAULT_ARGS.copy()
        args.update(
            {
                "filename": str(tmp_path / "res.partner.csv"),
                "fail": True,
                "ignore": None,
            }
        )

        mock_import_data.return_value = (True, {"total_records": 1})
        run_import(**args)
        called_kwargs = mock_import_data.call_args.kwargs
        assert "_ERROR_REASON" in called_kwargs["ignore"]

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    def test_run_import_for_migration_success(
        self, mock_import_data: MagicMock
    ) -> None:
        """Tests the successful execution of run_import_for_migration."""
        header = ["id", "name"]
        data = [[1, "Test"], [2, "Another"]]

        run_import_for_migration(
            config="dummy.conf",
            model="res.partner",
            header=header,
            data=data,
        )

        mock_import_data.assert_called_once()
        kwargs = mock_import_data.call_args.kwargs
        assert kwargs["model"] == "res.partner"
        assert kwargs["unique_id_field"] == "id"
        assert "tmp" in kwargs["file_csv"]
        assert kwargs["context"] == {"tracking_disable": True}

    @patch("odoo_data_flow.importer._show_error_panel")
    def test_run_import_file_not_found(self, mock_show_error: MagicMock) -> None:
        """Tests that the import fails if the source file is not found."""
        test_args = TestRunImport.DEFAULT_ARGS.copy()
        test_args["filename"] = "non_existent_file.csv"
        run_import(**test_args)
        mock_show_error.assert_called_once()

    @patch("odoo_data_flow.importer._show_error_panel")
    def test_run_import_invalid_json_context(
        self, mock_show_error: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that the import fails if the context is not valid JSON."""
        source_file = tmp_path / "source.csv"
        source_file.touch()
        test_args = TestRunImport.DEFAULT_ARGS.copy()
        test_args["filename"] = str(source_file)
        test_args["context"] = "this is not json"
        run_import(**test_args)
        mock_show_error.assert_called_once()

    @patch("odoo_data_flow.importer.import_threaded.import_data")
    @patch("odoo_data_flow.importer._run_preflight_checks", return_value=True)
    @patch("odoo_data_flow.importer.Panel")
    def test_run_import_summary_panel(
        self,
        mock_panel: MagicMock,
        mock_preflight: MagicMock,
        mock_import_data: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tests that the summary panel is displayed with the correct stats."""
        source_file = tmp_path / "source.csv"
        source_file.touch()
        mock_import_data.return_value = (
            True,
            {"total_records": 10, "created_records": 8, "updated_relations": 2},
        )
        test_args = TestRunImport.DEFAULT_ARGS.copy()
        test_args["filename"] = str(source_file)
        run_import(**test_args)
        mock_panel.assert_called_once()
        renderable = mock_panel.call_args[0][0]
        assert (
            "Import for [cyan]res.partner[/cyan] finished successfully." in renderable
        )


@patch("odoo_data_flow.importer.import_threaded.import_data")
def test_run_import_with_vies_disabled(
    mock_import_data: MagicMock, tmp_path: Path
) -> None:
    """Tests that the VIES check is disabled when the context is set."""
    source_file = tmp_path / "source.csv"
    source_file.touch()
    mock_import_data.return_value = (True, {"total_records": 1})
    test_args = TestRunImport.DEFAULT_ARGS.copy()
    test_args["filename"] = str(source_file)
    test_args["context"] = {"vat_check_vies": False}
    run_import(**test_args)
    mock_import_data.assert_called_once()
    assert mock_import_data.call_args.kwargs["context"] == {"vat_check_vies": False}


@patch("odoo_data_flow.importer.relational_import.run_direct_relational_import")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_orchestrates_direct_relational_strategy(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_relational_import: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify the importer correctly orchestrates the direct relational strategy."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name,category_id\n1,test,cat1")

    # Create a dummy temp file for the relational import to return
    relational_temp_file = tmp_path / "relational_temp.csv"
    relational_temp_file.touch()

    def preflight_side_effect(*args: Any, **kwargs: Any) -> bool:
        kwargs["import_plan"]["strategies"] = {
            "category_id": {
                "strategy": "direct_relational_import",
                "type": "many2many",
                "relation_table": "res.partner.category.rel",
                "relation_field": "partner_id",
                "relation": "category_id",
            }
        }
        return True

    mock_preflight.side_effect = preflight_side_effect
    mock_import_data.return_value = (True, {"id_map": {"p1": 1}})
    mock_relational_import.return_value = {
        "file_csv": str(relational_temp_file),
        "model": "res.partner.category.rel",
        "unique_id_field": "partner_id",
    }

    test_args = TestRunImport.DEFAULT_ARGS.copy()
    test_args["filename"] = str(source_file)
    test_args["no_preflight_checks"] = False

    run_import(**test_args)
    # The main import_data is called for the main file and for the relational file.
    assert mock_import_data.call_count == 2
    mock_relational_import.assert_called_once()
    call_args = mock_relational_import.call_args[0]
    assert call_args[1] == "res.partner"
    assert call_args[2] == "category_id"


@patch("odoo_data_flow.importer.relational_import.run_write_tuple_import")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_orchestrates_write_tuple_strategy(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_write_tuple_import: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify the importer correctly orchestrates the write tuple strategy."""
    source_file = tmp_path / "source.csv"
    source_file.write_text("id,name,category_id\n1,test,cat1")

    def preflight_side_effect(*args: Any, **kwargs: Any) -> bool:
        kwargs["import_plan"]["strategies"] = {
            "category_id": {
                "strategy": "write_tuple",
                "type": "many2many",
                "relation_table": "res.partner.category.rel",
                "relation_field": "partner_id",
                "relation": "category_id",
            }
        }
        return True

    mock_preflight.side_effect = preflight_side_effect
    mock_import_data.return_value = (True, {"id_map": {"p1": 1}})

    test_args = TestRunImport.DEFAULT_ARGS.copy()
    test_args["filename"] = str(source_file)
    test_args["no_preflight_checks"] = False

    run_import(**test_args)
    mock_import_data.assert_called_once()
    mock_write_tuple_import.assert_called_once()
    call_args = mock_write_tuple_import.call_args[0]
    assert call_args[1] == "res.partner"
    assert call_args[2] == "category_id"


@patch("odoo_data_flow.importer.relational_import.run_write_o2m_tuple_import")
@patch("odoo_data_flow.importer.relational_import.run_write_tuple_import")
@patch("odoo_data_flow.importer.import_threaded.import_data")
@patch("odoo_data_flow.importer._run_preflight_checks")
def test_run_import_orchestrates_combined_strategies(
    mock_preflight: MagicMock,
    mock_import_data: MagicMock,
    mock_write_tuple: MagicMock,
    mock_write_o2m: MagicMock,
    tmp_path: Path,
) -> None:
    """Verify the importer correctly orchestrates multiple strategies in one run."""
    source_file = tmp_path / "source.csv"
    source_file.write_text(
        "id,name,parent_id,category_id,line_ids\n"
        'p1,Parent,,cat1,"[{"product": "prodA"}]"\n'
        "c1,Child,p1,cat2,\n"
    )

    def preflight_side_effect(*args: Any, **kwargs: Any) -> bool:
        import_plan = kwargs["import_plan"]
        import_plan["deferred_fields"] = ["parent_id", "category_id", "line_ids"]
        import_plan["strategies"] = {
            "category_id": {
                "strategy": "write_tuple",
                "relation": "res.partner.category",
            },
            "line_ids": {"strategy": "write_o2m_tuple"},
        }
        return True

    mock_preflight.side_effect = preflight_side_effect
    mock_import_data.return_value = (True, {"id_map": {"p1": 1, "c1": 2}})

    test_args = TestRunImport.DEFAULT_ARGS.copy()
    test_args["filename"] = str(source_file)
    test_args["no_preflight_checks"] = False

    run_import(**test_args)
    mock_import_data.assert_called_once()
    mock_write_tuple.assert_called_once()
    mock_write_o2m.assert_called_once()
    assert mock_write_tuple.call_args[0][2] == "category_id"
    assert mock_write_o2m.call_args[0][2] == "line_ids"
