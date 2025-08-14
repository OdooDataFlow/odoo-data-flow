"""Tests for the refactored, low-level, multi-threaded import logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress

from odoo_data_flow.import_threaded import (
    _create_batch_individually,
    _create_batches,
    _execute_load_batch,
    _format_odoo_error,
    _orchestrate_pass_1,
    _orchestrate_pass_2,
    _read_data_file,
    _setup_fail_file,
    import_data,
)


class TestImportDataRefactored:
    """Tests for the main `import_data` orchestrator."""

    @patch("odoo_data_flow.import_threaded._read_data_file")
    @patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
    @patch("odoo_data_flow.import_threaded._run_threaded_pass")
    def test_import_data_success_path_no_defer(
        self,
        mock_run_pass: MagicMock,
        mock_get_conn: MagicMock,
        mock_read_file: MagicMock,
    ) -> None:
        """Test a successful single-pass import (no deferred fields)."""
        # Arrange
        mock_read_file.return_value = (["id", "name"], [["xml_a", "A"]])
        mock_run_pass.return_value = (
            {"id_map": {"xml_a": 101}, "failed_lines": []},  # results dict
            False,  # aborted = False
        )

        mock_get_conn.return_value.get_model.return_value = MagicMock()

        # Act
        result, _ = import_data(
            config_file="dummy.conf",
            model="res.partner",
            unique_id_field="id",
            file_csv="dummy.csv",
        )

        # Assert
        assert result is True
        mock_run_pass.assert_called_once()  # Only Pass 1 should run

    @patch("odoo_data_flow.import_threaded._read_data_file")
    @patch("odoo_data_flow.import_threaded.conf_lib.get_connection_from_config")
    @patch("odoo_data_flow.import_threaded._run_threaded_pass")
    def test_import_data_success_path_with_defer(
        self,
        mock_run_pass: MagicMock,
        mock_get_conn: MagicMock,
        mock_read_file: MagicMock,
    ) -> None:
        """Test a successful two-pass import (with deferred fields)."""
        # Arrange
        mock_read_file.return_value = (
            ["id", "name", "parent_id"],
            [["xml_a", "A", ""], ["xml_b", "B", "xml_a"]],
        )
        # Simulate results for Pass 1 and Pass 2
        mock_run_pass.side_effect = [
            (
                {"id_map": {"xml_a": 101, "xml_b": 102}, "failed_lines": []},
                False,
            ),  # Pass 1 (results, aborted)
            (
                {"failed_writes": []},
                False,
            ),  # Pass 2 (results, aborted)
        ]
        mock_get_conn.return_value.get_model.return_value = MagicMock()

        # Act
        result = import_data(
            config_file="dummy.conf",
            model="res.partner",
            unique_id_field="id",
            file_csv="dummy.csv",
            deferred_fields=["parent_id"],
        )

        # Assert
        assert result[0] is True
        assert mock_run_pass.call_count == 2  # Both passes should run

    @patch("odoo_data_flow.import_threaded._read_data_file")
    def test_import_data_fails_if_unique_id_not_in_header(
        self, mock_read_file: MagicMock
    ) -> None:
        """Test that the import fails if the unique_id_field is missing."""
        # Arrange
        mock_read_file.return_value = (["name"], [["A"]])  # No 'id' column

        # Act
        result, _ = import_data(
            config_file="dummy.conf",
            model="res.partner",
            unique_id_field="id",  # We expect 'id' but it's not there
            file_csv="dummy.csv",
        )

        # Assert
        assert result is False

    @patch("odoo_data_flow.import_threaded._create_batches")
    @patch("odoo_data_flow.import_threaded._run_threaded_pass")
    def test_orchestrate_pass_1_does_not_sort_for_o2m(
        self, mock_run_pass: MagicMock, mock_create_batches: MagicMock
    ) -> None:
        """Verify Pass 1 does NOT sort data when o2m is True."""
        mock_run_pass.return_value = ({}, False)
        header = ["id", "name", "parent_id"]
        data = [
            ["child1", "C1", "parent1"],
            ["parent1", "P1", ""],
        ]

        with Progress() as progress:
            _orchestrate_pass_1(
                progress,
                MagicMock(),
                "res.partner",
                header,
                data,
                "id",
                [],
                [],
                {},
                None,
                None,
                1,
                10,
                o2m=True,
                split_by_cols=None,
            )

        # Check that the data passed to _create_batches was NOT sorted
        call_args = mock_create_batches.call_args[0]
        unsorted_data = call_args[0]
        assert unsorted_data[0][0] == "child1"
        assert unsorted_data[1][0] == "parent1"


class TestExecuteLoadBatch:
    """Tests for the _execute_load_batch function's resilience features."""

    @patch("odoo_data_flow.import_threaded._create_batch_individually")
    def test_batch_scales_down_on_memory_error(
        self, mock_create_individually: MagicMock
    ) -> None:
        """Verify batch size is reduced on memory errors and eventually succeeds."""
        mock_model = MagicMock()
        # Fail on batches of 4, then 2, then succeed on 1
        mock_model.load.side_effect = [
            Exception("out of memory"),
            Exception("memory error"),
            {"ids": [1]},
            {"ids": [2]},
            {"ids": [3]},
            {"ids": [4]},
        ]
        mock_progress = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": mock_progress,
            "unique_id_field_index": 0,
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [
            ["rec1", "A"],
            ["rec2", "B"],
            ["rec3", "C"],
            ["rec4", "D"],
        ]

        result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

        assert result["success"] is True
        assert len(result["id_map"]) == 4
        assert result["id_map"] == {"rec1": 1, "rec2": 2, "rec3": 3, "rec4": 4}
        assert mock_model.load.call_count == 6
        mock_create_individually.assert_not_called()
        mock_progress.console.print.assert_any_call(
            "[yellow]WARN:[/] Batch 1 hit scalable error. "
            "Reducing chunk size to 2 and retrying."
        )
        mock_progress.console.print.assert_any_call(
            "[yellow]WARN:[/] Batch 1 hit scalable error. "
            "Reducing chunk size to 1 and retrying."
        )

    @patch("odoo_data_flow.import_threaded._create_batch_individually")
    def test_batch_scales_down_on_gateway_error(
        self, mock_create_individually: MagicMock
    ) -> None:
        """Verify batch size is reduced on 502 gateway errors."""
        mock_model = MagicMock()
        mock_model.load.side_effect = [
            Exception("502 Bad Gateway"),
            {"ids": [1, 2]},
            {"ids": [3, 4]},
        ]
        mock_progress = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": mock_progress,
            "unique_id_field_index": 0,
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [["rec1", "A"], ["rec2", "B"], ["rec3", "C"], ["rec4", "D"]]

        result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

        assert result["success"] is True
        assert len(result["id_map"]) == 4
        assert mock_model.load.call_count == 3
        mock_create_individually.assert_not_called()
        mock_progress.console.print.assert_called_once_with(
            "[yellow]WARN:[/] Batch 1 hit scalable error. "
            "Reducing chunk size to 2 and retrying."
        )

    @patch("odoo_data_flow.import_threaded._create_batch_individually")
    def test_batch_falls_back_for_non_scalable_error(
        self, mock_create_individually: MagicMock
    ) -> None:
        """Verify fallback to create for regular errors."""
        mock_model = MagicMock()
        mock_model.load.side_effect = [ValueError("Invalid field value")]
        mock_create_individually.return_value = {
            "id_map": {"rec1": 1},
            "failed_lines": [["rec2", "B", "Error"]],
        }
        mock_progress = MagicMock()
        thread_state = {
            "model": mock_model,
            "progress": mock_progress,
            "unique_id_field_index": 0,
            "ignore_list": [],
        }
        batch_header = ["id", "name"]
        batch_lines = [["rec1", "A"], ["rec2", "B"]]

        result = _execute_load_batch(thread_state, batch_lines, batch_header, 1)

        assert result["success"] is True
        assert result["id_map"] == {"rec1": 1}
        assert len(result["failed_lines"]) == 1
        mock_model.load.assert_called_once()
        mock_create_individually.assert_called_once()


class TestBatchingHelpers:
    """Tests for the batch creation helper functions."""

    def test_create_batches_handles_o2m_format(self) -> None:
        """Test _create_batches with the o2m flag enabled.

        Verifies that records with empty key fields are correctly grouped with
        their preceding parent record into a single batch.
        """
        # --- Arrange ---
        header = ["id", "name", "line_item"]
        data = [
            ["order1", "Order One", "item_A"],
            ["", "", "item_B"],  # Child of order1
            ["order2", "Order Two", "item_C"],
            ["", "", "item_D"],  # Child of order2
            ["", "", "item_E"],  # Child of order2
            ["order3", "Order Three", "item_F"],
        ]

        # --- Act ---
        batches = list(
            _create_batches(
                data=data,
                split_by_cols=None,  # Not grouping by column value
                header=header,
                batch_size=10,  # Batch size is large enough to not interfere
                o2m=True,
            )
        )

        # --- Assert ---
        assert len(batches) == 3
        assert batches[0][1] == [
            ["order1", "Order One", "item_A"],
            ["", "", "item_B"],
        ]
        assert batches[1][1] == [
            ["order2", "Order Two", "item_C"],
            ["", "", "item_D"],
            ["", "", "item_E"],
        ]
        assert batches[2][1] == [
            ["order3", "Order Three", "item_F"],
        ]

    def test_create_batches_no_data(self) -> None:
        """Test that _create_batches handles empty data."""
        header = ["id", "name"]
        data: list[list[str]] = []
        batches = list(_create_batches(data, None, header, 10, False))
        assert len(batches) == 0


class TestPass2Batching:
    """Tests for the Pass 2 batching and writing logic."""

    @patch("odoo_data_flow.import_threaded._run_threaded_pass")
    def test_pass_2_groups_writes_correctly(self, mock_run_pass: MagicMock) -> None:
        """Verify that Pass 2 groups records by identical write values."""
        # Arrange
        mock_run_pass.return_value = ({}, False)  # Simulate a successful run
        mock_model = MagicMock()
        header = ["id", "name", "parent_id", "user_id"]
        all_data = [
            ["c1", "C1", "p1", "u1"],
            ["c2", "C2", "p1", "u1"],
            ["c3", "C3", "p2", "u1"],
            ["c4", "C4", "p2", "u2"],
        ]
        id_map = {
            "c1": 1,
            "c2": 2,
            "c3": 3,
            "c4": 4,
            "p1": 101,
            "p2": 102,
            "u1": 201,
            "u2": 202,
        }
        deferred_fields = ["parent_id", "user_id"]

        # Act
        with Progress() as progress:
            _orchestrate_pass_2(
                progress,
                mock_model,
                "res.partner",
                header,
                all_data,
                "id",
                id_map,
                deferred_fields,
                {},
                MagicMock(),
                MagicMock(),
                max_connection=1,
                batch_size=10,
            )

        # Assert
        # We expect two separate write calls because the vals are different
        assert mock_run_pass.call_count == 1

        # Get the batches that were passed to the runner
        call_args = mock_run_pass.call_args[0]
        batches = list(call_args[2])  # The batches iterable

        assert len(batches) == 3  # Three unique sets of values to write

        # Convert batches to a more easily searchable dict
        batch_dict = {
            frozenset(vals.items()): ids for (ids, vals) in [b[1] for b in batches]
        }

        # Check group 1: parent=p1, user=u1
        group1_key = frozenset({"parent_id": 101, "user_id": 201}.items())
        assert group1_key in batch_dict
        assert sorted(batch_dict[group1_key]) == [1, 2]

        # Check group 2: parent=p2, user=u1
        group2_key = frozenset({"parent_id": 102, "user_id": 201}.items())
        assert group2_key in batch_dict
        assert batch_dict[group2_key] == [3]

        # Check group 3: parent=p2, user=u2
        group3_key = frozenset({"parent_id": 102, "user_id": 202}.items())
        assert group3_key in batch_dict
        assert batch_dict[group3_key] == [4]

    @patch("odoo_data_flow.import_threaded._run_threaded_pass")
    def test_pass_2_handles_failed_batch(self, mock_run_pass: MagicMock) -> None:
        """Verify that a failed batch write in Pass 2 is handled correctly."""
        # Arrange
        mock_fail_writer = MagicMock()
        mock_model = MagicMock()

        header = ["id", "name", "parent_id"]
        all_data = [["c1", "C1", "p1"], ["c2", "C2", "p1"]]
        id_map = {"c1": 1, "c2": 2, "p1": 101}
        deferred_fields = ["parent_id"]

        # Simulate a failure from the threaded runner for this batch
        failed_write_result = {
            "failed_writes": [
                (1, {"parent_id": 101}, "Access Error"),
                (2, {"parent_id": 101}, "Access Error"),
            ],
        }
        mock_run_pass.return_value = (failed_write_result, False)  # result, aborted

        # Act
        with Progress() as progress:
            result = _orchestrate_pass_2(
                progress,
                mock_model,
                "res.partner",
                header,
                all_data,
                "id",
                id_map,
                deferred_fields,
                {},
                mock_fail_writer,
                MagicMock(),  # fail_handle
                max_connection=1,
                batch_size=10,
            )

        # Assert
        assert result[0] is False  # The orchestration should report failure
        mock_fail_writer.writerows.assert_called_once()

        # Check that the rows written to the fail file are correct
        failed_rows = mock_fail_writer.writerows.call_args[0][0]
        assert len(failed_rows) == 2
        assert failed_rows[0] == ["c1", "C1", "p1", "Access Error"]
        assert failed_rows[1] == ["c2", "C2", "p1", "Access Error"]

    def test_orchestrate_pass_2_no_relations(self) -> None:
        """Test that Pass 2 handles no relations to update."""
        mock_model = MagicMock()
        header = ["id", "name"]
        all_data = [["c1", "C1"], ["c2", "C2"]]
        id_map = {"c1": 1, "c2": 2}
        deferred_fields: list[str] = []
        with Progress() as progress:
            result, updates = _orchestrate_pass_2(
                progress,
                mock_model,
                "res.partner",
                header,
                all_data,
                "id",
                id_map,
                deferred_fields,
                {},
                None,
                None,
                1,
                10,
            )
        assert result is True
        assert updates == 0


class TestImportThreadedEdgeCases:
    """Tests for edge cases and error handling in import_threaded.py."""

    def test_format_odoo_error_not_a_string(self) -> None:
        """Test that _format_odoo_error handles non-string errors."""
        error_obj = {"key": "value"}
        formatted = _format_odoo_error(error_obj)
        assert formatted == "{'key': 'value'}"

    def test_format_odoo_error_fallback(self) -> None:
        """Test that _format_odoo_error handles non-dictionary strings."""
        error_string = "A simple error message"
        formatted = _format_odoo_error(error_string)
        assert formatted == "A simple error message"

    def test_read_data_file_not_found(self) -> None:
        """Test that _read_data_file handles a FileNotFoundError."""
        header, data = _read_data_file("non_existent_file.csv", ",", "utf-8", 0)
        assert header == []
        assert data == []

    @patch("builtins.open", side_effect=ValueError("bad file"))
    def test_read_data_file_general_exception(self, mock_open: MagicMock) -> None:
        """Test that _read_data_file handles a general exception."""
        with pytest.raises(ValueError):
            _read_data_file("any.csv", ",", "utf-8", 0)

    def test_read_data_file_no_id_column(self, tmp_path: Path) -> None:
        """Test that a ValueError is raised if the 'id' column is missing."""
        source_file = tmp_path / "source.csv"
        source_file.write_text("name,age\nAlice,30")
        with pytest.raises(
            ValueError, match="Source file must contain an 'id' column."
        ):
            _read_data_file(str(source_file), ",", "utf-8", 0)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_setup_fail_file_os_error(self, mock_open: MagicMock) -> None:
        """Test that _setup_fail_file handles an OSError."""
        writer, handle = _setup_fail_file("fail.csv", ["id"], ",", "utf-8")
        assert writer is None
        assert handle is None

    def test_create_batch_individually_malformed_row(self) -> None:
        """Test handling of malformed rows."""
        mock_model = MagicMock()
        batch_header = ["id", "name"]
        # This row has only one column, but the header has two
        batch_lines = [["record1"]]

        result = _create_batch_individually(
            mock_model, batch_lines, batch_header, 0, {}, []
        )

        assert len(result["failed_lines"]) == 1
        assert "Row has 1 columns, but header has 2" in result["failed_lines"][0][-1]
        assert result["error_summary"] == "Malformed CSV row detected"

    @patch(
        "odoo_data_flow.import_threaded.concurrent.futures.as_completed",
        side_effect=KeyboardInterrupt,
    )
    def test_run_threaded_pass_keyboard_interrupt(
        self, mock_as_completed: MagicMock
    ) -> None:
        """Test that a KeyboardInterrupt is handled gracefully."""
        from odoo_data_flow.import_threaded import RPCThreadImport, _run_threaded_pass

        rpc_thread = RPCThreadImport(1, Progress(), MagicMock())
        rpc_thread.task_id = rpc_thread.progress.add_task("test")
        target_func = MagicMock()
        target_func.__name__ = "mock_func"
        with patch.object(rpc_thread, "spawn_thread", return_value=MagicMock()):
            _, aborted = _run_threaded_pass(rpc_thread, target_func, [(1, {})], {})
            assert aborted is True

    @patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config",
        side_effect=Exception("Conn fail"),
    )
    def test_import_data_connection_failure(self, mock_get_conn: MagicMock) -> None:
        """Test that import_data handles a connection failure gracefully."""
        # Arrange
        with patch(
            "odoo_data_flow.import_threaded._read_data_file",
            return_value=(["id"], [["a"]]),
        ):
            # Act
            success, count = import_data("dummy.conf", "res.partner", "id", "dummy.csv")

            # Assert
            assert success is False
            assert count == {}

    @patch("odoo_data_flow.import_threaded._read_data_file", return_value=([], []))
    def test_import_data_no_header(self, mock_read_file: MagicMock) -> None:
        """Test that import_data handles a CSV with no header."""
        success, stats = import_data("dummy.conf", "res.partner", "id", "dummy.csv")
        assert success is False
        assert stats == {}

    @patch("odoo_data_flow.lib.internal.ui._show_error_panel")
    @patch(
        "odoo_data_flow.import_threaded.conf_lib.get_connection_from_config",
        side_effect=Exception("Conn fail"),
    )
    def test_import_data_connection_failure_shows_panel(
        self, mock_get_conn: MagicMock, mock_show_error: MagicMock
    ) -> None:
        """Test that import_data shows the error panel on connection failure."""
        # Arrange
        with patch(
            "odoo_data_flow.import_threaded._read_data_file",
            return_value=(["id"], [["a"]]),
        ):
            # Act
            import_data("dummy.conf", "res.partner", "id", "dummy.csv")

            # Assert
            mock_show_error.assert_called_once()
            call_args, _ = mock_show_error.call_args
            assert call_args[0] == "Odoo Connection Error"
            assert "Could not connect to Odoo" in call_args[1]

    def test_filter_ignored_columns(self) -> None:
        """Test that ignored columns are correctly filtered."""
        from odoo_data_flow.import_threaded import _filter_ignored_columns

        header = ["id", "name", "age", "city"]
        data = [
            ["1", "Alice", "30", "New York"],
            ["2", "Bob", "25", "London"],
        ]
        ignore = ["age", "city"]
        new_header, new_data = _filter_ignored_columns(ignore, header, data)
        assert new_header == ["id", "name"]
        assert new_data == [["1", "Alice"], ["2", "Bob"]]


class TestRecursiveBatching:
    """Tests for the recursive batch creation logic."""

    def test_recursive_batching_single_column(self) -> None:
        """Test recursive batching with a single grouping column."""
        from odoo_data_flow.import_threaded import _recursive_create_batches

        header = ["id", "name", "country"]
        data = [
            ["1", "A", "USA"],
            ["2", "B", "USA"],
            ["3", "C", "Canada"],
            ["4", "D", "USA"],
        ]
        batches = list(_recursive_create_batches(data, ["country"], header, 10, False))
        assert len(batches) == 2
        assert batches[0][1][0][2] == "Canada"
        assert batches[1][1][0][2] == "USA"

    def test_recursive_batching_multiple_columns(self) -> None:
        """Test recursive batching with multiple grouping columns."""
        from odoo_data_flow.import_threaded import _recursive_create_batches

        header = ["id", "name", "country", "state"]
        data = [
            ["1", "A", "USA", "CA"],
            ["2", "B", "USA", "NY"],
            ["3", "C", "Canada", "QC"],
            ["4", "D", "USA", "CA"],
        ]
        batches = list(
            _recursive_create_batches(data, ["country", "state"], header, 10, False)
        )
        assert len(batches) == 3
        # Note: The order of batches is not guaranteed, so we check the content
        # of each batch.
        batch_contents = [tuple(row) for _, batch_data in batches for row in batch_data]
        assert ("1", "A", "USA", "CA") in batch_contents
        assert ("4", "D", "USA", "CA") in batch_contents
        assert ("2", "B", "USA", "NY") in batch_contents
        assert ("3", "C", "Canada", "QC") in batch_contents

    def test_recursive_batching_group_col_not_found(self) -> None:
        """Test that an error is logged if a grouping column is not found."""
        from odoo_data_flow.import_threaded import _recursive_create_batches

        header = ["id", "name"]
        data = [["1", "A"]]
        with patch("odoo_data_flow.import_threaded.log") as mock_log:
            list(_recursive_create_batches(data, ["non_existent"], header, 10, False))
            mock_log.error.assert_called_once_with(
                "Grouping column 'non_existent' not found. Cannot use --groupby."
            )
