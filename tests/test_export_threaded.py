"""Test the Export Handling mechanism."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from odoo_data_flow.export_threaded import (
    RPCThreadExport,
    _clean_batch,
    _initialize_export,
    _process_export_batches,
    export_data,
)


@pytest.fixture
def mock_conf_lib() -> Generator[MagicMock, None, None]:
    """Fixture to mock conf_lib.get_connection_from_config."""
    with patch(
        "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config"
    ) as mock_conn:
        mock_model_obj = MagicMock()
        mock_model_obj.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "is_company": {"type": "boolean"},
            "phone": {"type": "char"},
        }
        mock_connection = MagicMock()
        mock_connection.get_model.return_value = mock_model_obj
        mock_conn.return_value = mock_connection
        yield mock_conn


class TestRPCThreadExport:
    """Tests for the RPCThreadExport class."""

    def test_execute_batch_technical_names(self) -> None:
        """Tests that model.read is called when technical_names is True."""
        mock_model = MagicMock()
        thread = RPCThreadExport(1, mock_model, ["id"], technical_names=True)
        thread._execute_batch([1], 1)
        mock_model.read.assert_called_once_with([1], ["id"])
        mock_model.export_data.assert_not_called()

    def test_execute_batch_export_data(self) -> None:
        """Tests that model.export_data is called when technical_names is False."""
        mock_model = MagicMock()
        mock_model.export_data.return_value = {"datas": [["Test"]]}
        thread = RPCThreadExport(1, mock_model, ["name"], technical_names=False)
        result = thread._execute_batch([1], 1)
        mock_model.export_data.assert_called_once()
        mock_model.read.assert_not_called()
        assert result == [{"name": "Test"}]

    def test_rpc_thread_export_memory_error(self) -> None:
        """Test for memory errors.

        Test that the RPCThreadExport class handles MemoryError and subsequent failures.
        """
        mock_model = MagicMock()
        mock_model.read.side_effect = [
            Exception({"data": {"name": "builtins.MemoryError"}}),
            [{"id": 1}],
            Exception("A permanent error"),
        ]
        thread = RPCThreadExport(1, mock_model, ["id"], technical_names=True)
        result = thread._execute_batch([1, 2], 1)
        assert result == [{"id": 1}]
        assert mock_model.read.call_count == 3


class TestCleanBatch:
    """Tests for the _clean_batch helper function."""

    def test_clean_batch_converts_false_correctly(self) -> None:
        """Tests that False is converted to None for non-booleans."""
        dirty_data = [
            {"id": 1, "name": "Test", "is_company": True, "phone": False},
            {"id": 2, "name": False, "is_company": False, "phone": "12345"},
        ]
        field_types = {
            "id": "integer",
            "name": "char",
            "is_company": "boolean",
            "phone": "char",
        }
        cleaned_df = _clean_batch(dirty_data, field_types)
        expected_df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["Test", None],
                "is_company": [True, False],
                "phone": [None, "12345"],
            }
        )
        assert_frame_equal(cleaned_df, expected_df)

    def test_clean_batch_empty_input(self) -> None:
        """Tests that an empty list is handled correctly."""
        assert _clean_batch([], {}).is_empty()

    def test_clean_batch_with_boolean(self) -> None:
        """Test that _clean_batch handles boolean values correctly."""
        data = [{"id": 1, "active": True}, {"id": 2, "active": False}]
        field_types = {"id": "integer", "active": "boolean"}
        df = _clean_batch(data, field_types)
        assert df.to_dicts() == data


class TestExportData:
    """Tests for the main export_data orchestrator function."""

    def test_export_in_memory_success(self, mock_conf_lib: MagicMock) -> None:
        """Tests the success path for a default in-memory export."""
        # --- Arrange ---
        header = ["id", "name", "active"]
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2]

        # Ensure read() returns ONLY the columns in the header
        mock_model.read.return_value = [
            {"id": 1, "name": "Test 1", "active": True},
            {"id": 2, "name": "Test 2", "active": False},
        ]
        # Ensure fields_get() returns ONLY the types for the header
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
            "active": {"type": "boolean"},
        }

        # --- Act ---
        result_df = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=header,
            output=None,
            technical_names=True,
        )

        # --- Assert ---
        assert result_df is not None
        expected_df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["Test 1", "Test 2"],
                "active": [True, False],
            }
        ).with_columns(pl.col("id").cast(pl.Int64))

        assert_frame_equal(result_df, expected_df)

    def test_export_to_file_default_mode_success(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests a file export using the default in-memory concat mode."""
        # --- Arrange ---
        header = ["id", "name"]
        output_file = tmp_path / "output.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2]
        mock_model.read.return_value = [
            {"id": 1, "name": "Test 1"},
            {"id": 2, "name": "Test 2"},
        ]
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        # --- Act ---
        result_df = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=header,
            output=str(output_file),
            technical_names=True,
            streaming=False,
        )

        # --- Assert ---
        assert output_file.exists()
        assert result_df is not None

        expected_df = pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["Test 1", "Test 2"],
            }
        ).with_columns(pl.col("id").cast(pl.Int64))

        # Add separator and sort both frames
        on_disk_df = pl.read_csv(output_file, separator=";")
        assert_frame_equal(result_df.sort("id"), expected_df.sort("id"))
        assert_frame_equal(on_disk_df.sort("id"), expected_df.sort("id"))

    def test_export_to_file_streaming_success(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests the success path where data is streamed to a CSV file."""
        # --- Arrange ---
        header = ["id", "name"]
        output_file = tmp_path / "output.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2]
        mock_model.read.side_effect = [
            [{"id": 2, "name": "Test 2"}],  # Simulate out-of-order completion
            [{"id": 1, "name": "Test 1"}],
        ]
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        # --- Act ---
        result_df = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=header,
            output=str(output_file),
            technical_names=True,
            streaming=True,
            batch_size=1,
        )

        # --- Assert ---
        assert result_df is None
        assert output_file.exists()

        # Add separator and sort both frames
        on_disk_df = pl.read_csv(output_file, separator=";")
        expected_df = pl.DataFrame({"id": [1, 2], "name": ["Test 1", "Test 2"]})
        assert_frame_equal(on_disk_df.sort("id"), expected_df.sort("id"))

    def test_export_handles_connection_failure(self) -> None:
        """Tests that None is returned if the initial connection fails."""
        with patch(
            "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
            side_effect=Exception("Connection Error"),
        ):
            result = export_data(
                config_file="bad.conf",
                model="res.partner",
                domain=[],
                header=["id"],
                output="fail.csv",
            )
        assert result is None

    def test_export_handles_no_records_found(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests behavior when no records match the domain."""
        output_file = tmp_path / "empty.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = []
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        result_df = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
            output=str(output_file),
            separator=",",
        )

        assert output_file.exists()
        assert isinstance(result_df, pl.DataFrame)
        assert result_df.is_empty()

        with open(output_file) as f:
            assert f.read().strip() == "id,name"

    def test_export_handles_memory_error_fallback(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that the batch is split and retried on server MemoryError."""
        # --- Arrange ---
        output_file = tmp_path / "output.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2, 3, 4]

        # Simulate Odoo failing with MemoryError on the first large batch,
        # then succeeding on the two smaller retry batches.
        memory_error_response = Exception(
            {
                "code": 200,
                "message": "Odoo Server Error",
                "data": {"name": "builtins.MemoryError", "debug": "..."},
            }
        )
        mock_model.read.side_effect = [
            memory_error_response,
            [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],  # 1st retry
            [{"id": 3, "name": "C"}, {"id": 4, "name": "D"}],  # 2nd retry
        ]
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        # --- Act ---
        result_df = export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
            output=str(output_file),
            technical_names=True,
            streaming=True,
            batch_size=4,
        )

        # --- Assert ---
        assert result_df is None
        assert output_file.exists()
        assert mock_model.read.call_count == 3  # 1 failure + 2 retries

        # Verify the final file has all data from the successful retries
        on_disk_df = pl.read_csv(output_file, separator=";")
        expected_df = pl.DataFrame({"id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
        assert_frame_equal(on_disk_df.sort("id"), expected_df.sort("id"))

    def test_export_handles_empty_batch_result(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that an empty result from a batch is handled gracefully."""
        # --- Arrange ---
        output_file = tmp_path / "output.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2]
        # Simulate one batch succeeding and one returning no data
        mock_model.read.side_effect = [[{"id": 1, "name": "A"}], []]
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        # --- Act ---
        export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
            output=str(output_file),
            technical_names=True,
            batch_size=1,
        )

        # --- Assert ---
        # The file should contain only the data from the successful batch
        on_disk_df = pl.read_csv(output_file, separator=";")
        assert len(on_disk_df) == 1
        assert on_disk_df["id"][0] == 1

    def test_export_handles_permanent_worker_failure(
        self, mock_conf_lib: MagicMock, tmp_path: Path
    ) -> None:
        """Tests that a non-MemoryError exception in a worker is survivable."""
        # --- Arrange ---
        output_file = tmp_path / "output.csv"
        mock_model = mock_conf_lib.return_value.get_model.return_value
        mock_model.search.return_value = [1, 2]
        # Simulate one batch succeeding and one failing with a different error
        mock_model.read.side_effect = [
            [{"id": 1, "name": "A"}],
            ValueError("A permanent error"),
        ]
        mock_model.fields_get.return_value = {
            "id": {"type": "integer"},
            "name": {"type": "char"},
        }

        # --- Act ---
        export_data(
            config_file="dummy.conf",
            model="res.partner",
            domain=[],
            header=["id", "name"],
            output=str(output_file),
            technical_names=True,
            batch_size=1,
        )

        # --- Assert ---
        # The export should complete with data from the successful batch
        assert output_file.exists()
        on_disk_df = pl.read_csv(output_file, separator=";")
        assert len(on_disk_df) == 1

    def test_initialize_export_connection_error(self) -> None:
        """Test that _initialize_export handles connection errors."""
        with patch(
            "odoo_data_flow.export_threaded.conf_lib.get_connection_from_config",
            side_effect=Exception("Conn error"),
        ):
            model, field_types = _initialize_export("dummy.conf", "res.partner", ["id"])
            assert model is None
            assert field_types is None

    @patch("concurrent.futures.as_completed")
    def test_process_export_batches_task_failure(
        self, mock_as_completed: MagicMock
    ) -> None:
        """Test that _process_export_batches handles a failing future."""
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Task failed")
        mock_as_completed.return_value = [mock_future]
        mock_rpc_thread = MagicMock()
        mock_rpc_thread.futures = [mock_future]

        result = _process_export_batches(
            mock_rpc_thread, 1, "res.partner", None, {}, ";", False
        )
        if result is not None:
            assert result.is_empty()

    @patch("concurrent.futures.as_completed")
    def test_process_export_batches_empty_result(
        self, mock_as_completed: MagicMock
    ) -> None:
        """Test that _process_export_batches handles an empty result from a future."""
        mock_future = MagicMock()
        mock_future.result.return_value = []
        mock_as_completed.return_value = [mock_future]
        mock_rpc_thread = MagicMock()
        mock_rpc_thread.futures = [mock_future]

        result = _process_export_batches(
            mock_rpc_thread, 1, "res.partner", None, {}, ";", False
        )
        if result is not None:
            assert result.is_empty()

    def test_process_export_batches_no_dfs_with_output(self) -> None:
        """Test _process_export_batches with no dataframes and an output file."""
        mock_rpc_thread = MagicMock()
        mock_rpc_thread.futures = []
        with patch("polars.DataFrame.write_csv") as mock_write_csv:
            result = _process_export_batches(
                mock_rpc_thread,
                0,
                "res.partner",
                "out.csv",
                {"id": "integer"},
                ";",
                False,
            )
            if result is not None:
                assert result.is_empty()
            mock_write_csv.assert_called_once_with("out.csv", separator=";")

    def test_export_data_streaming_no_output(self) -> None:
        """Test that export_data handles streaming with no output."""
        with patch(
            "odoo_data_flow.export_threaded._initialize_export",
            return_value=(MagicMock(), {"id": "integer"}),
        ):
            result = export_data(
                "dummy.conf", "res.partner", [], ["id"], None, streaming=True
            )
            assert result is None


@patch("odoo_data_flow.export_threaded.concurrent.futures.as_completed")
@patch("odoo_data_flow.export_threaded._clean_batch")
@patch("odoo_data_flow.export_threaded.Progress")
def test_process_export_batches_handles_inconsistent_schemas(
    mock_progress: MagicMock,
    mock_clean_batch: MagicMock,
    mock_as_completed: MagicMock,
) -> None:
    """Tests that batches with inconsistent schemas are correctly concatenated.

    This validates the fix for the `SchemaError` that occurred when one batch
    inferred a column as Boolean and another inferred it as String. The test
    ensures that the final concatenated DataFrame has a consistent, correct schema.
    """
    # --- Arrange ---
    # 1. Mock the RPC thread and its futures to simulate two batches
    mock_rpc_thread = MagicMock(spec=RPCThreadExport)
    future1, future2 = MagicMock(), MagicMock()
    future1.result.return_value = [{"id": 1, "is_special": True}]
    future2.result.return_value = [{"id": 2, "is_special": "False"}]

    # FIX: Assign the 'futures' attribute to the mock object
    mock_rpc_thread.futures = [future1, future2]

    # Mock the return value of as_completed to prevent hanging
    mock_as_completed.return_value = [future1, future2]

    # Explicitly create the nested executor.shutdown attribute on the mock
    mock_rpc_thread.executor = MagicMock()
    mock_rpc_thread.executor.shutdown.return_value = None

    # 2. Mock `_clean_batch` to return DataFrames with inconsistent dtypes,
    #    simulating the real-world failure case.
    mock_clean_batch.side_effect = [
        pl.DataFrame(
            {"id": [1], "is_special": [True]}
        ),  # Polars infers this as Boolean
        pl.DataFrame(
            {"id": [2], "is_special": ["False"]}
        ),  # Polars infers this as String
    ]

    # 3. Define the field types and total records
    field_types = {"id": "integer", "is_special": "boolean"}
    total_ids = 2

    # --- Act ---
    # Run the function in in-memory mode (output=None) to get the DataFrame back
    final_df = _process_export_batches(
        rpc_thread=mock_rpc_thread,
        total_ids=total_ids,
        model_name="test.model",
        output=None,
        field_types=field_types,
        separator=",",
        streaming=False,
    )

    assert final_df is not None
    # --- Assert ---
    # The final DataFrame should have a consistent schema and correct data
    expected_schema = {
        "id": pl.Int64(),
        "is_special": pl.Boolean(),
    }
    expected_df = pl.DataFrame(
        {"id": [1, 2], "is_special": [True, False]},
        schema=expected_schema,
    )

    # Sort by 'id' to ensure a stable order for comparison
    final_df = final_df.sort("id")

    assert_frame_equal(final_df, expected_df)
