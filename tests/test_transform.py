"""Test the core Processor class and its subclasses."""

from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import (
    MapperRepr,
    Processor,
    ProductProcessorV9,
    ProductProcessorV10,
)


def test_mapper_repr_and_call() -> None:
    """Tests the __repr__ and __call__ methods of the MapperRepr class."""
    # Test __repr__
    mapper_repr = MapperRepr("mapper.val('test')", lambda x: x.upper())
    assert repr(mapper_repr) == "mapper.val('test')"
    # Test __call__
    assert mapper_repr("hello") == "HELLO"


def test_processor_init_fails_without_args() -> None:
    """Tests that the Processor raises a ValueError if initialized with no args."""
    with pytest.raises(
        ValueError, match="must be initialized with either a 'filename' or both"
    ):
        Processor()


def test_read_file_xml_syntax_error(tmp_path: Path) -> None:
    """Tests that a syntax error in an XML file is handled correctly."""
    xml_file = tmp_path / "malformed.xml"
    xml_file.write_text("<root><record>a</record></root")  # Malformed XML

    processor = Processor(filename=str(xml_file), xml_root_tag="./record")
    # Expect empty header and data due to the parsing error
    assert processor.header == []
    assert processor.data == []


@patch("odoo_data_flow.lib.transform.etree.parse")
def test_read_file_xml_generic_exception(mock_parse: MagicMock, tmp_path: Path) -> None:
    """Tests that a generic exception during XML parsing is handled."""
    mock_parse.side_effect = Exception("Generic XML read error")
    xml_file = tmp_path / "any.xml"
    xml_file.touch()

    processor = Processor(filename=str(xml_file), xml_root_tag="./record")
    assert processor.header == []
    assert processor.data == []


def test_read_file_csv_not_found() -> None:
    """Tests that a non-existent CSV file is handled correctly."""
    processor = Processor(filename="non_existent_file.csv")
    assert processor.header == []
    assert processor.data == []


@patch("odoo_data_flow.lib.transform.csv.reader")
def test_read_file_csv_generic_exception(
    mock_reader: MagicMock, tmp_path: Path
) -> None:
    """Tests that a generic exception during CSV reading is handled."""
    mock_reader.side_effect = Exception("Generic CSV read error")
    csv_file = tmp_path / "any.csv"
    csv_file.touch()

    processor = Processor(filename=str(csv_file))
    assert processor.header == []
    assert processor.data == []


@patch("odoo_data_flow.lib.transform.log.warning")
def test_check_failure(mock_log_warning: MagicMock) -> None:
    """Tests that the check method logs a warning when a check fails."""
    processor = Processor(header=[], data=[])

    def failing_check(h: list[str], d: list[list[Any]]) -> bool:
        return False

    result = processor.check(failing_check, message="Custom fail message")

    assert result is False
    mock_log_warning.assert_called_once()
    assert "Custom fail message" in mock_log_warning.call_args[0][0]


def test_join_file_success(tmp_path: Path) -> None:
    """Tests that join_file successfully merges data from two files."""
    master_file = tmp_path / "master.csv"
    master_file.write_text("id,name\n1,master_record")
    child_file = tmp_path / "child.csv"
    child_file.write_text("child_id,value\n1,child_value")

    processor = Processor(filename=str(master_file), separator=",")
    processor.join_file(
        str(child_file), master_key="id", child_key="child_id", separator=","
    )

    assert processor.header == ["id", "name", "child_child_id", "child_value"]
    assert processor.data == [["1", "master_record", "1", "child_value"]]


def test_join_file_missing_key(tmp_path: Path) -> None:
    """Tests that join_file handles a missing join key gracefully."""
    master_file = tmp_path / "master.csv"
    master_file.write_text("id,name\n1,master_record")
    child_file = tmp_path / "child.csv"
    child_file.write_text("child_id,value\n1,child_value")

    processor = Processor(filename=str(master_file), separator=",")
    original_header_len = len(processor.header)

    with patch("odoo_data_flow.lib.transform.log.error") as mock_log_error:
        processor.join_file(
            str(child_file),
            master_key="non_existent_key",
            child_key="child_id",
            separator=",",
        )
        mock_log_error.assert_called_once()
        assert "Join key error" in mock_log_error.call_args[0][0]

    # The header and data should remain unchanged because the join failed
    assert len(processor.header) == original_header_len


@patch("odoo_data_flow.lib.transform.Console")
@patch("odoo_data_flow.lib.transform.Processor._read_file")
def test_join_file_dry_run(
    mock_read_file: MagicMock, mock_console_class: MagicMock
) -> None:
    """Tests that join_file in dry_run mode creates a table and does not modify data."""
    # 1. Setup
    # Initialize a processor with some master data in memory
    processor = Processor(header=["id", "name"], data=[["1", "master_record"]])
    # original_header = list(processor.header)
    original_data = [list(row) for row in processor.data]

    # Configure the mock _read_file to return our child data
    mock_read_file.return_value = (["child_id", "value"], [["1", "child_value"]])
    # mock_console_instance = mock_console_class.return_value

    # 2. Action
    processor.join_file(
        "any_child_file.csv",
        master_key="id",
        child_key="child_id",
        dry_run=True,
    )

    # 3. Assertions
    # Assert that the original processor data was NOT modified
    # assert processor.header == original_header
    assert processor.data == original_data


def test_process_with_legacy_mapper() -> None:
    """Tests that process works with a legacy mapper that only accepts one arg."""
    header = ["col1"]
    data = [["A"]]
    processor = Processor(header=header, data=data)

    # This lambda only accepts one argument, which would cause a TypeError
    # without the backward-compatibility logic in _process_mapping.
    legacy_mapping = {"new_col": lambda line: line["col1"].lower()}
    _head, processed_data = processor.process(legacy_mapping, filename_out="")
    assert list(processed_data) == [["a"]]


def test_v9_extract_attribute_value_data_malformed_mapping() -> None:
    """Tests that _extract_attribute_value_data handles a malformed mapping.

    This test ensures the `if not isinstance(values_dict, dict): continue`
    branch is covered.
    """
    processor = ProductProcessorV9(header=["col1"], data=[["val1"]])

    malformed_mapping: dict[str, Callable[..., Any]] = {
        "name": mapper.val("col1"),
        "attribute_id/id": lambda line, state=None: "some_id",
    }

    result = processor._extract_attribute_value_data(
        malformed_mapping, ["Color"], [{"col1": "val1"}]
    )
    assert result == set()


def test_process_returns_set() -> None:
    """Tests that process correctly returns a set when t='set'."""
    processor = Processor(header=["col1"], data=[["A"], ["B"], ["A"]])
    _head, processed_data = processor.process(
        {"new_col": mapper.val("col1")}, filename_out="", t="set"
    )
    assert isinstance(processed_data, set)
    assert len(processed_data) == 2
    assert ("A",) in processed_data
    assert ("B",) in processed_data


@patch("odoo_data_flow.lib.transform.Console")
def test_process_dry_run(mock_console_class: MagicMock) -> None:
    """Tests that dry_run mode prints a table and does not write files."""
    processor = Processor(header=["col1"], data=[["A"]])
    mapping = {"new_col": mapper.val("col1")}
    mock_console_instance = mock_console_class.return_value
    processor.process(mapping, "file.csv", dry_run=True)

    # Assert that no file was added to the write queue
    assert not processor.file_to_write
    mock_console_instance.print.assert_called_once()


@patch("odoo_data_flow.lib.transform.write_file")
def test_write_to_file_append_and_no_fail(mock_write_file: MagicMock) -> None:
    """Tests write_to_file with append=True and fail=False."""
    processor = Processor(header=["id"], data=[["1"]])
    processor.process({"id": mapper.val("id")}, "file1.csv", params={"model": "model1"})
    processor.process({"id": mapper.val("id")}, "file2.csv", params={"model": "model2"})

    processor.write_to_file("script.sh", fail=False, append=True)

    assert mock_write_file.call_count == 2
    assert mock_write_file.call_args_list[0].kwargs["init"] is False
    assert mock_write_file.call_args_list[1].kwargs["init"] is False


def test_v10_process_attribute_value_data() -> None:
    """Tests the attribute value data processing for the V10+ workflow."""
    header = ["Color", "Size"]
    data = [["Blue", "L"], ["Red", "L"], ["Blue", "M"]]
    processor = ProductProcessorV10(header=header, data=data)

    processor.process_attribute_value_data(
        attribute_list=["Color", "Size"],
        attribute_value_prefix="val_prefix",
        attribute_prefix="attr_prefix",
        filename_out="product.attribute.value.csv",
        import_args={},
    )

    assert "product.attribute.value.csv" in processor.file_to_write
    result = processor.file_to_write["product.attribute.value.csv"]
    assert result["header"] == ["id", "name", "attribute_id/id"]

    # We expect 4 unique values: Blue, L, Red, M
    assert len(result["data"]) == 4
    # Check for one of the generated rows to ensure correctness
    expected_row = ["val_prefix.Color_Blue", "Blue", "attr_prefix.Color"]
    assert any(row == expected_row for row in result["data"])


def test_v9_extract_attribute_value_data_legacy_mapper() -> None:
    """Tests that _extract_attribute_value_data handles legacy mappers."""
    processor = ProductProcessorV9(header=["col1"], data=[["val1"]])

    # This mapping uses a legacy 1-argument lambda
    legacy_mapping: dict[str, Callable[..., Any]] = {
        "name": lambda line: {"Color": line["col1"]},
        "attribute_id/id": lambda line: "some_id",
    }

    result = processor._extract_attribute_value_data(
        legacy_mapping, ["Color"], [{"col1": "val1"}]
    )
    # The result should contain a tuple with the resolved value 'val1'
    assert ("val1", "some_id") in result


def test_v9_process_attribute_mapping_with_custom_id_gen(tmp_path: Path) -> None:
    """Tests the full process_attribute_mapping method from ProductProcessorV9.

    This test uses a custom ID generation function.
    """
    header = ["template_id", "Color", "Size"]
    data = [
        ["TPL1", "Blue", "L"],
        ["TPL2", "Red", "M"],
        ["TPL1", "Green", "L"],
    ]
    processor = ProductProcessorV9(header=header, data=data)
    attributes = ["Color", "Size"]
    prefix = "test_prefix"
    output_path = str(tmp_path) + "/"

    value_mapping = {
        "id": mapper.m2m_attribute_value(prefix, *attributes),
        "name": mapper.val_att(attributes),
        "attribute_id/id": mapper.m2o_att_name(prefix, attributes),
    }
    line_mapping = {
        "product_tmpl_id/id": mapper.m2o_map("tmpl_", "template_id"),
        "attribute_id/id": mapper.m2o_att_name(prefix, attributes),
        "value_ids/id": mapper.m2o_att(prefix, attributes),
    }

    def custom_id_gen(tmpl_id: str, vals: dict[str, Any]) -> str:
        return f"custom_line_id_for_{tmpl_id}"

    processor.process_attribute_mapping(
        value_mapping,
        line_mapping,
        attributes,
        prefix,
        output_path,
        {},
        id_gen_fun=custom_id_gen,
    )

    # Assert that all three files were added to the write queue
    assert len(processor.file_to_write) == 3
    line_file_data = processor.file_to_write[output_path + "product.attribute.line.csv"]
    assert line_file_data["data"][0][0] == "custom_line_id_for_tmpl_.TPL1"
