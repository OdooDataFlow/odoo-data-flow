"""Test the trasnform functions."""

from pathlib import Path
from typing import Any, Callable

import pytest

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import (
    MapperRepr,
    Processor,
    ProductProcessorV9,
)


def test_mapper_repr() -> None:
    """Tests the __repr__ method of the MapperRepr class."""
    mapper_repr = MapperRepr("mapper.val('test')", lambda: "value")
    assert repr(mapper_repr) == "mapper.val('test')"


def test_processor_init_fails_without_args() -> None:
    """Tests that the Processor raises a ValueError if initialized without args."""
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


def test_read_file_csv_not_found() -> None:
    """Tests that a non-existent CSV file is handled correctly."""
    processor = Processor(filename="non_existent_file.csv")
    assert processor.header == []
    assert processor.data == []


def test_join_file_missing_key(tmp_path: Path) -> None:
    """Tests that join_file handles a missing join key gracefully."""
    master_file = tmp_path / "master.csv"
    master_file.write_text("id,name\n1,master_record")
    child_file = tmp_path / "child.csv"
    child_file.write_text("child_id,value\n1,child_value")

    processor = Processor(filename=str(master_file), separator=",")
    original_header_len = len(processor.header)

    # Attempt to join on a key that doesn't exist in the master file
    processor.join_file(
        str(child_file),
        master_key="non_existent_key",
        child_key="child_id",
        separator=",",
    )

    # The header and data should remain unchanged because the join failed
    assert len(processor.header) == original_header_len


def test_process_with_legacy_mapper() -> None:
    """Tests that process works with a legacy mapper that only accepts one arg."""
    header = ["col1"]
    data = [["A"]]
    processor = Processor(header=header, data=data)

    # This lambda only accepts one argument, which would cause a TypeError
    # without the backward-compatibility logic in _process_mapping.
    legacy_mapping = {"new_col": lambda line: line["col1"].lower()}

    head, processed_data = processor.process(legacy_mapping, filename_out="")
    assert list(processed_data) == [["a"]]


def test_process_returns_set() -> None:
    """Tests that process correctly returns a set when t='set'."""
    header = ["col1"]
    # Include duplicate data
    data = [["A"], ["B"], ["A"]]
    processor = Processor(header=header, data=data)
    mapping = {"new_col": mapper.val("col1")}

    # Process with t='set' to get unique records
    head, processed_data = processor.process(mapping, filename_out="", t="set")

    assert isinstance(processed_data, set)
    # The set should only contain unique values
    assert len(processed_data) == 2
    assert ("A",) in processed_data
    assert ("B",) in processed_data


def test_v9_extract_attribute_value_data_malformed_mapping() -> None:
    """Tests that _extract_attribute_value_data handles a malformed mapping.

    This test ensures the `if not isinstance(values_dict, dict): continue`
    branch is covered.
    """
    processor = ProductProcessorV9(header=["col1"], data=[["val1"]])

    # Create a malformed mapping where the 'name' mapper returns a string,
    #  not a dict
    # The lambda is defined to accept an optional state to handle the fallback
    # logic.
    # Explicitly type the dictionary to satisfy mypy.
    malformed_mapping: dict[str, Callable[..., Any]] = {
        "name": mapper.val("col1"),
        "attribute_id/id": lambda line, state=None: "some_id",
    }

    # This should run without error and simply return an empty set
    result = processor._extract_attribute_value_data(
        malformed_mapping, ["Color"], [{"col1": "val1"}]
    )
    assert result == set()
