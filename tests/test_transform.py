"""Test the core Processor class and its subclasses."""

import inspect
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.transform import (
    MapperRepr,
    Processor,
    ProductProcessorV9,
    ProductProcessorV10,
)


def test_mapper_repr_and_call() -> None:
    """Tests the __repr__ and __call__ methods of the MapperRepr class."""
    mapper_repr = MapperRepr("mapper.val('test')", lambda x, _: x.upper())
    assert repr(mapper_repr) == "mapper.val('test')"
    # Mappers are now called with (row, state), so we pass two args
    assert mapper_repr("hello", {}) == "HELLO"


def test_processor_init_fails_without_source() -> None:
    """Tests that the Processor raises a ValueError if initialized without a source."""
    with pytest.raises(
        ValueError,
        match="must be initialized with either a 'source_filename' or a 'dataframe'",
    ):
        Processor(mapping={})


@patch("odoo_data_flow.lib.transform.build_polars_schema")
@patch("odoo_data_flow.lib.transform.pl.read_csv")
def test_processor_init_with_connection_and_model(
    mock_read_csv: MagicMock, mock_build_schema: MagicMock
) -> None:
    """Tests that __init__ calls the schema builder when a connection is passed."""
    mock_conn = MagicMock()
    mock_build_schema.return_value = {"name": pl.String}
    Processor(
        mapping={},
        source_filename="file.csv",
        connection=mock_conn,
        model="res.partner",
    )
    mock_build_schema.assert_called_once_with(mock_conn, "res.partner")
    assert mock_read_csv.call_args.kwargs["schema_overrides"] == {"name": pl.String}


def test_processor_init_with_typed_mapping() -> None:
    """Tests that __init__ correctly parses a typed mapping."""
    typed_mapping = {
        "active": (pl.Boolean, mapper.val("active")),
        "city": mapper.val("city"),
    }
    processor = Processor(mapping=typed_mapping, dataframe=pl.DataFrame())
    assert processor.schema_overrides == {"active": pl.Boolean}
    assert "active" in processor.logic_mapping
    assert "city" in processor.logic_mapping
    assert callable(processor.logic_mapping["active"])


def test_get_o2o_mapping() -> None:
    """Tests the one-to-one mapping generation."""
    df = pl.DataFrame({"id": [1], "name": ["Test"], "city": ["Drunen"]})
    processor = Processor(mapping={}, dataframe=df)
    o2o_map = processor.get_o2o_mapping()
    assert list(o2o_map.keys()) == ["id", "name", "city"]
    assert o2o_map["name"]({"name": "Test Name"}, {}) == "Test Name"


def test_split() -> None:
    """Tests splitting a Processor into multiple based on a key."""
    df = pl.DataFrame(
        {"country": ["NL", "BE", "NL"], "name": ["Alice", "Bob", "Charlie"]}
    )
    processor = Processor(mapping={"name": mapper.val("name")}, dataframe=df)

    def split_by_country(row: dict[str, Any], _: Any) -> str:
        return str(row["country"])

    split_processors = processor.split(split_by_country)

    assert isinstance(split_processors, dict)
    assert set(split_processors.keys()) == {"NL", "BE"}
    assert isinstance(split_processors["NL"], Processor)
    assert len(split_processors["NL"].dataframe) == 2
    assert len(split_processors["BE"].dataframe) == 1
    assert "Alice" in split_processors["NL"].dataframe["name"].to_list()
    assert "Bob" in split_processors["BE"].dataframe["name"].to_list()


def test_process_with_polars_expression() -> None:
    """Tests that the process method can handle a direct Polars expression."""
    df = pl.DataFrame({"value": [10, 20]})
    mapping = {"doubled": pl.col("value") * 2}
    processor = Processor(mapping=mapping, dataframe=df)
    result = processor.process(filename_out="")
    assert result["doubled"].to_list() == [20, 40]


def test_process_m2m() -> None:
    """Tests that process(m2m=True) unnivots data correctly."""
    # 1. Setup: Create a "wide" DataFrame, similar to product attributes.
    wide_df = pl.DataFrame(
        {
            "ref": ["P1", "P2"],
            "Color": ["Blue", "Red"],
            "Size": ["L", "M"],
        }
    )

    # Define a simple mapping that will operate on the "long" data
    # after the internal unpivot is done.
    simple_mapping = {
        "product_ref": mapper.val("ref"),
        "attribute": mapper.val("m2m_source_column"),
        "value": mapper.val("m2m_source_value"),
    }

    processor = Processor(mapping=simple_mapping, dataframe=wide_df)

    # 2. Action: Call process with m2m=True and specify the columns to unpivot.
    result_df = processor.process(
        filename_out="",
        m2m=True,
        m2m_columns=["Color", "Size"],
        dry_run=True,  # Use dry_run to get the DataFrame back without writing files
    )

    # 3. Assertions
    # The result should have 4 rows (2 products * 2 attributes)
    assert result_df.shape == (4, 3)
    assert list(result_df.columns) == ["product_ref", "attribute", "value"]

    # Check the contents for correctness
    expected_data = [
        ("P1", "Color", "Blue"),
        ("P1", "Size", "L"),
        ("P2", "Color", "Red"),
        ("P2", "Size", "M"),
    ]
    # Convert result to a set of tuples for easy, order-independent comparison
    actual_data = {tuple(row) for row in result_df.iter_rows()}
    assert actual_data == set(expected_data)


def test_read_file_xml_syntax_error(tmp_path: Path) -> None:
    """Tests that a syntax error in an XML file is handled correctly."""
    xml_file = tmp_path / "malformed.xml"
    xml_file.write_text("<root><record>a</record></root")  # Malformed XML
    processor = Processor(
        mapping={},
        source_filename=str(xml_file),
        xml_root_tag="./record",
    )
    assert processor.dataframe.is_empty()


@patch("odoo_data_flow.lib.transform.pl.read_csv")
def test_read_file_csv_generic_exception(
    mock_read_csv: MagicMock, tmp_path: Path
) -> None:
    """Tests that a generic exception during CSV reading is handled."""
    mock_read_csv.side_effect = Exception("Generic CSV read error")
    csv_file = tmp_path / "any.csv"
    csv_file.touch()
    processor = Processor(mapping={}, source_filename=str(csv_file))
    assert processor.dataframe.is_empty()


@patch("odoo_data_flow.lib.transform.log.warning")
def test_check_failure(mock_log_warning: MagicMock) -> None:
    """Tests that the check method logs a warning when a check fails."""
    processor = Processor(mapping={}, dataframe=pl.DataFrame())
    result = processor.check(lambda df: False, message="Custom fail message")
    assert result is False
    mock_log_warning.assert_called_once()


def test_join_file_missing_key(tmp_path: Path) -> None:
    """Tests that join_file handles a missing join key gracefully."""
    master_file = tmp_path / "master.csv"
    master_file.write_text("id,name\n1,master_record")
    child_file = tmp_path / "child.csv"
    child_file.write_text("child_id,value\n1,child_value")
    processor = Processor(mapping={}, source_filename=str(master_file), separator=",")
    with pytest.raises(ColumnNotFoundError):
        processor.join_file(
            str(child_file),
            master_key="non_existent_key",
            child_key="child_id",
            separator=",",
        )


@patch("odoo_data_flow.lib.transform.Console")
def test_join_file_dry_run(mock_console_class: MagicMock, tmp_path: Path) -> None:
    """Tests that join_file in dry_run mode does not modify data."""
    master_df = pl.DataFrame({"id": [1], "name": ["master_record"]})
    processor = Processor(mapping={}, dataframe=master_df)
    original_df = processor.dataframe.clone()
    child_file = tmp_path / "child.csv"
    child_file.write_text("child_id,value\n1,child_value")
    processor.join_file(
        str(child_file),
        master_key="id",
        child_key="child_id",
        separator=",",
        dry_run=True,
    )
    assert str(original_df) == str(processor.dataframe)
    mock_console_class.return_value.print.assert_called_once()


@patch("odoo_data_flow.lib.transform.Console")
def test_process_dry_run(mock_console_class: MagicMock) -> None:
    """Tests that dry_run mode prints a table and does not write files."""
    df = pl.DataFrame({"col1": ["A"]})
    mapping = {"new_col": mapper.val("col1")}
    processor = Processor(mapping=mapping, dataframe=df)
    mock_console_instance = mock_console_class.return_value
    processor.process(filename_out="file.csv", dry_run=True)
    assert not processor.file_to_write
    mock_console_instance.print.assert_called_once()


def test_v9_extract_attribute_value_data_malformed_mapping() -> None:
    """Tests that _extract_attribute_value_data handles a malformed mapping."""
    df = pl.DataFrame([{"col1": "val1"}])
    processor = ProductProcessorV9(mapping={}, dataframe=df)
    malformed_mapping: dict[str, Callable[..., Any]] = {
        "name": mapper.val("col1"),
        "attribute_id/id": lambda line, state=None: "some_id",
    }
    result = processor._extract_attribute_value_data(malformed_mapping, ["col1"])
    assert not result.is_empty()


def test_processor_with_callable_in_concat() -> None:
    """Tests that Processor handles mappers with callable arguments."""
    df = pl.DataFrame({"col1": ["A"], "col2": ["B"]})

    def custom_mapper(line: dict[str, Any], state: dict[str, Any]) -> str:
        return str(line["col2"])

    mapping = {"concatenated": mapper.concat("-", "col1", custom_mapper)}
    processor = Processor(mapping=mapping, dataframe=df)
    result_df = processor.process(filename_out="")
    assert result_df["concatenated"].to_list() == ["A-B"]


def test_processor_val_postprocess_type_error_fallback() -> None:
    """Tests that Processor.val handles TypeError in postprocess."""
    df = pl.DataFrame({"value": ["test"]})

    # This postprocess function will fail if called with two arguments (the state dict)
    # but will succeed if called with just one (the value).
    def fallback_postprocess(val: str, state: Optional[dict[str, Any]] = None) -> str:
        if state is not None:
            raise TypeError(
                "This function should have been called with only one argument"
            )
        return val.upper()

    # Patch inspect.signature to simulate a function that *looks* like it takes 2 args
    # but actually raises TypeError when called with 2 args. This forces the fallback.
    with patch("inspect.signature") as mock_signature:
        mock_signature.return_value.parameters = {
            "arg1": MagicMock(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
            "arg2": MagicMock(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
        }
        # Assign the fallback_postprocess to the mapper. The code should initially
        # try to call it with two arguments, hit the TypeError, and then
        # successfully call it with one argument.
        mapping = {"processed": mapper.val("value", postprocess=fallback_postprocess)}
        processor = Processor(mapping=mapping, dataframe=df)
        result_df = processor.process(filename_out="")

        # The actual result should come from the successful 1-arg fallback call.
        assert result_df["processed"].to_list() == ["TEST"]


def test_read_file_unsupported_type(tmp_path: Path) -> None:
    """Tests that reading an unsupported file type returns an empty DataFrame."""
    unsupported_file = tmp_path / "data.unsupported"
    unsupported_file.touch()
    processor = Processor(mapping={}, source_filename=str(unsupported_file))
    assert processor.dataframe.is_empty()


def test_read_file_xml_no_nodes(tmp_path: Path) -> None:
    """Test malformed xml file.

    Tests that reading an XML file with no matching nodes returns
    an empty DataFrame.
    """
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("<root><item>a</item></root>")
    processor = Processor(
        mapping={},
        source_filename=str(xml_file),
        xml_root_tag="./nonexistent",
    )
    assert processor.dataframe.is_empty()


def test_process_with_empty_mapping() -> None:
    """Tests that processing with an empty mapping returns the original DataFrame."""
    df = pl.DataFrame({"col1": [1, 2]})
    processor = Processor(mapping={}, dataframe=df)
    result_df = processor.process(filename_out="")
    assert result_df.equals(df)


def test_process_m2m_no_columns() -> None:
    """Tests that calling process with m2m=True and no m2m_columns raises ValueError."""
    processor = Processor(mapping={}, dataframe=pl.DataFrame())
    with pytest.raises(ValueError):
        processor.process(filename_out="", m2m=True)


def test_process_m2m_method() -> None:
    """Tests the dedicated process_m2m method."""
    df = pl.DataFrame({"id": [1], "tags": ["a,b,c"]})
    mapping = {"id": mapper.val("id"), "tag": mapper.val("tags")}
    processor = Processor(mapping=mapping, dataframe=df)
    processor.process_m2m(id_column="id", m2m_columns=["tags"], filename_out="out.csv")
    assert "out.csv" in processor.file_to_write
    result_df = processor.file_to_write["out.csv"]["dataframe"]
    assert result_df.shape == (3, 2)
    assert result_df["tag"].to_list() == ["a", "b", "c"]


def test_product_processor_v10() -> None:
    """Tests the ProductProcessorV10 methods."""
    df = pl.DataFrame({"Color": ["Blue"], "Size": ["L"]})
    processor = ProductProcessorV10(mapping={}, dataframe=df)
    processor.process_attribute_data(
        attributes_list=["Color", "Size"],
        attribute_prefix="pa",
        filename_out="attrib.csv",
        import_args={},
    )
    processor.process_attribute_value_data(
        attribute_list=["Color", "Size"],
        attribute_prefix="pa",
        attribute_value_prefix="pav",
        filename_out="values.csv",
        import_args={},
    )
    assert "attrib.csv" in processor.file_to_write
    assert "values.csv" in processor.file_to_write


def test_product_processor_v9() -> None:
    """Tests the ProductProcessorV9 methods."""
    df = pl.DataFrame({"Color": ["Blue"], "Size": ["L"]})
    mapping = {"name": mapper.val("attribute_value_name")}
    line_mapping = {"product_id": mapper.val("id")}
    processor = ProductProcessorV9(mapping=mapping, dataframe=df)
    processor.process_attribute_mapping(
        mapping=mapping,
        line_mapping=line_mapping,
        attributes_list=["Color", "Size"],
        attribute_prefix="pa",
        path="",
        import_args={},
    )
    assert "product.attribute.csv" in processor.file_to_write
    assert "product.attribute.value.csv" in processor.file_to_write
    assert "product.attribute.line.csv" in processor.file_to_write
