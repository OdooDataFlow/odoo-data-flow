"""Test the core mapper functions."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import requests  # type: ignore[import-untyped]

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.internal.exceptions import SkippingError

# --- Test Data ---
LINE_SIMPLE = {"col1": "A", "col2": "B", "col3": "C", "empty_col": ""}
LINE_NUMERIC = {"price": "12,50", "qty": "100"}
LINE_M2M = {"tags": "T1, T2", "other_tags": "T3", "empty_tags": ""}
LINE_BOOL = {"is_active": "yes", "is_vip": "no"}
LINE_HIERARCHY = {
    "order_ref": "SO001",
    "product_sku": "PROD-A",
    "product_qty": "5",
}


def test_val_postprocess_builtin() -> None:
    """Post Process val tester.

    Tests the val mapper's postprocess with a built-in function that
    cannot be inspected, covering the try/except block.
    """
    mapper_func = mapper.val("col1", postprocess=str.lower)
    assert mapper_func(LINE_SIMPLE, {}) == "a"


def test_val_postprocess_fallback() -> None:
    """Test post process fallback.

    Tests the val mapper's fallback from a 2-arg to a 1-arg postprocess call.
    This simulates a callable that doesn't accept the 'state' argument.
    """

    def one_arg_lambda(x: str) -> str:
        return x.lower()

    # Force the two-argument call to ensure the fallback to one argument is tested
    with patch("inspect.signature") as mock_signature:
        # Pretend the signature check passed, forcing a TypeError on the call.
        # The parameters attribute must be a dictionary-like object.
        mock_signature.return_value.parameters = {
            "arg1": MagicMock(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
            "arg2": MagicMock(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
        }
        mapper_func = mapper.val("col1", postprocess=one_arg_lambda)
        assert mapper_func(LINE_SIMPLE, {}) == "a"


def test_concat_mapper_all() -> None:
    """Tests that concat_mapper_all returns an empty string if any value is empty."""
    mapper_func = mapper.concat_mapper_all("_", "col1", "col2")
    assert mapper_func(LINE_SIMPLE, {}) == "A_B"
    mapper_func_fail = mapper.concat_mapper_all("_", "col1", "empty_col")
    assert mapper_func_fail(LINE_SIMPLE, {}) == ""


def test_concat_skip_on_empty() -> None:
    """Tests that concat raises SkippingError when skip=True and result is empty."""
    mapper_func = mapper.concat("_", "empty_col", skip=True)
    with pytest.raises(SkippingError):
        mapper_func(LINE_SIMPLE, {})


def test_num_mapper() -> None:
    """Tests the num mapper for comma replacement."""
    mapper_func = mapper.num("price")
    assert mapper_func(LINE_NUMERIC, {}) == "12.50"


def test_m2o_map_success() -> None:
    """Tests a successful m2o_map operation."""
    mapper_func = mapper.m2o_map("prefix", "col1", "col2")
    assert mapper_func(LINE_SIMPLE, {}) == "prefix.A_B"


def test_m2m_multi_column() -> None:
    """Tests the m2m mapper in multi-column mode."""
    mapper_func = mapper.m2m("tag_prefix", "tags", "other_tags")
    result = mapper_func(LINE_M2M, {})
    assert "tag_prefix.T1__T2" in result
    assert "tag_prefix.T3" in result


def test_m2m_multi_column_with_missing_field() -> None:
    """Tests the m2m mapper in multi-column mode with a non-existent field."""
    mapper_func = mapper.m2m("tag_prefix", "tags", "non_existent_field")
    result = mapper_func(LINE_M2M, {})
    assert result == "tag_prefix.T1__T2"


def test_m2m_multi_column_with_empty_value() -> None:
    """Tests the m2m mapper in multi-column mode with an empty field value."""
    line_with_empty = {"f1": "val1", "f2": ""}
    mapper_func = mapper.m2m("p", "f1", "f2")
    result = mapper_func(line_with_empty, {})
    assert result == "p.val1"


def test_m2m_single_empty_field() -> None:
    """Tests the m2m mapper in single-column mode with an empty field."""
    mapper_func = mapper.m2m("tag_prefix", "empty_tags", sep=",")
    assert mapper_func(LINE_M2M, {}) == ""


def test_m2m_map_with_concat() -> None:
    """Tests m2m_map wrapping another mapper."""
    concat_mapper = mapper.concat(",", "tags", "other_tags")
    m2m_mapper = mapper.m2m_map("tag_prefix", concat_mapper)
    result = m2m_mapper(LINE_M2M, {})
    assert "tag_prefix.T1" in result
    assert "tag_prefix.T2" in result
    assert "tag_prefix.T3" in result


def test_m2m_map_with_empty_result() -> None:
    """Tests m2m_map when the wrapped mapper returns an empty value."""
    empty_mapper = mapper.val("empty_col")
    m2m_mapper = mapper.m2m_map("tag_prefix", empty_mapper)
    assert m2m_mapper(LINE_SIMPLE, {}) == ""


def test_m2m_id_list_empty() -> None:
    """Tests that m2m_id_list returns an empty string for empty input."""
    mapper_func = mapper.m2m_id_list("prefix", "empty_col")
    assert mapper_func(LINE_SIMPLE, {}) == ""


def test_m2m_value_list_empty() -> None:
    """Tests that m2m_value_list returns an empty list for empty input."""
    mapper_func = mapper.m2m_value_list("empty_col")
    assert mapper_func(LINE_SIMPLE, {}) == []


def test_map_val_m2m() -> None:
    """Tests the map_val mapper in m2m mode."""
    translation_map = {"T1": "Tag One", "T2": "Tag Two"}
    mapper_func = mapper.map_val(translation_map, mapper.val("tags"), m2m=True)
    assert mapper_func(LINE_M2M, {}) == "Tag One,Tag Two"


def test_record_mapper() -> None:
    """Tests that the record mapper correctly creates a dictionary of results."""
    line_mapping = {
        "product_id/id": mapper.m2o_map("prod_", "product_sku"),
        "product_uom_qty": mapper.num("product_qty"),
    }
    record_mapper = mapper.record(line_mapping)
    result = record_mapper(LINE_HIERARCHY, {})
    assert isinstance(result, dict)
    assert result.get("product_id/id") == "prod_.PROD-A"
    assert result.get("product_uom_qty") == "5"


def test_binary_empty_path() -> None:
    """Tests that the binary mapper returns an empty string for an empty path."""
    mapper_func = mapper.binary("empty_col")
    assert mapper_func(LINE_SIMPLE, {}) == ""


def test_binary_skip_on_not_found() -> None:
    """Tests that binary raises SkippingError when skip=True and file not found."""
    mapper_func = mapper.binary("col1", skip=True)
    with pytest.raises(SkippingError):
        mapper_func(LINE_SIMPLE, {})


@patch("odoo_data_flow.lib.mapper.log.warning")
def test_binary_file_not_found_no_skip(mock_log_warning: MagicMock) -> None:
    """Tests that a warning is logged when a file is not found and skip=False."""
    mapper_func = mapper.binary("col1", skip=False)
    assert mapper_func(LINE_SIMPLE, {}) == ""
    mock_log_warning.assert_called_once()
    assert "File not found" in mock_log_warning.call_args[0][0]


def test_binary_url_map_empty() -> None:
    """Tests that binary_url_map returns empty string for an empty URL."""
    mapper_func = mapper.binary_url_map("empty_col")
    assert mapper_func(LINE_SIMPLE, {}) == ""


@patch("odoo_data_flow.lib.mapper.requests.get")
def test_binary_url_map_skip_on_not_found(mock_requests_get: MagicMock) -> None:
    """Tests that binary_url_map raises SkippingError when request fails."""
    mock_requests_get.side_effect = requests.exceptions.RequestException("Timeout")
    mapper_func = mapper.binary_url_map("col1", skip=True)
    with pytest.raises(SkippingError):
        mapper_func(LINE_SIMPLE, {})


@patch("odoo_data_flow.lib.mapper.requests.get")
@patch("odoo_data_flow.lib.mapper.log.warning")
def test_binary_url_map_request_exception(
    mock_log_warning: MagicMock, mock_requests_get: MagicMock
) -> None:
    """Tests that a warning is logged when a URL request fails and skip=False."""
    mock_requests_get.side_effect = requests.exceptions.RequestException("Timeout")
    mapper_func = mapper.binary_url_map("col1", skip=False)
    assert mapper_func(LINE_SIMPLE, {}) == ""
    mock_log_warning.assert_called_once()
    assert "Cannot fetch file" in mock_log_warning.call_args[0][0]


def test_legacy_mappers() -> None:
    """Tests the legacy attribute mappers."""
    line = {"Color": "Blue", "Size": "L", "Finish": ""}

    val_att_mapper = mapper.val_att(["Color", "Size", "Finish"])
    assert val_att_mapper(line, {}) == {"Color": "Blue", "Size": "L"}

    m2o_att_mapper = mapper.m2o_att("ATT", ["Color", "Size"])
    assert m2o_att_mapper(line, {}) == {
        "Color": "ATT.Color_Blue",
        "Size": "ATT.Size_L",
    }

    concat_legacy_mapper = mapper.concat_field_value_m2m("_", "Color", "Size")
    assert concat_legacy_mapper(line, {}) == "Color_Blue,Size_L"

    m2m_att_val_mapper = mapper.m2m_attribute_value("PREFIX", "Color", "Size")
    assert "PREFIX.Color_Blue" in m2m_att_val_mapper(line, {})
    assert "PREFIX.Size_L" in m2m_att_val_mapper(line, {})


def test_modern_template_attribute_mapper() -> None:
    """Tests the m2m_template_attribute_value mapper for modern Odoo versions."""
    # Case 1: template_id exists, should return concatenated values
    line_with_template = {"template_id": "TPL1", "Color": "Blue", "Size": "L"}
    mapper_func = mapper.m2m_template_attribute_value("PREFIX", "Color", "Size")
    assert mapper_func(line_with_template, {}) == "Blue,L"

    # Case 2: template_id is missing, should return an empty string
    line_without_template = {"Color": "Blue", "Size": "L"}
    assert mapper_func(line_without_template, {}) == ""


def test_split_mappers() -> None:
    """Tests the split helper functions."""
    split_line_func = mapper.split_line_number(100)
    assert split_line_func({}, 0) == 0
    assert split_line_func({}, 99) == 0
    assert split_line_func({}, 100) == 1

    split_file_func = mapper.split_file_number(8)
    assert split_file_func({}, 7) == 7
    assert split_file_func({}, 8) == 0
