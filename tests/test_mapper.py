"""Test the core mapper functions."""

import inspect
import logging
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import requests  # type: ignore[import-untyped]

from odoo_data_flow.lib import mapper
from odoo_data_flow.lib.internal.exceptions import SkippingError

# Import MapperFunc for type hinting in this test file
from odoo_data_flow.lib.mapper import MapperFunc

# --- Type Aliases ---
LineDict = dict[str, Any]
StateDict = dict[str, Any]


# --- Mocking external dependencies for isolated testing ---


# Placeholder for to_m2o that mimics its behavior
def _mock_to_m2o(prefix: str, value: Any, default: str = "") -> str:
    if value:
        return f"{prefix}.{value}"
    return default


# Placeholder for _get_field_value that mimics its behavior
def _mock_get_field_value(line: LineDict, field: str, default: Any = None) -> Any:
    return line.get(field, default)


# Placeholder for concat that mimics its behavior, and will now explicitly use state
def _mock_concat(
    sep: str, *fields: Any, default: str = "", skip: bool = False
) -> Callable[[LineDict, StateDict], str]:
    def concat_fun(line: LineDict, state: StateDict) -> str:
        state.setdefault("concat_calls", 0)
        state["concat_calls"] = int(state["concat_calls"]) + 1
        raw_state_suffix = str(state.get("suffix", ""))

        parts: list[str] = [
            str(line.get(f, "")) for f in fields if line.get(f) is not None
        ]

        if raw_state_suffix:
            # If the raw_state_suffix already starts with the separator,
            # remove it for joining
            if raw_state_suffix.startswith(sep):
                effective_suffix = raw_state_suffix[len(sep) :]
            else:
                effective_suffix = raw_state_suffix
            parts.append(effective_suffix)

        value = sep.join(parts)  # Calculate the value before potentially skipping

        # FIX: Add the skipping logic here
        if skip and not value:
            raise SkippingError(f"Missing value for concat with fields {fields}")

        return value

    return concat_fun


# --- Pytest Fixtures for Patching ---
@pytest.fixture(autouse=True)
def mock_mapper_dependencies(mocker: MagicMock) -> None:
    """Fixture to mock external dependencies in mapper.py."""
    mocker.patch("odoo_data_flow.lib.mapper.to_m2o", side_effect=_mock_to_m2o)
    mocker.patch(
        "odoo_data_flow.lib.mapper._get_field_value", side_effect=_mock_get_field_value
    )
    mocker.patch("odoo_data_flow.lib.mapper.concat", side_effect=_mock_concat)
    # Patch log to prevent actual logging during tests if desired,
    # or let it log to stdout
    mocker.patch("odoo_data_flow.lib.mapper.log", logging.getLogger("test_logger"))


# --- Test Data ---
LINE_SIMPLE: LineDict = {"col1": "A", "col2": "B", "col3": "C", "empty_col": ""}
LINE_NUMERIC: LineDict = {"price": "12,50", "qty": "100"}
LINE_M2M: LineDict = {"tags": "T1, T2", "other_tags": "T3", "empty_tags": ""}
LINE_BOOL: LineDict = {"is_active": "yes", "is_vip": "no"}
LINE_HIERARCHY: LineDict = {
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


def test_val_postprocess_fallback(mocker: MagicMock) -> None:
    """Test post process fallback.

    Tests the val mapper's fallback from a 2-arg to a 1-arg postprocess call.
    This simulates a callable that doesn't accept the 'state' argument.
    """

    def one_arg_lambda(x: str) -> str:
        return x.lower()

    mock_signature = mocker.patch("inspect.signature")
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
    # With the new m2m logic, this should split both "tags" and "other_tags"
    # "tags" ("T1, T2") should become "tag_prefix.T1", "tag_prefix.T2"
    # "other_tags" ("T3") should become "tag_prefix.T3"
    # Joined by comma: "tag_prefix.T1,tag_prefix.T2,tag_prefix.T3"
    mapper_func = mapper.m2m("tag_prefix", "tags", "other_tags")
    result = mapper_func(LINE_M2M, {})
    assert result == "tag_prefix.T1,tag_prefix.T2,tag_prefix.T3"


def test_m2m_multi_column_with_missing_field() -> None:
    """Tests the m2m mapper in multi-column mode with a non-existent field."""
    # "tags" ("T1, T2") should become "tag_prefix.T1", "tag_prefix.T2"
    # "non_existent_field" will be empty/None
    # Joined by comma: "tag_prefix.T1,tag_prefix.T2"
    mapper_func = mapper.m2m("tag_prefix", "tags", "non_existent_field")
    result = mapper_func(LINE_M2M, {})
    assert result == "tag_prefix.T1,tag_prefix.T2"


def test_m2m_multi_column_with_empty_value() -> None:
    """Tests the m2m mapper in multi-column mode with an empty field value."""
    line_with_empty: LineDict = {"f1": "val1", "f2": ""}
    mapper_func = mapper.m2m("p", "f1", "f2")
    assert mapper_func(line_with_empty, {}) == "p.val1"


def test_m2m_single_empty_field() -> None:
    """Tests the m2m mapper in single-column mode with an empty field."""
    # "empty_tags" is "", so it should return an empty string.
    mapper_func = mapper.m2m("tag_prefix", "empty_tags", sep=",")
    assert mapper_func(LINE_M2M, {}) == ""


# Add a specific test for m2m single column with a comma-separated value
def test_m2m_single_column_splits_value() -> None:
    """Tests that m2m in single-column mode correctly splits the field value."""
    line = {"products": "PROD1, PROD2,PROD3"}
    mapper_func = mapper.m2m("prod_prefix", "products", sep=",")
    assert (
        mapper_func(line, {}) == "prod_prefix.PROD1,prod_prefix.PROD2,prod_prefix.PROD3"
    )


def test_m2m_single_column_splits_value_with_custom_sep() -> None:
    """Tests that m2m in single-column mode correctly splits with custom separator."""
    line = {"items": "ITEM-A; ITEM-B"}
    mapper_func = mapper.m2m("item_prefix", "items", sep=";")
    assert mapper_func(line, {}) == "item_prefix.ITEM-A,item_prefix.ITEM-B"


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
    assert mapper_func(LINE_SIMPLE, {}) == []


def test_m2m_value_list_empty() -> None:
    """Tests that m2m_value_list returns an empty list for empty input."""
    mapper_func = mapper.m2m_value_list("empty_col")
    assert mapper_func(LINE_SIMPLE, {}) == []


def test_map_val_m2m() -> None:
    """Tests the map_val mapper in m2m mode."""
    translation_map: dict[str, str] = {"T1": "Tag One", "T2": "Tag Two"}
    mapper_func = mapper.map_val(translation_map, mapper.val("tags"), m2m=True)
    assert mapper_func(LINE_M2M, {}) == "Tag One,Tag Two"


def test_record_mapper() -> None:
    """Tests that the record mapper correctly creates a dictionary of results."""
    line_mapping: dict[str, MapperFunc] = {
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


def test_binary_url_map_skip_on_not_found(mocker: MagicMock) -> None:
    """Tests that binary_url_map raises SkippingError when request fails."""
    mock_requests_get = mocker.patch("odoo_data_flow.lib.mapper.requests.get")
    mock_requests_get.side_effect = requests.exceptions.RequestException("Timeout")
    mapper_func = mapper.binary_url_map("col1", skip=True)
    with pytest.raises(SkippingError):
        mapper_func(LINE_SIMPLE, {})


def test_binary_url_map_request_exception(mocker: MagicMock) -> None:
    """Tests that a warning is logged when a URL request fails and skip=False."""
    mock_requests_get = mocker.patch("odoo_data_flow.lib.mapper.requests.get")
    mock_log_warning = mocker.patch("odoo_data_flow.lib.mapper.log.warning")

    mock_requests_get.side_effect = requests.exceptions.RequestException("Timeout")
    mapper_func = mapper.binary_url_map("col1", skip=False)
    assert mapper_func(LINE_SIMPLE, {}) == ""
    mock_log_warning.assert_called_once()
    assert "Cannot fetch file" in mock_log_warning.call_args[0][0]


def test_legacy_mappers() -> None:
    """Tests the legacy attribute mappers."""
    line: LineDict = {"Color": "Blue", "Size": "L", "Finish": ""}

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
    line_with_template: LineDict = {"template_id": "TPL1", "Color": "Blue", "Size": "L"}
    mapper_func = mapper.m2m_template_attribute_value("PREFIX", "Color", "Size")
    assert mapper_func(line_with_template, {}) == "Blue,L"

    # Case 2: template_id is missing, should return an empty string
    line_without_template: LineDict = {"Color": "Blue", "Size": "L"}
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


def test_bool_val_mapper() -> None:
    """Tests the bool_val mapper with various configurations."""
    line = {"is_active": "yes", "is_vip": "no", "is_member": "true", "is_guest": ""}

    # Test with true_values
    mapper_true = mapper.bool_val("is_active", true_values=["yes", "true"])
    assert mapper_true(line, {}) == "1"
    assert mapper_true({"is_active": "no"}, {}) == "0"

    # Test with false_values
    mapper_false = mapper.bool_val("is_vip", false_values=["no", "false"])
    assert mapper_false(line, {}) == "0"
    assert mapper_false({"is_vip": "yes"}, {}) == "0"

    # Test with both true and false values
    mapper_both = mapper.bool_val(
        "is_member", true_values=["true"], false_values=["false"]
    )
    assert mapper_both(line, {}) == "1"
    assert mapper_both({"is_member": "false"}, {}) == "0"
    assert mapper_both({"is_member": "other"}, {}) == "0"  # Fallback to default

    # Test with default value
    mapper_default_true = mapper.bool_val("is_guest", default=True)
    assert mapper_default_true(line, {}) == "1"
    mapper_default_false = mapper.bool_val("is_guest", default=False)
    assert mapper_default_false(line, {}) == "0"

    # Test truthiness fallback
    mapper_truthy = mapper.bool_val("is_active")
    assert mapper_truthy(line, {}) == "1"
    assert mapper_truthy({"is_active": ""}, {}) == "0"


# --- NEW TESTS ---


def test_m2o_fun_state_present_but_unused(mocker: MagicMock) -> None:
    """Confirms m2o_fun works correctly when 'state' is provided but not directly used.

    Args:
        mocker: The pytest-mock fixture for patching.
    """
    mock_get_field_value = mocker.patch(
        "odoo_data_flow.lib.mapper._get_field_value", side_effect=_mock_get_field_value
    )
    mock_to_m2o = mocker.patch(
        "odoo_data_flow.lib.mapper.to_m2o", side_effect=_mock_to_m2o
    )

    mapper_func = mapper.m2o(prefix="test_prefix", field="name")
    line: LineDict = {"name": "TestValue"}
    state: StateDict = {"some_key": "some_value", "another_key": 123}

    result = mapper_func(line, state)

    assert result == "test_prefix.TestValue"
    mock_get_field_value.assert_called_once_with(line, "name")
    mock_to_m2o.assert_called_once_with("test_prefix", "TestValue", default="")
    assert state == {"some_key": "some_value", "another_key": 123}


def test_m2o_fun_with_skip_and_empty_value_state_unused(mocker: MagicMock) -> None:
    """Tests m2o_fun with 'skip' when value is empty, confirming state is unused.

    Args:
        mocker: The pytest-mock fixture for patching.
    """
    mock_get_field_value = mocker.patch(
        "odoo_data_flow.lib.mapper._get_field_value", side_effect=_mock_get_field_value
    )
    mock_to_m2o = mocker.patch(
        "odoo_data_flow.lib.mapper.to_m2o", side_effect=_mock_to_m2o
    )

    mapper_func = mapper.m2o(prefix="test_prefix", field="name", skip=True)
    line: LineDict = {"name": ""}
    state: StateDict = {"initial": "value"}

    with pytest.raises(SkippingError) as cm:
        mapper_func(line, state)

    assert str(cm.value) == "Missing Value for name"
    mock_get_field_value.assert_called_once_with(line, "name")
    mock_to_m2o.assert_not_called()
    assert state == {"initial": "value"}


def test_m2o_map_fun_state_passed_to_concat_mapper(mocker: MagicMock) -> None:
    """Test m2o_map state management.

    Confirms m2o_map passes 'state' to the underlying concat_mapper and
    concat_mapper uses it.

    Args:
        mocker: The pytest-mock fixture for patching.
    """
    mock_concat_actual = mocker.patch(
        "odoo_data_flow.lib.mapper.concat", side_effect=_mock_concat
    )
    mock_to_m2o = mocker.patch(
        "odoo_data_flow.lib.mapper.to_m2o", side_effect=_mock_to_m2o
    )

    mapper_func = mapper.m2o_map("test_prefix", "first", "last")
    line: LineDict = {"first": "John", "last": "Doe"}
    state: StateDict = {"suffix": "APP"}

    result = mapper_func(line, state)

    assert result == "test_prefix.John_Doe_APP"
    mock_concat_actual.assert_called_once_with("_", "first", "last")

    assert state["concat_calls"] == 1
    assert state["suffix"] == "APP"

    mock_to_m2o.assert_called_once_with("test_prefix", "John_Doe_APP", default="")


def test_m2o_map_fun_state_modified_by_concat_mapper(mocker: MagicMock) -> None:
    """Confirms m2o_map's underlying concat_mapper can modify the state.

    Args:
        mocker: The pytest-mock fixture for patching.
    """
    mocker.patch("odoo_data_flow.lib.mapper.concat", side_effect=_mock_concat)
    mapper_func = mapper.m2o_map("test_prefix", "code")
    line: LineDict = {"code": "A1B2"}
    state: StateDict = {"concat_calls": 0}

    mapper_func(line, state)

    assert state["concat_calls"] == 1


def test_m2o_map_fun_with_skip_and_empty_concat_value_state_passed(
    mocker: MagicMock,
) -> None:
    """Tests m2o_map.

    Tests m2o_map with 'skip' when concatenated value is empty,
    confirming state is passed.

    Args:
        mocker: The pytest-mock fixture for patching.
    """
    mock_concat_actual = mocker.patch(
        "odoo_data_flow.lib.mapper.concat", side_effect=_mock_concat
    )
    mock_to_m2o = mocker.patch(
        "odoo_data_flow.lib.mapper.to_m2o", side_effect=_mock_to_m2o
    )

    mapper_func = mapper.m2o_map("test_prefix", "non_existent_field", skip=True)
    line: LineDict = {}
    state: StateDict = {"some_info": "data"}

    with pytest.raises(SkippingError) as cm:
        mapper_func(line, state)

    assert "Missing value for m2o_map with prefix 'test_prefix'" in str(cm.value)
    mock_concat_actual.assert_called_once_with("_", "non_existent_field")
    assert state["concat_calls"] == 1
    mock_to_m2o.assert_not_called()
