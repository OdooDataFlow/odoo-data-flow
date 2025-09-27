"""Tests for the tools module."""

from odoo_data_flow.lib.internal.tools import (
    AttributeLineDict,
    batch,
    to_m2m,
    to_m2o,
    to_xmlid,
)


def test_to_xmlid() -> None:
    """Test the to_xmlid function."""
    assert to_xmlid("A.B,C\nD|E F") == "A.B_C_D_E_F"
    assert (
        to_xmlid("  leading and trailing spaces  ") == "__leading_and_trailing_spaces__"
    )
    assert to_xmlid("no_special_chars") == "no_special_chars"
    assert to_xmlid("") == ""


def test_to_m2o() -> None:
    """Test the to_m2o function."""
    assert to_m2o("prefix", "value") == "prefix.value"
    assert to_m2o("prefix.", "value") == "prefix.value"
    assert to_m2o("prefix", " a value with spaces ") == "prefix._a_value_with_spaces_"
    assert to_m2o("prefix", "") == ""
    assert to_m2o("prefix", "", default="default_value") == "default_value"


def test_to_m2m() -> None:
    """Test the to_m2m function."""
    assert to_m2m("prefix", "val1,val2,val3") == "prefix.val1,prefix.val2,prefix.val3"
    assert to_m2m("prefix", " val1 , val2 ") == "prefix.val1,prefix.val2"
    assert to_m2m("prefix", "") == ""
    assert to_m2m("prefix", "val1") == "prefix.val1"


def test_batch() -> None:
    """Test the batch function."""
    assert list(batch(range(10), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert list(batch(range(5), 5)) == [[0, 1, 2, 3, 4]]
    assert list(batch(range(3), 5)) == [[0, 1, 2]]
    assert list(batch([], 5)) == []


def test_attribute_line_dict() -> None:
    """Test the AttributeLineDict class."""

    # - Arrange
    def id_gen_fun(template_id: str, attributes: dict[str, list[str]]) -> str:
        return f"id_{template_id}_{next(iter(attributes.keys()))}"

    attribute_list_ids = [
        ["att_id_1", "att_name_1"],
        ["att_id_2", "att_name_2"],
    ]
    aggregator = AttributeLineDict(attribute_list_ids, id_gen_fun)
    header = ["product_tmpl_id/id", "attribute_id/id", "value_ids/id"]
    line1 = [
        "template_1",
        {"att_name_1": "att_id_1"},
        {"att_name_1": "val_1"},
    ]
    line2 = [
        "template_1",
        {"att_name_2": "att_id_2"},
        {"att_name_2": "val_2"},
    ]
    line3 = [
        "template_2",
        {"att_name_1": "att_id_1"},
        {"att_name_1": "val_3"},
    ]

    # Act
    aggregator.add_line(line1, header)
    aggregator.add_line(line2, header)
    aggregator.add_line(line3, header)
    lines_header, lines_out = aggregator.generate_line()

    # Assert
    assert lines_header == [
        "id",
        "product_tmpl_id/id",
        "attribute_id/id",
        "value_ids/id",
    ]
    assert lines_out == [
        [
            "id_template_1_att_id_1",
            "template_1",
            "att_id_1",
            "val_1",
        ],
        [
            "id_template_1_att_id_1",
            "template_1",
            "att_id_2",
            "val_2",
        ],
        [
            "id_template_2_att_id_1",
            "template_2",
            "att_id_1",
            "val_3",
        ],
    ]
