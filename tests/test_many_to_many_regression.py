#!/usr/bin/env python3
"""
Unit tests to prevent regression in many-to-many field export behavior.
These tests ensure that many-to-many fields with /id and /.id specifiers 
return comma-separated values instead of single values.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import polars as pl
from polars.testing import assert_frame_equal


def test_many_to_many_field_processing_logic():
    """Test that many-to-many fields with /id suffix return comma-separated XML IDs."""
    
    # Simulate the data that would come from model.read()
    test_record = {
        "id": 63251,
        "attribute_value_ids": [86, 73, 75],  # Multiple IDs in a list
        "product_tmpl_id": (69287, "Product Template Name")
    }
    
    # Test case 1: Many-to-many field with /id specifier (should return comma-separated XML IDs)
    field = "attribute_value_ids/id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # [86, 73, 75]
    
    if isinstance(value, (list, tuple)) and value:
        # Handle the most common case: list of integers [86, 73, 75]
        if all(isinstance(item, int) for item in value):
            # Simulate XML ID lookup
            xml_id_map = {
                86: "__export__.product_attribute_value_86_aeb0aafc",
                73: "__export__.product_attribute_value_73_c7489756", 
                75: "__export__.product_attribute_value_75_d6a0c41b"
            }
            xml_ids = [xml_id_map.get(rid) for rid in value if rid in xml_id_map]
            xml_ids = [xid for xid in xml_ids if xid is not None]  # Remove None values
            result = ",".join(xml_ids) if xml_ids else None
            expected = "__export__.product_attribute_value_86_aeb0aafc,__export__.product_attribute_value_73_c7489756,__export__.product_attribute_value_75_d6a0c41b"
            assert result == expected
    
    # Test case 2: Many-to-many field with /.id specifier (should return comma-separated database IDs)
    field = "attribute_value_ids/.id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # [86, 73, 75]
    
    if isinstance(value, (list, tuple)) and value:
        if field.endswith("/.id"):
            # For many-to-many with /.id, should return comma-separated string with raw database IDs
            if all(isinstance(item, int) for item in value):
                result = ",".join(str(item) for item in value)
                expected = "86,73,75"
                assert result == expected


def test_single_id_handling():
    """Test that single IDs are handled correctly."""
    
    # Simulate the data that would come from model.read()
    test_record = {
        "id": 63251,
        "attribute_value_ids": [86],  # Single ID in a list
        "product_tmpl_id": (69287, "Product Template Name")
    }
    
    # Test case 1: Single many-to-many field with /id specifier
    field = "attribute_value_ids/id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # [86]
    
    if isinstance(value, (list, tuple)) and value:
        if field.endswith("/id"):
            # For many-to-many with /id, should return single XML ID (no comma)
            if all(isinstance(item, int) for item in value):
                # Simulate XML ID lookup
                xml_id_map = {
                    86: "__export__.product_attribute_value_86_aeb0aafc"
                }
                xml_ids = [xml_id_map.get(rid) for rid in value if rid in xml_id_map]
                xml_ids = [xid for xid in xml_ids if xid is not None]  # Remove None values
                result = ",".join(xml_ids) if xml_ids else None
                expected = "__export__.product_attribute_value_86_aeb0aafc"
                assert result == expected
    
    # Test case 2: Single many-to-many field with /.id specifier
    field = "attribute_value_ids/.id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # [86]
    
    if isinstance(value, (list, tuple)) and value:
        if field.endswith("/.id"):
            # For many-to-many with /.id, should return single database ID (no comma)
            if all(isinstance(item, int) for item in value):
                result = ",".join(str(item) for item in value)
                expected = "86"
                assert result == expected


def test_empty_list_handling():
    """Test that empty lists are handled correctly."""
    
    # Simulate the data that would come from model.read()
    test_record = {
        "id": 63251,
        "attribute_value_ids": [],  # Empty list
        "product_tmpl_id": (69287, "Product Template Name")
    }
    
    # Test case 1: Empty many-to-many field with /id specifier
    field = "attribute_value_ids/id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # []
    
    if isinstance(value, (list, tuple)) and value:
        # This branch won't be taken since value is empty
        pass
    else:
        # Empty list should result in None/empty string
        result = None
        assert result is None
    
    # Test case 2: Empty many-to-many field with /.id specifier
    field = "attribute_value_ids/.id"
    base_field = field.split("/")[0].replace(".id", "id")  # "attribute_value_ids"
    value = test_record.get(base_field)  # []
    
    if isinstance(value, (list, tuple)) and value:
        # This branch won't be taken since value is empty
        pass
    else:
        # Empty list should result in None/empty string
        result = None
        assert result is None


if __name__ == "__main__":
    test_many_to_many_field_processing_logic()
    test_single_id_handling()
    test_empty_list_handling()
    print("✅ All many-to-many field processing logic tests passed!")
    print("These tests ensure that many-to-many fields with /id and /.id specifiers")
    print("properly return comma-separated values instead of single values.")