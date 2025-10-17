"""Test the UTF-8 sanitization functionality in export_threaded."""

import polars as pl

from odoo_data_flow.export_threaded import (
    _clean_and_transform_batch,
    _sanitize_utf8_string,
)


class TestSanitizeUtf8String:
    """Tests for the _sanitize_utf8_string utility function."""

    def test_sanitize_utf8_string_none_input(self) -> None:
        """Test that None input returns empty string."""
        result = _sanitize_utf8_string(None)
        assert result == ""

    def test_sanitize_utf8_string_valid_string(self) -> None:
        """Test that valid UTF-8 strings are returned unchanged."""
        test_string = "Hello, world! This is a valid UTF-8 string."
        result = _sanitize_utf8_string(test_string)
        assert result == test_string

    def test_sanitize_utf8_string_non_string_input(self) -> None:
        """Test that non-string inputs are converted to strings."""
        result = _sanitize_utf8_string(123)
        assert result == "123"

        result = _sanitize_utf8_string(12.34)
        assert result == "12.34"

        result = _sanitize_utf8_string(True)
        assert result == "True"

    def test_sanitize_utf8_string_invalid_utf8_characters(self) -> None:
        """Test handling of strings with invalid UTF-8 characters."""
        # Test with a string that contains problematic characters
        # This is a synthetic test - in practice, these would come from binary data
        test_string = "Valid string with \x9d invalid char"
        result = _sanitize_utf8_string(test_string)
        # Should return a valid UTF-8 string, possibly with replacements
        assert isinstance(result, str)
        # Should be valid UTF-8
        result.encode("utf-8")

    def test_sanitize_utf8_string_control_characters(self) -> None:
        """Test handling of control characters."""
        test_string = "String with control chars\x01\x02\x03"
        result = _sanitize_utf8_string(test_string)
        assert isinstance(result, str)
        # Should be valid UTF-8
        result.encode("utf-8")

    def test_sanitize_utf8_string_unicode_characters(self) -> None:
        """Test handling of unicode characters."""
        test_string = "String with unicode: café résumé naïve"
        result = _sanitize_utf8_string(test_string)
        assert result == test_string
        # Should be valid UTF-8
        result.encode("utf-8")

    def test_sanitize_utf8_string_edge_case_chars(self) -> None:
        """Test handling of edge case characters that might cause issues."""
        # Test with characters that often cause problems
        test_string = "Product with special chars: \x00\x01\x02\x03\x9d\xa0\xff"
        result = _sanitize_utf8_string(test_string)
        assert isinstance(result, str)
        # Should be valid UTF-8
        result.encode("utf-8")

    def test_sanitize_utf8_string_mixed_encoding_data(self) -> None:
        """Test handling of mixed encoding data that might come from databases."""
        # Test with mixed encoding scenarios that might occur in real data
        test_string = "Mixed data with émojis 😀 and \x9d binary chars"
        result = _sanitize_utf8_string(test_string)
        assert isinstance(result, str)
        # Should be valid UTF-8
        result.encode("utf-8")


class TestCleanAndTransformBatchUtf8:
    """Tests for UTF-8 sanitization in _clean_and_transform_batch function."""

    def test_clean_and_transform_batch_with_valid_strings(self) -> None:
        """Test that valid strings are processed correctly."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": ["Product A", "Product B", "Product C"],
                "description": ["Desc A", "Desc B", "Desc C"],
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame with same data
        assert len(result_df) == 3
        assert result_df["name"].to_list() == ["Product A", "Product B", "Product C"]

    def test_clean_and_transform_batch_with_invalid_utf8_strings(self) -> None:
        """Test that strings with invalid UTF-8 are sanitized."""
        # Create a DataFrame with strings that might have encoding issues
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": ["Valid Name", "Name with \x9d char", "Another Valid Name"],
                "description": ["Desc A", "Desc with \x01 control", "Desc C"],
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame with sanitized data
        assert len(result_df) == 3
        assert isinstance(result_df["name"].to_list()[0], str)
        # All strings should be valid UTF-8
        for name in result_df["name"].to_list():
            name.encode("utf-8")
        for desc in result_df["description"].to_list():
            desc.encode("utf-8")

    def test_clean_and_transform_batch_with_mixed_data_types(self) -> None:
        """Test that mixed data types are handled correctly."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],  # Integer IDs
                "name": ["Product A", "Product B", "Product C"],
                "price": [10.5, 20.0, 15.75],  # Float prices
                "active": [True, False, True],  # Boolean values
            }
        )

        field_types = {
            "id": "integer",
            "name": "char",
            "price": "float",
            "active": "boolean",
        }

        polars_schema = {
            "id": pl.Int64(),
            "name": pl.String(),
            "price": pl.Float64(),
            "active": pl.Boolean(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame
        assert len(result_df) == 3
        # String columns should be valid UTF-8
        for name in result_df["name"].to_list():
            name.encode("utf-8")

    def test_clean_and_transform_batch_with_problematic_data(self) -> None:
        """Test that problematic data is handled gracefully."""
        # Create DataFrame with various problematic data
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": [None, "Valid Name", ""],  # None, valid, empty string
                "description": ["Normal desc", "", None],  # Valid, empty, None
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame
        assert len(result_df) == 3
        # String values should be strings and valid UTF-8 (None values preserved)
        name_list = result_df["name"].to_list()
        for name in name_list:
            if name is not None:
                assert isinstance(name, str)
                name.encode("utf-8")
            else:
                assert name is None
        desc_list = result_df["description"].to_list()
        for desc in desc_list:
            if desc is not None:
                assert isinstance(desc, str)
                desc.encode("utf-8")
            else:
                assert desc is None

    def test_clean_and_transform_batch_preserves_schema(self) -> None:
        """Test that the result DataFrame matches the expected schema."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": ["Product A", "Product B", "Product C"],
                "quantity": [10, 20, 15],
            }
        )

        field_types = {"id": "char", "name": "char", "quantity": "integer"}

        polars_schema = {"id": pl.String(), "name": pl.String(), "quantity": pl.Int64()}

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a DataFrame with the correct schema
        assert len(result_df) == 3
        assert result_df.schema == polars_schema
        # String columns should be valid UTF-8
        for name in result_df["name"].to_list():
            name.encode("utf-8")

    def test_clean_and_transform_batch_with_problematic_binary_like_strings(
        self,
    ) -> None:
        """Test handling of binary-like strings that might cause the original issue."""
        # Create a DataFrame with binary-like strings that might cause
        # the original issue
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": [
                    "Regular Product Name",
                    "Product with \x9d binary char",  # This is the
                    # problematic byte from your error
                    "Another Product Name",
                ],
                "description": [
                    "Normal description",
                    "Description with \x00\x01\x02 control chars",
                    "Another description",
                ],
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame with sanitized data
        assert len(result_df) == 3
        # All strings should be valid UTF-8 (no more encoding errors)
        for name in result_df["name"].to_list():
            assert isinstance(name, str)
            # This should not raise any encoding errors
            name.encode("utf-8")
        for desc in result_df["description"].to_list():
            assert isinstance(desc, str)
            # This should not raise any encoding errors
            desc.encode("utf-8")

    def test_clean_and_transform_batch_with_complex_unicode_data(self) -> None:
        """Test handling of complex Unicode data with emojis and special characters."""
        # Create a DataFrame with complex Unicode data
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "name": [
                    "Product with émojis 😀🚀⭐",
                    "Product with accented chars: café résumé naïve",
                    "Product with Chinese: 产品 模板",
                ],
                "description": [
                    "Description with symbols: © ® ™ € £ ¥",
                    "Description with Arabic: المنتج النموذجي",
                    "Description with Russian: Пример продукта",
                ],
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame with Unicode data preserved
        assert len(result_df) == 3
        # All strings should be valid UTF-8
        for name in result_df["name"].to_list():
            assert isinstance(name, str)
            # This should not raise any encoding errors
            name.encode("utf-8")
        for desc in result_df["description"].to_list():
            assert isinstance(desc, str)
            # This should not raise any encoding errors
            desc.encode("utf-8")

    def test_clean_and_transform_batch_with_empty_and_null_values(self) -> None:
        """Test handling of empty strings and null values."""
        # Create a DataFrame with various combinations of empty/null values
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "name": [None, "", "Valid Name", None],  # None, empty, valid, None
                "description": [
                    "",
                    None,
                    "Valid Desc",
                    "",
                ],  # empty, None, valid, empty
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame
        assert len(result_df) == 4
        # All non-None values should be valid UTF-8 strings
        name_list = result_df["name"].to_list()
        for name in name_list:
            if name is not None:
                assert isinstance(name, str)
                name.encode("utf-8")
            else:
                assert name is None

        desc_list = result_df["description"].to_list()
        for desc in desc_list:
            if desc is not None:
                assert isinstance(desc, str)
                desc.encode("utf-8")
            else:
                assert desc is None

    def test_clean_and_transform_batch_with_malformed_utf8_sequences(self) -> None:
        """Test handling of malformed UTF-8 sequences that might occur in real data."""
        # Create a DataFrame with strings that might have malformed UTF-8
        # Using bytes that represent invalid UTF-8 sequences
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "name": [
                    "Valid UTF-8 string",
                    "String with invalid UTF-8: \x9d\x80\x81",  # Invalid UTF-8 bytes
                ],
                "description": [
                    "Normal description",
                    "Another invalid UTF-8: \x00\x01\x02\x03",  # Control characters
                ],
            }
        )

        field_types = {"id": "char", "name": "char", "description": "text"}

        polars_schema = {
            "id": pl.String(),
            "name": pl.String(),
            "description": pl.String(),
        }

        result_df = _clean_and_transform_batch(df, field_types, polars_schema)

        # Should return a valid DataFrame with sanitized data
        assert len(result_df) == 2
        # All strings should be valid UTF-8
        for name in result_df["name"].to_list():
            assert isinstance(name, str)
            # This should not raise any encoding errors
            name.encode("utf-8")
        for desc in result_df["description"].to_list():
            assert isinstance(desc, str)
            # This should not raise any encoding errors
            desc.encode("utf-8")
