"""Additional tests for importer.py to improve coverage."""

from pathlib import Path

from odoo_data_flow.importer import _count_lines, _infer_model_from_filename


def test_count_lines_file_not_found() -> None:
    """Test _count_lines with non-existent file."""
    result = _count_lines("/path/that/does/not/exist.csv")
    assert result == 0


def test_count_lines_with_content() -> None:
    """Test _count_lines with actual content."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("line1\nline2\nline3\n")
        temp_path = f.name

    try:
        result = _count_lines(temp_path)
        assert result == 3
    finally:
        Path(temp_path).unlink()


def test_infer_model_from_filename() -> None:
    """Test _infer_model_from_filename with various patterns."""
    # Test with standard patterns
    assert _infer_model_from_filename("res_partner.csv") == "res.partner"
    assert _infer_model_from_filename("account_move_line.csv") == "account.move.line"
    assert _infer_model_from_filename("product_product.csv") == "product.product"

    # Test with path
    assert _infer_model_from_filename("/some/path/res_partner.csv") == "res.partner"

    # Test with no match
    assert _infer_model_from_filename("unknown_file.txt") == "unknown.file"
