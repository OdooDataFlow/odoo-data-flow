"""Tests for the sorting strategies."""

from pathlib import Path

import polars as pl
import pytest

from odoo_data_flow.lib.sort import sort_for_self_referencing


@pytest.fixture
def hierarchical_csv(tmp_path: Path) -> str:
    """Creates a sample hierarchical CSV file for testing."""
    csv_content = """id,name,parent_id
p1,Parent One,
c1,Child One,p1
c2,Child Two,p1
p2,Parent Two,
c3,Child Three,p2
"""
    file_path = tmp_path / "hierarchical.csv"
    file_path.write_text(csv_content)
    return str(file_path)


@pytest.fixture
def non_hierarchical_csv(tmp_path: Path) -> str:
    """Creates a sample non-hierarchical CSV file for testing."""
    csv_content = """id,name,category_id
1,Product A,cat1
2,Product B,cat1
3,Product C,cat2
"""
    file_path = tmp_path / "non_hierarchical.csv"
    file_path.write_text(csv_content)
    return str(file_path)


def test_sorts_correctly_when_self_referencing(hierarchical_csv: str) -> None:
    """Verify that a self-referencing CSV is sorted correctly."""
    sorted_file = sort_for_self_referencing(
        hierarchical_csv, id_column="id", parent_column="parent_id", separator=","
    )
    assert sorted_file is not None
    # Make sure it's not False (error case)
    assert sorted_file is not False
    # Make sure it's a string (not True)
    assert isinstance(sorted_file, str)

    sorted_df = pl.read_csv(sorted_file)
    # Parents (p1, p2) should be the first two rows
    parent_ids = sorted_df.head(2).get_column("id").to_list()
    assert "p1" in parent_ids
    assert "p2" in parent_ids
    # Check the full order
    expected_order = ["p1", "p2", "c1", "c2", "c3"]
    assert sorted_df.get_column("id").to_list() == expected_order


def test_returns_none_when_not_self_referencing(non_hierarchical_csv: str) -> None:
    """Verify that None is returned if the hierarchy is not self-referencing."""
    sorted_file = sort_for_self_referencing(
        non_hierarchical_csv, id_column="id", parent_column="category_id", separator=","
    )
    assert sorted_file is None


def test_returns_none_if_columns_missing() -> None:
    """Verify that None is returned if the required columns don't exist."""
    # Create a dummy file
    csv_content = "a,b,c\n1,2,3"
    file_path = Path("dummy.csv")
    file_path.write_text(csv_content)

    assert (
        sort_for_self_referencing(
            str(file_path), id_column="id", parent_column="parent_id", separator=","
        )
        is None
    )
    file_path.unlink()


def test_returns_false_for_non_existent_file() -> None:
    """Verify that False is returned if the input file does not exist."""
    result = sort_for_self_referencing(
        "non_existent.csv", id_column="id", parent_column="parent_id", separator=","
    )
    assert result is False
