import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_spc_calibration_size import (
    development_pair_order,
    restrict_development_rows,
    selected_manifest,
)


def synthetic_rows():
    rows = []
    for subcategory in ("A", "B"):
        for pair_id in ("1", "2", "3"):
            rows.append(
                {
                    "_dataset": "dev_mc",
                    "subcategory": subcategory,
                    "pair_id": pair_id,
                }
            )
    rows.append(
        {"_dataset": "test_mc", "subcategory": "A", "pair_id": "held-out"}
    )
    return rows


def test_selected_calibration_subsets_are_nested_by_seed():
    order = development_pair_order(synthetic_rows(), seed=7)
    one = selected_manifest(order, 1)
    two = selected_manifest(order, 2)
    for subcategory in one:
        assert set(one[subcategory]).issubset(two[subcategory])


def test_restrict_development_rows_always_keeps_blind_test_rows():
    rows = synthetic_rows()
    selected = selected_manifest(development_pair_order(rows, seed=7), 1)
    restricted = restrict_development_rows(rows, selected)
    assert sum(row["_dataset"].startswith("dev_") for row in restricted) == 2
    assert sum(row["_dataset"].startswith("test_") for row in restricted) == 1
