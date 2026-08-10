from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_closure_did_sensitivity import (
    WIDE_VALUE_COLUMNS,
    build_closure_sensitivity_wide,
)


ROOT = Path(__file__).resolve().parents[1]
LONG_PATH = ROOT / "outputs" / "final" / "closure_did_sensitivity.csv"
WIDE_PATH = ROOT / "outputs" / "final" / "closure_did_sensitivity_wide.csv"


def test_closure_wide_csv_has_one_clean_header() -> None:
    frame = pd.read_csv(WIDE_PATH)

    assert frame.columns.is_unique
    assert not any(str(column).startswith("Unnamed:") for column in frame.columns)
    assert len(frame) == 12
    assert set(frame["outcome"]) == {
        "sch_flg",
        "meet_flg",
        "success_within_20",
        "utlz_within_25",
    }
    assert "specification" not in set(frame["outcome"].astype(str))


def test_closure_wide_preserves_all_numeric_values() -> None:
    long = pd.read_csv(LONG_PATH)
    actual = pd.read_csv(WIDE_PATH).sort_values(["outcome", "change_type"]).reset_index(
        drop=True
    )
    expected = build_closure_sensitivity_wide(long).sort_values(
        ["outcome", "change_type"]
    ).reset_index(drop=True)

    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == long.groupby(["outcome", "change_type"]).ngroups
    for column in [
        f"{metric}_spec_{specification}"
        for metric in WIDE_VALUE_COLUMNS
        for specification in ("A", "B", "C")
    ]:
        assert np.allclose(actual[column], expected[column], equal_nan=True)
