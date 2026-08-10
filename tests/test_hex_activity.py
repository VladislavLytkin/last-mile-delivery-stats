"""Тесты панели активности и эмпирической классификации закрытия."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from last_mile.filter import STUDY_END, STUDY_START
from last_mile.hex_activity import (
    EMPIRICAL_CLOSED,
    CONTINUES_ACTIVE,
    INSUFFICIENT_POST_WINDOW,
    SPARSE_PRE_ACTIVITY,
    build_hex_activity_panel,
    classify_empirical_activity,
    compute_closure_window_metrics,
    validate_hex_activity_panel,
)


def _toy_hex_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hex": ["H1", "H2", "H3"],
            "treatment_date": pd.to_datetime(
                ["2022-08-01", "2022-08-01", None]
            ),
            "change_type": ["closed", "region_only", "other"],
            "is_treated": [True, True, False],
            "subject": ["s", "s", "s"],
            "region_id_old": [1, 1, 1],
            "region_id_new": [-100, 2, 1],
            "work_mode_old": [1, 1, 1],
            "work_mode_new": [0, 1, 1],
            "cohort": pd.to_datetime(["2022-08-01", "2022-08-01", None]),
        }
    )


def _toy_apps() -> pd.DataFrame:
    # H1: active pre, zero post -> empirical_closed candidate
    # H2: active both sides
    # H3: control with sparse activity
    rows = []
    for d in pd.date_range("2022-07-20", "2022-07-31", freq="D"):
        rows.append(
            {
                "hex": "H1",
                "request_timestamp": d + pd.Timedelta(hours=10),
                "change_type": "closed",
                "is_treated": True,
                "cohort": pd.Timestamp("2022-08-01"),
            }
        )
        rows.append(
            {
                "hex": "H2",
                "request_timestamp": d + pd.Timedelta(hours=11),
                "change_type": "region_only",
                "is_treated": True,
                "cohort": pd.Timestamp("2022-08-01"),
            }
        )
    for d in pd.date_range("2022-08-01", "2022-08-20", freq="D"):
        rows.append(
            {
                "hex": "H2",
                "request_timestamp": d + pd.Timedelta(hours=12),
                "change_type": "region_only",
                "is_treated": True,
                "cohort": pd.Timestamp("2022-08-01"),
            }
        )
    for d in ["2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01", "2022-09-01"]:
        rows.append(
            {
                "hex": "H3",
                "request_timestamp": pd.Timestamp(d) + pd.Timedelta(hours=9),
                "change_type": "other",
                "is_treated": False,
                "cohort": pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def test_build_hex_activity_panel_includes_zero_days():
    hex_meta = _toy_hex_meta()
    apps = _toy_apps()
    panel = build_hex_activity_panel(
        hex_meta=hex_meta,
        apps_clean=apps,
        study_start=pd.Timestamp("2022-07-20"),
        study_end=pd.Timestamp("2022-08-20"),
    )
    assert panel.duplicated(subset=["hex", "date"]).sum() == 0
    assert panel["n_orders"].isna().sum() == 0
    assert (panel["n_orders"] == 0).any()
    apps_in_window = apps[
        (apps["request_timestamp"].dt.normalize() >= pd.Timestamp("2022-07-20"))
        & (apps["request_timestamp"].dt.normalize() <= pd.Timestamp("2022-08-20"))
    ]
    assert int(panel["n_orders"].sum()) == len(apps_in_window)
    n_days = (pd.Timestamp("2022-08-20") - pd.Timestamp("2022-07-20")).days + 1
    assert len(panel) == 3 * n_days


def test_validate_rejects_order_sum_mismatch():
    hex_meta = _toy_hex_meta()
    apps = _toy_apps()
    panel = build_hex_activity_panel(
        hex_meta=hex_meta,
        apps_clean=apps,
        study_start=pd.Timestamp("2022-07-20"),
        study_end=pd.Timestamp("2022-08-20"),
    )
    bad = panel.copy()
    bad.loc[0, "n_orders"] = bad.loc[0, "n_orders"] + 1
    with pytest.raises(ValueError, match="сумма n_orders"):
        validate_hex_activity_panel(bad, apps)


def test_empirical_closed_and_continues_active():
    hex_meta = _toy_hex_meta()
    apps = _toy_apps()
    panel = build_hex_activity_panel(
        hex_meta=hex_meta,
        apps_clean=apps,
        study_start=pd.Timestamp("2022-07-01"),
        study_end=pd.Timestamp("2022-09-01"),
    )
    metrics = compute_closure_window_metrics(panel, window_days=28)
    cls = classify_empirical_activity(metrics, window_days=28, min_pre_orders=5)
    by_hex = cls.set_index("hex")["empirical_status"]
    assert by_hex["H1"] == EMPIRICAL_CLOSED
    assert by_hex["H2"] == CONTINUES_ACTIVE


def test_insufficient_post_window_near_study_end():
    hex_meta = pd.DataFrame(
        {
            "hex": ["Hlate"],
            "treatment_date": [pd.Timestamp("2022-10-10")],
            "change_type": ["closed"],
            "is_treated": [True],
            "subject": ["s"],
            "region_id_old": [1],
            "region_id_new": [-100],
            "work_mode_old": [1],
            "work_mode_new": [0],
            "cohort": [pd.Timestamp("2022-10-10")],
        }
    )
    apps_rows = []
    for d in pd.date_range("2022-09-20", "2022-10-09", freq="D"):
        apps_rows.append(
            {
                "hex": "Hlate",
                "request_timestamp": d + pd.Timedelta(hours=8),
                "change_type": "closed",
                "is_treated": True,
                "cohort": pd.Timestamp("2022-10-10"),
            }
        )
    apps = pd.DataFrame(apps_rows)
    panel = build_hex_activity_panel(
        hex_meta=hex_meta,
        apps_clean=apps,
        study_start=STUDY_START,
        study_end=STUDY_END,
    )
    metrics = compute_closure_window_metrics(panel, window_days=28)
    cls = classify_empirical_activity(metrics, window_days=28)
    assert cls.loc[0, "empirical_status"] == INSUFFICIENT_POST_WINDOW


def test_safe_ratio_not_zero_when_pre_orders_zero():
    metrics = pd.DataFrame(
        [
            {
                "hex": "Hz",
                "window_days": 28,
                "treatment_date": pd.Timestamp("2022-08-01"),
                "change_type": "closed",
                "subject": "s",
                "cohort": pd.Timestamp("2022-08-01"),
                "pre_observed_days": 28,
                "post_observed_days": 28,
                "pre_orders": 0,
                "post_orders": 0,
                "pre_mean_orders_per_day": 0.0,
                "post_mean_orders_per_day": 0.0,
                "pre_active_days": 0,
                "post_active_days": 0,
                "pre_zero_share": 1.0,
                "post_zero_share": 1.0,
                "post_to_pre_orders_ratio": np.nan,
                "last_request_date": pd.NaT,
                "last_request_relative_day": np.nan,
                "first_post_request_date": pd.NaT,
                "first_post_request_relative_day": np.nan,
                "max_consecutive_zero_days_post": 28,
                "full_post_window": True,
                "sufficient_pre_activity": False,
            }
        ]
    )
    cls = classify_empirical_activity(metrics, window_days=28)
    assert cls.loc[0, "empirical_status"] == SPARSE_PRE_ACTIVITY
