"""Минимальные проверки согласованности эконометрического пайплайна."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from last_mile.filter import (
    get_cohort_map,
    resolve_hexagon_treatments,
    classify_hexagons,
)
from last_mile.cohort_diagnostics import (
    EXCLUDED_COHORT_FOR_POST_MEETING,
    cohort_composition_report,
)
from last_mile.outcomes import add_success_within_horizon, success_horizon_coverage

REPO = Path(__file__).resolve().parents[1]


def test_resolve_multi_treatment_excludes_distinct_dates():
    hexagons = pd.DataFrame(
        {
            "hex": ["A", "A", "B", "B", "C", "D"],
            "treatment_date": pd.to_datetime(
                ["2022-08-12", "2022-09-22", "2022-08-12", "2022-08-12", None, "2022-08-19"]
            ),
            "region_id_old": [1, 1, 1, 1, -100, 1],
            "region_id_new": [2, 2, 2, 2, -100, 1],
            "work_mode_old": [0, 0, 0, 0, 0, 0],
            "work_mode_new": [0, 0, 0, 0, 0, 1],
            "subject": ["x"] * 6,
        }
    )
    cls = classify_hexagons(hexagons)
    resolved = resolve_hexagon_treatments(cls)

    assert "A" not in set(resolved["single_treatment"]["hex"])
    assert "A" in set(resolved["multi_treatment"]["hex"])
    assert resolved["single_treatment"]["hex"].is_unique
    # B has duplicate same date → kept once
    assert (resolved["single_treatment"]["hex"] == "B").sum() == 1
    assert set(resolved["single_treatment"]["hex"]) >= {"B", "C", "D"}


def test_resolve_same_date_metadata_conflict_is_diagnosed_and_excluded():
    hexagons = pd.DataFrame(
        {
            "hex": ["A", "A", "B", "B"],
            "treatment_date": pd.to_datetime(
                ["2022-08-12", "2022-08-12", "2022-08-19", "2022-08-19"]
            ),
            "region_id_old": [1, 1, 3, 3],
            "region_id_new": [2, 9, 4, 4],
            "work_mode_old": [0, 0, 1, 1],
            "work_mode_new": [0, 0, 2, 2],
            "subject": ["x", "x", "y", "y"],
        }
    )
    resolved = resolve_hexagon_treatments(classify_hexagons(hexagons))

    assert set(resolved["metadata_conflicts"]["hex"]) == {"A"}
    assert "A" not in set(resolved["single_treatment"]["hex"])
    assert (resolved["single_treatment"]["hex"] == "B").sum() == 1
    policy = resolved["diagnostics"].set_index("hex").loc["A", "policy"]
    assert policy == "exclude_same_date_metadata_conflict"


def test_cohort_map_preserves_treatment_metadata():
    hexagons = pd.DataFrame(
        {
            "hex": ["A"],
            "treatment_date": pd.to_datetime(["2022-08-12"]),
            "region_id_old": [1],
            "region_id_new": [2],
            "work_mode_old": [0],
            "work_mode_new": [1],
            "subject": ["x"],
        }
    )
    single = resolve_hexagon_treatments(classify_hexagons(hexagons))[
        "single_treatment"
    ]
    cohort_map = get_cohort_map(single)

    for col in [
        "treatment_date",
        "region_id_old",
        "region_id_new",
        "work_mode_old",
        "work_mode_new",
    ]:
        assert col in cohort_map.columns
    assert cohort_map.loc[0, "region_id_new"] == 2


def test_get_cohort_map_rejects_duplicate_hex():
    df = pd.DataFrame(
        {
            "hex": ["A", "A"],
            "cohort": pd.to_datetime(["2022-08-12", "2022-09-22"]),
            "change_type": ["region_only", "region_only"],
            "is_treated": [True, True],
        }
    )
    with pytest.raises(ValueError, match="не уникальна"):
        get_cohort_map(df)


def test_cohort_map_unique_after_resolve():
    hexagons = pd.DataFrame(
        {
            "hex": ["A", "B", "C"],
            "treatment_date": pd.to_datetime(["2022-08-12", None, "2022-08-19"]),
            "region_id_old": [1, -100, 1],
            "region_id_new": [2, -100, 1],
            "work_mode_old": [0, 0, 0],
            "work_mode_new": [0, 0, 1],
            "subject": ["x", "y", "z"],
        }
    )
    cls = classify_hexagons(hexagons)
    single = resolve_hexagon_treatments(cls)["single_treatment"]
    cohort_map = get_cohort_map(single)
    assert cohort_map["hex"].is_unique


def test_utlz_within_25_definition():
    request = pd.Timestamp("2022-08-01")
    cases = pd.DataFrame(
        {
            "request_timestamp": [request] * 4,
            "real_utilization_dttm": [
                request + pd.Timedelta(days=-1),  # negative lag
                request + pd.Timedelta(days=10),
                request + pd.Timedelta(days=25),
                request + pd.Timedelta(days=26),
            ],
        }
    )
    lag = (
        cases["real_utilization_dttm"] - cases["request_timestamp"]
    ).dt.total_seconds() / 86400.0
    valid = cases.loc[lag >= 0].copy()
    horizon = valid["request_timestamp"] + pd.Timedelta(days=25)
    valid["utlz_within_25"] = (
        valid["real_utilization_dttm"].notna()
        & (valid["real_utilization_dttm"] <= horizon)
    ).astype(int)

    assert list(valid["utlz_within_25"]) == [1, 1, 0]
    assert (lag < 0).sum() == 1


def test_success_within_20_uses_event_time_and_mature_sample():
    request = pd.Timestamp("2022-09-01")
    applications = pd.DataFrame(
        {
            "request_timestamp": [
                request,
                request,
                request,
                pd.Timestamp("2022-10-10"),
            ],
            "first_success_dttm": [
                request + pd.Timedelta(days=20),
                request + pd.Timedelta(days=21),
                request - pd.Timedelta(days=1),
                pd.Timestamp("2022-10-15"),
            ],
        }
    )
    result = add_success_within_horizon(
        applications, horizon_days=20, observation_end="2022-10-18"
    )

    assert list(result["mature_success_20"]) == [True, True, True, False]
    assert list(result.loc[:2, "success_within_20"]) == [1, 0, 0]
    assert pd.isna(result.loc[3, "success_within_20"])


def test_success_horizon_coverage_is_monotone_in_maturity():
    applications = pd.DataFrame(
        {
            "request_timestamp": pd.to_datetime(
                ["2022-09-01", "2022-09-25", "2022-10-10"]
            ),
            "first_success_dttm": pd.to_datetime(
                ["2022-09-10", None, "2022-10-12"]
            ),
        }
    )
    _, coverage = success_horizon_coverage(
        applications,
        horizons=(7, 20, 30),
        observation_end="2022-10-18",
    )
    assert coverage["n_requests_mature"].is_monotonic_decreasing
    assert coverage["mature_share"].between(0, 1).all()


def test_post_summary_covariance_and_ci():
    beta = np.array([0.01, 0.02, 0.03])
    w = np.array([10.0, 20.0, 30.0])
    a = w / w.sum()
    assert pytest.approx(a.sum()) == 1.0

    cov = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
    estimate = float(a @ beta)
    variance = float(a @ cov @ a)
    se = float(np.sqrt(variance))
    ci_low = estimate - 1.96 * se
    ci_high = estimate + 1.96 * se

    assert se > 0
    assert pytest.approx(ci_low) == estimate - 1.96 * se
    assert pytest.approx(ci_high) == estimate + 1.96 * se
    # Not equal to mean of individual SEs
    mean_se = np.mean(np.sqrt(np.diag(cov)))
    assert abs(se - mean_se) > 1e-6


def test_no_silent_drop_duplicates_in_get_cohort_map_source():
    src = (REPO / "last_mile" / "filter.py").read_text(encoding="utf-8")
    # get_cohort_map must not call drop_duplicates on hex
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_cohort_map":
            body = ast.unparse(node)
            assert "drop_duplicates" not in body


def test_harmonized_notebook_does_not_estimate_panelols():
    nb_path = REPO / "notebooks" / "harmonized_empirical_figures.ipynb"
    if not nb_path.exists():
        pytest.skip("harmonized notebook missing")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")
    assert "PanelOLS(" not in code
    assert "from linearmodels" not in code
    assert "import linearmodels" not in code
    md0 = "".join(nb["cells"][0].get("source", [])) if nb["cells"] else ""
    assert "не является источником эконометрических результатов" in md0


def test_cohort_diagnostics_keys_and_meet_exclusion():
    cohort_map = pd.DataFrame(
        {
            "hex": ["h1", "h2", "h3"],
            "cohort": pd.to_datetime(["2022-07-27", "2022-08-12", "2022-08-12"]),
            "change_type": ["region_and_workmode", "region_only", "workmode_only"],
            "is_treated": [True, True, True],
        }
    )
    report = cohort_composition_report(cohort_map)
    assert "composition_meet" in report
    assert "composition_success" in report
    assert "composition_utilization" in report
    assert "composition_post_meeting" in report
    # 2022-07-27 present in sch, absent in post-meeting
    assert EXCLUDED_COHORT_FOR_POST_MEETING in report["composition_sch"].index
    assert EXCLUDED_COHORT_FOR_POST_MEETING not in report["composition_post_meeting"].index


def test_latex_forbidden_phrases():
    tex = (REPO / "thesis" / "main.tex").read_text(encoding="utf-8")
    assert "гипотеза подтверждается частично" not in tex
    assert r"\(p=0.008541\), при округлении \(p=0.009\)" in tex
    assert r"\(p=0.004515\), при округлении \(p=0.005\)" in tex
    # Old utilization point estimates from obsolete specification
    for old in ["+2.36", "-1.10", "+1.19"]:
        # Allow if in a different context; flag if near utlz
        if old in tex and "utlz" in tex[max(0, tex.find(old) - 80) : tex.find(old) + 80]:
            pytest.fail(f"Found obsolete utilization estimate {old} near utlz context")
