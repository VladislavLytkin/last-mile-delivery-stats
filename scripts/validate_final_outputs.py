# -*- coding: utf-8 -*-
"""Validate canonical final empirical CSV outputs. Exit nonzero on failure."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final"
DIAGNOSTICS = ROOT / "outputs" / "diagnostics"
FIGURES = ROOT / "figures" / "empirical"
RECALCULATION_STARTED = os.environ.get("LAST_MILE_RECALCULATION_STARTED")

OUTCOMES = ["sch_flg", "meet_flg", "success_within_20", "utlz_within_25"]
CHANGE_TYPES = ["region_and_workmode", "region_only", "workmode_only"]

# Exact ordered \includegraphics{...} set from thesis/main.tex.
# Keep this tuple in sync with the thesis includegraphics sequence.
PUBLICATION_FIGURES = (
    "hex_activity_event_time_active_share.pdf",
    "cohort_composition_core_shares.pdf",
    "pretrends_sch_by_cohort.pdf",
    "pretrends_success_by_cohort.pdf",
    "event_study_t_available_by_type.pdf",
    "event_study_sch_by_type.pdf",
    "event_study_meet_by_type.pdf",
    "event_study_success_by_type.pdf",
    "threshold_tradeoff_publication.pdf",
    "event_study_utlz_by_type.pdf",
    "did_threshold_sensitivity_natural_vs_common.pdf",
    "cumulative_success_curves_ru.pdf",
    "multi_horizon_did_common_support_ru.pdf",
    "dose_coefficients_conversions.pdf",
    "dose_coefficients_speed.pdf",
    "dose_event_study_meet_flg.pdf",
    "dose_event_study_t_available.pdf",
    "workmode_delta_support.pdf",
    "applications_event_time.pdf",
    "active_share_event_time.pdf",
    "closure_heatmap.pdf",
    "closure_timing_hist.pdf",
)

REQUIRED_MAIN = [
    "outcome",
    "change_type",
    "estimate",
    "std_error",
    "ci_lower",
    "ci_upper",
    "p_value",
    "pretrend_p_value",
    "n_obs",
    "n_hexagons",
    "n_treated_hexagons",
    "n_treated_requests",
    "weighting_rule",
    "excluded_cohorts",
    "horizon_days",
]

FORBIDDEN_PENDING = {"pending", "todo", "nan", "none"}


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        print("ERROR:", msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print("WARN:", msg)

    def check_file(self, name: str) -> Path | None:
        path = FINAL / name
        if not path.exists():
            self.err(f"missing file: {path}")
            return None
        if path.stat().st_size == 0:
            self.err(f"empty file: {path}")
            return None
        if RECALCULATION_STARTED:
            try:
                start_ts = pd.Timestamp(RECALCULATION_STARTED)
                mtime = pd.Timestamp(path.stat().st_mtime, unit="s")
                if start_ts.tzinfo is not None:
                    mtime = mtime.tz_localize("UTC").tz_convert(start_ts.tzinfo)
                if mtime < start_ts:
                    self.err(
                        f"{name} mtime {mtime} is before recalculation start {start_ts}"
                    )
            except Exception as exc:  # noqa: BLE001
                self.warn(f"could not compare mtime for {name}: {type(exc).__name__}")
        return path


def _ci_ok(est: float, se: float, lo: float, hi: float, tol: float = 0.05) -> bool:
    if not np.isfinite([est, se, lo, hi]).all():
        return False
    expected_lo = est - 1.96 * se
    expected_hi = est + 1.96 * se
    return abs(expected_lo - lo) <= max(tol * abs(se), 1e-6) and abs(
        expected_hi - hi
    ) <= max(tol * abs(se), 1e-6)


def validate_canonical_csv_hygiene(v: Validator) -> None:
    """Validate release-safe structure shared by every canonical tracked CSV."""

    paths = sorted(FINAL.glob("*.csv"))
    if not paths:
        v.err(f"no canonical CSV files found in {FINAL}")
        return

    row_level_columns = {
        "application_id",
        "request_id",
        "client_id",
        "customer_id",
        "user_id",
        "phone",
        "email",
        "address",
        "hex",
    }
    for path in paths:
        if path.stat().st_size == 0:
            v.err(f"empty canonical CSV: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh))
        except (OSError, StopIteration, csv.Error) as exc:
            v.err(f"could not read CSV header for {path.name}: {type(exc).__name__}")
            continue

        normalized = [column.strip() for column in header]
        if any(not column for column in normalized):
            v.err(f"{path.name}: blank column name in header")
        if len(set(normalized)) != len(normalized):
            v.err(f"{path.name}: duplicate column names in header")
        if any(column.lower().startswith("unnamed:") for column in normalized):
            v.err(f"{path.name}: random index column in header")
        if row_level_columns.intersection(column.lower() for column in normalized):
            found = sorted(row_level_columns.intersection(column.lower() for column in normalized))
            v.err(f"{path.name}: possible row-level identifier columns: {found}")

        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            v.err(f"could not parse canonical CSV {path.name}: {type(exc).__name__}")
            continue
        if frame.columns.duplicated().any():
            v.err(f"{path.name}: parsed columns are not unique")
        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            v.err(f"{path.name}: infinite numeric value")


def validate_main(v: Validator, path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    for col in REQUIRED_MAIN:
        if col not in df.columns:
            v.err(f"main_event_study_post_summary missing column: {col}")
    if v.errors:
        return None

    for outcome in OUTCOMES:
        for ct in CHANGE_TYPES:
            if not ((df["outcome"] == outcome) & (df["change_type"] == ct)).any():
                v.err(f"main missing combination: {outcome} x {ct}")

    # forbid stale utilization estimates
    for bad in (2.36, -1.10, 1.19):
        if "estimate_pp" in df.columns and (np.isclose(df["estimate_pp"], bad)).any():
            v.err(f"main contains obsolete utilization-like estimate_pp={bad}")

    for i, row in df.iterrows():
        label = f"main[{row.get('outcome')}/{row.get('change_type')}]"
        for col in [
            "estimate",
            "std_error",
            "ci_lower",
            "ci_upper",
            "p_value",
            "pretrend_p_value",
            "n_obs",
            "n_hexagons",
            "n_treated_hexagons",
            "n_treated_requests",
        ]:
            val = row[col]
            if pd.isna(val):
                v.err(f"{label}: {col} is NaN")
            elif isinstance(val, str) and val.strip().lower() in FORBIDDEN_PENDING:
                v.err(f"{label}: {col} has pending/TODO value")

        if pd.notna(row["std_error"]) and float(row["std_error"]) < 0:
            v.err(f"{label}: std_error < 0")
        if pd.notna(row["p_value"]) and not (0 <= float(row["p_value"]) <= 1):
            v.err(f"{label}: p_value out of [0,1]")
        if pd.notna(row["pretrend_p_value"]) and not (
            0 <= float(row["pretrend_p_value"]) <= 1
        ):
            v.err(f"{label}: pretrend_p_value out of [0,1]")
        for ncol in ["n_obs", "n_hexagons", "n_treated_hexagons"]:
            if pd.notna(row[ncol]) and float(row[ncol]) <= 0:
                v.err(f"{label}: {ncol} <= 0")

        if all(
            pd.notna(row[c])
            for c in ["estimate", "std_error", "ci_lower", "ci_upper"]
        ):
            est, se, lo, hi = map(
                float, [row["estimate"], row["std_error"], row["ci_lower"], row["ci_upper"]]
            )
            if not (lo <= est <= hi):
                v.err(f"{label}: estimate not inside CI")
            # CI may be stored in levels or pp; prefer matching units via estimate_pp if present
            if "estimate_pp" in df.columns and pd.notna(row.get("std_error_pp")):
                est_pp = float(row["estimate_pp"])
                se_pp = float(row["std_error_pp"])
                lo_pp = float(row.get("ci_lower_pp", row["ci_lower"]))
                hi_pp = float(row.get("ci_upper_pp", row["ci_upper"]))
                if not _ci_ok(est_pp, se_pp, lo_pp, hi_pp):
                    if not _ci_ok(est, se, lo, hi):
                        v.err(f"{label}: CI inconsistent with estimate±1.96*SE")
            elif not _ci_ok(est, se, lo, hi):
                v.err(f"{label}: CI inconsistent with estimate±1.96*SE")

        if row["outcome"] == "utlz_within_25":
            if float(row["horizon_days"]) != 25:
                v.err(f"{label}: horizon_days must be 25")

        excl = str(row.get("excluded_cohorts", ""))
        if row["outcome"] in {"meet_flg", "success_within_20", "utlz_within_25"}:
            if "2022-07-27" not in excl:
                v.err(f"{label}: excluded_cohorts must include 2022-07-27")
        if row["outcome"] == "sch_flg":
            if "2022-07-27" in excl:
                v.err(f"{label}: sch_flg should not exclude 2022-07-27 by outcome rule")
        if row["outcome"] == "success_within_20" and float(row["horizon_days"]) != 20:
            v.err(f"{label}: horizon_days must be 20")

    return df


def validate_conditional(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    outcome_col = "outcome" if "outcome" in df.columns else (
        "outcome_key" if "outcome_key" in df.columns else (
            "transition" if "transition" in df.columns else None
        )
    )
    if outcome_col is None:
        v.err("conditional_funnel_summary: no outcome/transition column")
        return

    # normalize SE/CI column names
    se_col = next((c for c in df.columns if c in {"std_error", "std_error_pp", "se", "se_pp"}), None)
    lo_col = next((c for c in df.columns if c in {"ci_lower", "ci_lower_pp", "lower", "lower_pp"}), None)
    hi_col = next((c for c in df.columns if c in {"ci_upper", "ci_upper_pp", "upper", "upper_pp"}), None)
    est_col = next(
        (c for c in df.columns if c in {"estimate", "estimate_pp", "avg_post_effect_pp"}),
        None,
    )
    p_col = next((c for c in df.columns if c in {"p_value", "pval"}), None)

    for need, col in [
        ("estimate", est_col),
        ("std_error", se_col),
        ("ci_lower", lo_col),
        ("ci_upper", hi_col),
        ("p_value", p_col),
    ]:
        if col is None:
            v.err(f"conditional_funnel_summary missing {need} column")

    if any(c is None for c in [est_col, se_col, lo_col, hi_col, p_col]):
        return

    text_blob = df.astype(str).to_string().lower()
    if "pending" in text_blob or "todo" in text_blob:
        v.err("conditional_funnel_summary still contains pending/TODO")

    for i, row in df.iterrows():
        label = f"cond[{row.get(outcome_col)}/{row.get('change_type')}]"
        vals = [row[est_col], row[se_col], row[lo_col], row[hi_col], row[p_col]]
        if any(pd.isna(x) for x in vals):
            # allow only if status/note says unidentified
            note = str(row.get("status", "")) + str(row.get("se_status", ""))
            if "unidentif" in note.lower() or "skip" in note.lower():
                v.warn(f"{label}: NaN SE/CI explained as unidentified")
            else:
                v.err(f"{label}: SE/CI/p NaN for identified specification")
            continue
        est, se, lo, hi, p = map(float, vals)
        if se < 0:
            v.err(f"{label}: std_error < 0")
        if not (lo <= est <= hi):
            v.err(f"{label}: estimate not inside CI")
        if not (0 <= p <= 1):
            v.err(f"{label}: p_value out of [0,1]")
        if not _ci_ok(est, se, lo, hi):
            v.err(f"{label}: CI inconsistent with estimate±1.96*SE")

        for ncol in ["n_obs", "n_hexagons", "n_treated_hexagons"]:
            if ncol in df.columns:
                if pd.isna(row[ncol]) or float(row[ncol]) <= 0:
                    v.err(f"{label}: {ncol} missing or <=0")


def validate_attgt(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        v.err("attgt_robustness_summary is empty")
        return
    for outcome in OUTCOMES:
        for ct in CHANGE_TYPES:
            if not ((df["outcome"] == outcome) & (df["change_type"] == ct)).any():
                v.err(f"attgt missing combination: {outcome} x {ct}")


def validate_speed(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        v.err("speed_did_post_summary is empty")
        return
    for ct in CHANGE_TYPES:
        if "change_type" in df.columns and not (df["change_type"] == ct).any():
            v.err(f"speed missing change_type {ct}")
    required = {
        "estimate_days",
        "std_error_days",
        "ci_lower_days",
        "ci_upper_days",
    }
    for col in sorted(required - set(df.columns)):
        v.err(f"speed_did_post_summary missing column: {col}")
    if required - set(df.columns):
        return
    optional_hours = {
        "estimate_hours",
        "std_error_hours",
        "ci_lower_hours",
        "ci_upper_hours",
    }
    if optional_hours <= set(df.columns):
        for _, row in df.iterrows():
            label = f"speed[{row['change_type']}]"
            for base, hours in [
                ("estimate_days", "estimate_hours"),
                ("std_error_days", "std_error_hours"),
                ("ci_lower_days", "ci_lower_hours"),
                ("ci_upper_days", "ci_upper_hours"),
            ]:
                if not np.isclose(float(row[hours]), 24.0 * float(row[base])):
                    v.err(f"{label}: {hours} is not 24*{base}")
    for _, row in df.iterrows():
        label = f"speed[{row['change_type']}]"
        if not _ci_ok(
            float(row["estimate_days"]),
            float(row["std_error_days"]),
            float(row["ci_lower_days"]),
            float(row["ci_upper_days"]),
        ):
            v.err(f"{label}: day CI inconsistent with estimate±1.96*SE")
        if "cluster_level" in df.columns and row["cluster_level"] != "hex":
            v.err(f"{label}: cluster_level must be hex")


def validate_assoc(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        v.err("speed_conversion_association_summary is empty")
        return
    required = {"section", "metric", "estimate", "n_obs", "note"}
    for col in sorted(required - set(df.columns)):
        v.err(f"speed_conversion_association_summary missing column: {col}")
    if required - set(df.columns):
        return
    if not (df["section"] == "fe_lpm").any():
        v.warn("association summary may lack fe_lpm section")

    legacy = (df["metric"] == "n_total_requests") | (
        df["note"].fillna("") == "all loaded requests"
    )
    if legacy.any():
        v.err("association summary contains forbidden legacy total-request label")

    metric = "n_analysis_requests_before_t_available_filter"
    analysis_rows = df[df["metric"] == metric]
    if len(analysis_rows) != 1:
        v.err(f"association summary must contain exactly one {metric} row")
    else:
        row = analysis_rows.iloc[0]
        if not np.isclose(float(row["estimate"]), 1_807_317):
            v.err(f"{metric} estimate must equal 1807317")
        if int(row["n_obs"]) != 1_807_317:
            v.err(f"{metric} n_obs must equal 1807317")
        expected_note = (
            "combined treated/control sample from build_did_samples; "
            "not the full raw dataset"
        )
        if row["note"] != expected_note:
            v.err(f"{metric} note is incorrect")

    funnel_path = FINAL / "conversion_funnel_summary.csv"
    if not funnel_path.exists():
        v.err(f"missing file: {funnel_path}")
        return
    funnel = pd.read_csv(funnel_path)
    if not {"scope", "requests"}.issubset(funnel.columns):
        v.err("conversion_funnel_summary must contain scope and requests")
        return
    full_rows = funnel[funnel["scope"] == "all"]
    if len(full_rows) != 1 or int(full_rows.iloc[0]["requests"]) != 1_807_426:
        v.err("conversion_funnel_summary all-scope requests must equal 1807426")
    if not analysis_rows.empty and np.isclose(
        float(analysis_rows.iloc[0]["estimate"]), 1_807_426
    ):
        v.err("association analytical sample must not be labeled as the full raw dataset")


def validate_assoc_fe(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    required = {
        "outcome",
        "estimate_pp_per_day",
        "std_error_pp_per_day",
        "ci_lower_pp_per_day",
        "ci_upper_pp_per_day",
        "p_value",
        "n_obs",
        "n_hexagons",
        "estimator",
        "cluster_level",
        "unit",
    }
    for col in sorted(required - set(df.columns)):
        v.err(f"speed_conversion_fe_summary missing column: {col}")
    if required - set(df.columns):
        return
    if set(df["outcome"]) != set(OUTCOMES):
        v.err("speed_conversion_fe_summary must contain exactly four outcomes")
    for _, row in df.iterrows():
        label = f"assoc[{row['outcome']}]"
        if not _ci_ok(
            float(row["estimate_pp_per_day"]),
            float(row["std_error_pp_per_day"]),
            float(row["ci_lower_pp_per_day"]),
            float(row["ci_upper_pp_per_day"]),
        ):
            v.err(f"{label}: CI inconsistent with estimate±1.96*SE")
        if not (0 <= float(row["p_value"]) <= 1):
            v.err(f"{label}: p_value out of [0,1]")
        if int(row["n_obs"]) <= 0 or int(row["n_hexagons"]) <= 0:
            v.err(f"{label}: N or n_hexagons is not positive")
        if row["cluster_level"] != "hex":
            v.err(f"{label}: cluster_level must be hex")


def validate_event_coefficients(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    required = {
        "outcome",
        "change_type",
        "event_week",
        "coef",
        "se",
        "ci_low",
        "ci_high",
        "estimated",
        "baseline_week",
        "ci_level",
        "cluster_level",
    }
    for col in sorted(required - set(df.columns)):
        v.err(f"event_study_coefficients missing column: {col}")
    if required - set(df.columns):
        return
    expected_outcomes = set(OUTCOMES + ["t_available"])
    if set(df["outcome"]) != expected_outcomes:
        v.err("event_study_coefficients has an unexpected outcome set")
    for outcome in expected_outcomes:
        for ct in CHANGE_TYPES:
            if not ((df["outcome"] == outcome) & (df["change_type"] == ct)).any():
                v.err(f"event coefficients missing combination: {outcome} x {ct}")
    estimated = df[df["estimated"].astype(bool)].copy()
    for i, row in estimated.iterrows():
        if not _ci_ok(
            float(row["coef"]),
            float(row["se"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
        ):
            v.err(f"event coefficients row {i}: CI inconsistent with estimate±1.96*SE")
    if not (df["baseline_week"] == -1).all():
        v.err("event_study_coefficients baseline_week must be -1")
    if not np.allclose(df["ci_level"], 0.95):
        v.err("event_study_coefficients ci_level must be 0.95")
    if not (df["cluster_level"] == "hex").all():
        v.err("event_study_coefficients cluster_level must be hex")


def validate_post_weights(v: Validator, path: Path) -> None:
    df = pd.read_csv(path)
    required = {
        "outcome",
        "change_type",
        "event_week",
        "weight",
        "event_week_estimate",
        "post_estimate",
        "post_std_error",
        "post_ci_lower",
        "post_ci_upper",
        "se_method",
        "cluster_level",
    }
    for col in sorted(required - set(df.columns)):
        v.err(f"event_study_post_weights missing column: {col}")
    if required - set(df.columns):
        return
    groups = df.groupby(["outcome", "change_type"], sort=False)
    if len(groups) != 15:
        v.err(f"event_study_post_weights expected 15 specifications, found {len(groups)}")
    for key, part in groups:
        if not np.isclose(part["weight"].sum(), 1.0):
            v.err(f"post weights {key}: weights do not sum to 1")
        reconstructed = float((part["weight"] * part["event_week_estimate"]).sum())
        estimate = float(part["post_estimate"].iloc[0])
        se = float(part["post_std_error"].iloc[0])
        lo = float(part["post_ci_lower"].iloc[0])
        hi = float(part["post_ci_upper"].iloc[0])
        if not np.isclose(reconstructed, estimate, atol=1e-12):
            v.err(f"post weights {key}: weighted weeks do not reconstruct estimate")
        if not _ci_ok(estimate, se, lo, hi):
            v.err(f"post weights {key}: CI inconsistent with estimate±1.96*SE")
        if not (part["se_method"] == "a'Cov(theta)a").all():
            v.err(f"post weights {key}: unexpected SE method")
        if not (part["cluster_level"] == "hex").all():
            v.err(f"post weights {key}: cluster_level must be hex")


def validate_treatment_resolution(v: Validator) -> None:
    detail_path = DIAGNOSTICS / "multi_treatment_hexagon_summary.csv"
    conflicts_path = DIAGNOSTICS / "same_date_metadata_conflicts.csv"
    resolution_path = DIAGNOSTICS / "treatment_resolution_summary.csv"
    for path in (detail_path, conflicts_path, resolution_path):
        if not path.exists():
            v.err(f"missing treatment diagnostic: {path}")
    if not all(p.exists() for p in (detail_path, conflicts_path, resolution_path)):
        return

    detail = pd.read_csv(detail_path)
    if detail["hex"].duplicated().any():
        v.err("treatment diagnostics are not unique by hex")
    n_multi = int(detail["policy"].eq("exclude_multi_treatment").sum())
    if n_multi != 15_832:
        v.err(f"expected 15,832 multi-date hexagons, found {n_multi:,}")
    conflicts = pd.read_csv(conflicts_path)
    if not conflicts.empty:
        v.err(f"same-date treatment metadata conflicts found: {len(conflicts):,} rows")
    resolution = pd.read_csv(resolution_path)
    if resolution["n_canonical_hexagons"].nunique() != 1:
        v.err("treatment resolution summary has inconsistent canonical denominator")


def validate_dose_outputs(v: Validator) -> None:
    dose_path = v.check_file("dose_did_summary.csv")
    if dose_path:
        dose = pd.read_csv(dose_path)
        required = {
            "outcome",
            "term",
            "estimate",
            "std_error",
            "p_value",
            "q_value",
            "cohort_specification",
        }
        for col in sorted(required - set(dose.columns)):
            v.err(f"dose_did_summary missing column: {col}")
        if not required - set(dose.columns):
            conversion = dose[
                dose["outcome"].isin(
                    ["sch_flg", "meet_flg", "success_within_20", "utlz_within_25"]
                )
            ]
            if not conversion["q_value"].between(0, 1).all():
                v.err("dose q-values must be in [0,1]")
            if set(dose["cohort_specification"]) != {
                "outcome_specific",
                "inclusive_2022-07-27",
            }:
                v.err("dose output must contain primary and inclusive-cohort variants")

    support_path = v.check_file("workmode_delta_support.csv")
    if support_path:
        support = pd.read_csv(support_path)
        required = {"delta_w_h", "change_type", "cohort", "n_hexagons", "denominator"}
        for col in sorted(required - set(support.columns)):
            v.err(f"workmode_delta_support missing column: {col}")
        if not required - set(support.columns):
            cohorts = support["cohort"].astype(str)
            if cohorts.str.contains("NaT|2022-10-19", na=False).any():
                v.err("workmode support contains NaT or excluded 2022-10-19 cohort")
            if (support["delta_w_h"] == 0).any():
                v.err("workmode support contains zero dose")
            if (support["n_hexagons"] <= 0).any():
                v.err("workmode support contains non-positive cell counts")
            if support["denominator"].nunique() != 1:
                v.err("workmode support has inconsistent denominators")
            elif int(support["n_hexagons"].sum()) != int(support["denominator"].iloc[0]):
                v.err("workmode support cells do not reconstruct denominator")

    for name, required in {
        "dose_outcome_support.csv": {
            "outcome",
            "cohort_specification",
            "n_treated_hexagons",
            "n_nonzero_dose_hexagons",
        },
        "dose_nonlinearity_summary.csv": {
            "outcome",
            "dose_bin",
            "estimate",
            "std_error",
            "p_value",
        },
        "sch_robustness_summary.csv": {
            "change_type",
            "specification",
            "estimate",
            "std_error",
        },
        "success_horizon_sensitivity.csv": {
            "horizon_days",
            "change_type",
            "estimate",
            "std_error",
            "n_treated_hexagons",
        },
        "success_horizon_coverage.csv": {
            "horizon_days",
            "n_requests_total",
            "n_requests_mature",
            "mature_share",
        },
    }.items():
        path = v.check_file(name)
        if not path:
            continue
        frame = pd.read_csv(path)
        for col in sorted(required - set(frame.columns)):
            v.err(f"{name} missing column: {col}")
        if name == "success_horizon_coverage.csv" and not required - set(frame.columns):
            if 20 not in set(frame["horizon_days"]):
                v.err("success horizon coverage must include the primary 20-day estimand")
            if not frame["mature_share"].between(0, 1).all():
                v.err("success horizon mature_share must be in [0,1]")


def validate_sample_flow(v: Validator) -> None:
    path = v.check_file("sample_sizes_by_outcome.csv")
    if not path:
        return
    frame = pd.read_csv(path)
    if frame.duplicated(["outcome", "change_type"]).any():
        v.err("sample_sizes_by_outcome has duplicate outcome × change_type rows")
    if "success_within_20" not in set(frame["outcome"]):
        v.err("sample flow does not contain canonical success_within_20 outcome")
    if "success_flg" in set(frame["outcome"]):
        v.err("sample flow mixes lifetime success_flg into the main estimands")


def _tracked_files() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git ls-files failed")
    return {line.strip().replace("\\", "/") for line in completed.stdout.splitlines()}


def validate_publication_artifacts(v: Validator) -> None:
    tex_path = ROOT / "thesis" / "main.tex"
    bib_path = ROOT / "thesis" / "references.bib"
    pdf_path = ROOT / "thesis" / "main.pdf"

    for path in (tex_path, bib_path, pdf_path):
        if not path.exists():
            v.err(f"missing publication artifact: {path}")
    if not tex_path.exists():
        return

    tex = tex_path.read_text(encoding="utf-8")
    if r"13\,256" in tex:
        v.err("thesis contains obsolete panel denominator 13,256")
    if r"9\,328" in tex:
        v.err("thesis contains obsolete treated-hex denominator 9,328")
    for current in (r"13\,255", r"9\,327", r"302\,398"):
        if current not in tex:
            v.err(f"thesis does not contain current publication number: {current}")
    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", tex, flags=re.IGNORECASE):
        v.err("thesis contains a personal email address")
    if "причинная альтернативная гипотеза не принимается" in tex:
        v.err("thesis contains obsolete causal-hypothesis wording")

    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", tex)
    if tuple(names) != PUBLICATION_FIGURES:
        v.err(
            "thesis includegraphics manifest must match PUBLICATION_FIGURES "
            f"(expected {len(PUBLICATION_FIGURES)}, found {len(names)})"
        )
    for name in names:
        candidate = FIGURES / name
        if not candidate.is_file():
            v.err(f"thesis figure is missing: {name}")

    if pdf_path.exists():
        if pdf_path.stat().st_size == 0 or pdf_path.read_bytes()[:5] != b"%PDF-":
            v.err("thesis/main.pdf is empty or lacks a PDF magic header")

    try:
        tracked = _tracked_files()
    except (OSError, RuntimeError) as exc:
        v.err(f"could not inspect tracked publication files: {exc}")
        return
    required_tracked = {
        f"figures/empirical/{name}" for name in PUBLICATION_FIGURES
    } | {"thesis/main.pdf"}
    missing_tracked = sorted(required_tracked - tracked)
    if missing_tracked:
        v.err(f"publication artifacts are not tracked: {missing_tracked}")
    tracked_empirical = {
        name for name in tracked if name.startswith("figures/empirical/")
    }
    tracked_png = sorted(name for name in tracked_empirical if name.lower().endswith(".png"))
    if tracked_png:
        v.err(f"tracked publication figures contain PNG files: {tracked_png}")
    extra_pdfs = sorted(
        name
        for name in tracked_empirical
        if name.lower().endswith(".pdf") and name not in required_tracked
    )
    if extra_pdfs:
        v.err(f"tracked publication figures contain unexpected PDFs: {extra_pdfs}")


def publication_main() -> int:
    v = Validator()
    validate_publication_artifacts(v)
    print("\n=== PUBLICATION ARTIFACT SUMMARY ===")
    print(f"errors={len(v.errors)} warnings={len(v.warnings)}")
    if v.errors:
        return 1
    print("OK")
    return 0


def main(*, full: bool = False) -> int:
    v = Validator()
    print(f"Validation mode: {'full' if full else 'release'}")
    validate_canonical_csv_hygiene(v)
    main_path = v.check_file("main_event_study_post_summary.csv")
    if main_path:
        validate_main(v, main_path)

    cond_path = v.check_file("conditional_funnel_summary.csv")
    if cond_path:
        validate_conditional(v, cond_path)

    att_path = v.check_file("attgt_robustness_summary.csv")
    if att_path:
        validate_attgt(v, att_path)

    speed_path = v.check_file("speed_did_post_summary.csv")
    if speed_path:
        validate_speed(v, speed_path)

    assoc_path = v.check_file("speed_conversion_association_summary.csv")
    if assoc_path:
        validate_assoc(v, assoc_path)

    assoc_fe_path = v.check_file("speed_conversion_fe_summary.csv")
    if assoc_fe_path:
        validate_assoc_fe(v, assoc_fe_path)

    coef_path = v.check_file("event_study_coefficients.csv")
    if coef_path:
        validate_event_coefficients(v, coef_path)

    weights_path = v.check_file("event_study_post_weights.csv")
    if weights_path:
        validate_post_weights(v, weights_path)

    if full:
        validate_treatment_resolution(v)
    validate_dose_outputs(v)
    validate_sample_flow(v)
    validate_publication_artifacts(v)

    print("\n=== SUMMARY ===")
    print(f"errors={len(v.errors)} warnings={len(v.warnings)}")
    if v.errors:
        if full:
            print(
                "Full validation requires generated diagnostics and thesis figures; "
                "run the full analysis pipeline before retrying with --full."
            )
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also require generated diagnostics and thesis figures from the full pipeline",
    )
    parser.add_argument(
        "--publication-only",
        action="store_true",
        help="validate only the tracked thesis PDF and publication figures",
    )
    args = parser.parse_args()
    if args.publication_only:
        raise SystemExit(publication_main())
    raise SystemExit(main(full=args.full))
