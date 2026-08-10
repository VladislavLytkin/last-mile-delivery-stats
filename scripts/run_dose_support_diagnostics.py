"""
Легкая диагностика support дозовой модели без переоценки DiD/event-study.

Повторяет определения R_h, Δw_h, W_h^+, W_h^- и фильтры выборки из
notebooks/final_empirical_recalculation.ipynb (dose block), затем сохраняет
таблицы support в outputs/final/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from last_mile.filter import CORE_CHANGE_TYPES, build_analysis_panel  # noqa: E402
from last_mile.outcomes import success_horizon_coverage  # noqa: E402

TYPE_ORDER = ["region_and_workmode", "region_only", "workmode_only"]
EXCLUDED_COHORT_FOR_CONVERSION = pd.Timestamp("2022-07-27")
UTILIZATION_OBSERVATION_END = pd.Timestamp("2022-10-18")
SUCCESS_OBSERVATION_END = pd.Timestamp("2022-10-18")
UTLZ_HORIZON_DAYS = 25
UTLZ_OUTCOME = "utlz_within_25"
SUCCESS_OUTCOME = "success_within_20"
SUCCESS_HORIZONS = (7, 14, 20, 25, 30)
CONVERSION_OUTCOMES_EXCLUDE_0727 = {
    "meet_flg",
    "success_within_20",
    "utlz_within_25",
    "utlz_flg",
}
DOSE_OUTCOMES = [
    "sch_flg",
    "meet_flg",
    "success_within_20",
    "utlz_within_25",
    "t_available",
]
TIME_OUTCOMES = {"t_available", "t_utilization"}
HEX_WEIGHT_COLS = [
    "hex",
    "subject",
    "treatment_date",
    "region_id_old",
    "region_id_new",
    "work_mode_old",
    "work_mode_new",
    "change_type",
]


def outcome_agg_func(outcome: str) -> str:
    return "median" if outcome in TIME_OUTCOMES else "mean"


def maturity_horizon_days(outcome: str) -> int:
    if outcome.startswith("success_within_"):
        return int(outcome.rsplit("_", 1)[1])
    if outcome in {"utlz_within_25", "utlz_flg"}:
        return int(UTLZ_HORIZON_DAYS)
    return 0


def filter_mature_requests(
    df: pd.DataFrame, outcome: str, *, obs_max: pd.Timestamp
) -> pd.DataFrame:
    h = maturity_horizon_days(outcome)
    if h <= 0:
        return df.copy()
    if outcome.startswith("success_within_"):
        obs_end = SUCCESS_OBSERVATION_END
    elif outcome in {"utlz_within_25", "utlz_flg"}:
        obs_end = UTILIZATION_OBSERVATION_END
    else:
        obs_end = obs_max
    return df[df["request_timestamp"] + pd.Timedelta(days=h) <= obs_end].copy()


def prepare_utlz_orders(df: pd.DataFrame) -> pd.DataFrame:
    neg = df["real_utilization_dttm"].notna() & (
        df["real_utilization_dttm"] < df["request_timestamp"]
    )
    out = df.loc[~neg].copy()
    horizon = out["request_timestamp"] + pd.Timedelta(days=UTLZ_HORIZON_DAYS)
    out[UTLZ_OUTCOME] = (
        out["real_utilization_dttm"].notna()
        & (out["real_utilization_dttm"] <= horizon)
    ).astype(float)
    return out


def build_weight_lookup(cohort_map: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    missing = set(HEX_WEIGHT_COLS) - set(cohort_map.columns)
    if missing:
        raise ValueError(f"cohort_map missing dose metadata: {sorted(missing)}")
    final_panel_hexes = set(panel["hex"].unique())
    hex_weights = cohort_map[HEX_WEIGHT_COLS].copy()
    hex_weights["treatment_date"] = pd.to_datetime(
        hex_weights["treatment_date"], errors="coerce"
    )
    hex_weights = hex_weights[
        hex_weights["hex"].isin(final_panel_hexes)
        & hex_weights["change_type"].isin(CORE_CHANGE_TYPES)
        & hex_weights["treatment_date"].notna()
        & (hex_weights["treatment_date"] != pd.Timestamp("2022-10-19"))
    ].copy()
    assert hex_weights["hex"].is_unique

    for col in ["work_mode_old", "work_mode_new"]:
        hex_weights[col] = pd.to_numeric(hex_weights[col], errors="coerce")
    hex_weights["delta_w_h"] = (
        hex_weights["work_mode_new"] - hex_weights["work_mode_old"]
    )
    hex_weights["W_h_plus"] = hex_weights["delta_w_h"].clip(lower=0)
    hex_weights["W_h_minus"] = (-hex_weights["delta_w_h"]).clip(lower=0)
    hex_weights["R_h"] = (
        hex_weights["region_id_old"] != hex_weights["region_id_new"]
    ).astype(int)

    assert (hex_weights["W_h_plus"] >= 0).all()
    assert (hex_weights["W_h_minus"] >= 0).all()
    assert ((hex_weights["W_h_plus"] == 0) | (hex_weights["W_h_minus"] == 0)).all()
    assert np.allclose(
        hex_weights["W_h_plus"] - hex_weights["W_h_minus"], hex_weights["delta_w_h"]
    )
    return hex_weights[
        [
            "hex",
            "treatment_date",
            "change_type",
            "R_h",
            "delta_w_h",
            "W_h_plus",
            "W_h_minus",
        ]
    ].copy()


def attach_weight_columns(df: pd.DataFrame, weight_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "treatment_date" not in out.columns:
        if "cohort" in out.columns:
            out["treatment_date"] = out["cohort"]
        else:
            out["treatment_date"] = pd.NaT
    out = out.merge(
        weight_lookup[
            ["hex", "treatment_date", "R_h", "delta_w_h", "W_h_plus", "W_h_minus"]
        ],
        on=["hex", "treatment_date"],
        how="left",
        validate="many_to_one",
    )
    for col in ["R_h", "delta_w_h", "W_h_plus", "W_h_minus"]:
        out[col] = out[col].fillna(0)
    return out


def prepare_samples(applications_path: Path, hexagons_path: Path) -> dict:
    result = build_analysis_panel(
        applications_path=applications_path,
        hexagons_path=hexagons_path,
        min_orders_per_hex=5,
    )
    panel = result["panel"].copy()
    apps_t = result["treated_orders"].copy()
    apps_c = result["control_orders"].copy()
    cohort_map = result["cohort_map"].copy()

    for df in (apps_t, apps_c):
        if "cohort" in df.columns and "treatment_date" not in df.columns:
            df["treatment_date"] = df["cohort"]
        if "date" not in df.columns:
            df["date"] = df["request_timestamp"].dt.normalize()
    if "days_from_treatment" not in apps_t.columns:
        apps_t["days_from_treatment"] = (
            apps_t["request_timestamp"] - apps_t["treatment_date"]
        ).dt.days

    apps_t_utlz = prepare_utlz_orders(apps_t)
    apps_c_utlz = prepare_utlz_orders(apps_c)
    apps_t, _ = success_horizon_coverage(
        apps_t, horizons=SUCCESS_HORIZONS, observation_end=SUCCESS_OBSERVATION_END
    )
    apps_c, _ = success_horizon_coverage(
        apps_c, horizons=SUCCESS_HORIZONS, observation_end=SUCCESS_OBSERVATION_END
    )

    obs_max = max(apps_t["request_timestamp"].max(), apps_c["request_timestamp"].max())
    weight_lookup = build_weight_lookup(cohort_map, panel)
    apps_t = attach_weight_columns(apps_t, weight_lookup)
    apps_c = attach_weight_columns(apps_c, weight_lookup)
    apps_t_utlz = attach_weight_columns(apps_t_utlz, weight_lookup)
    apps_c_utlz = attach_weight_columns(apps_c_utlz, weight_lookup)

    valid_cohorts_all = {
        pd.Timestamp(x) for x in pd.to_datetime(apps_t["treatment_date"].dropna().unique())
    }
    valid_cohorts_conversion = valid_cohorts_all - {EXCLUDED_COHORT_FOR_CONVERSION}

    def cohorts_for_outcome(outcome: str, *, include_2022_07_27: bool) -> set[pd.Timestamp]:
        if include_2022_07_27:
            return valid_cohorts_all
        if outcome in CONVERSION_OUTCOMES_EXCLUDE_0727 or outcome.startswith(
            "success_within_"
        ):
            return valid_cohorts_conversion
        return valid_cohorts_all

    def make_control_sample(outcome: str) -> pd.DataFrame:
        source = apps_c_utlz if outcome in {"utlz_within_25", "utlz_flg"} else apps_c
        return filter_mature_requests(source.copy(), outcome, obs_max=obs_max)

    def make_dose_treated_orders(
        outcome: str, *, include_2022_07_27: bool = False
    ) -> pd.DataFrame:
        source = apps_t_utlz if outcome in {"utlz_within_25", "utlz_flg"} else apps_t
        cohorts = cohorts_for_outcome(outcome, include_2022_07_27=include_2022_07_27)
        parts = []
        for ctype in TYPE_ORDER:
            part = source[
                (source["change_type"] == ctype)
                & source["treatment_date"].isin(cohorts)
            ].copy()
            part = filter_mature_requests(part, outcome, obs_max=obs_max)
            if not part.empty:
                parts.append(part)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def build_dose_order_frame(
        outcome: str, *, include_2022_07_27: bool = False
    ) -> pd.DataFrame:
        t = make_dose_treated_orders(
            outcome, include_2022_07_27=include_2022_07_27
        )
        if t.empty:
            raise ValueError(f"empty CORE treated sample for {outcome}")
        t = t.copy()
        t["is_treated"] = 1
        t["post"] = (t["days_from_treatment"] >= 0).astype(int)

        c = make_control_sample(outcome).copy()
        c["is_treated"] = 0
        c["post"] = 0
        c["R_h"] = 0
        c["delta_w_h"] = 0
        c["W_h_plus"] = 0
        c["W_h_minus"] = 0
        c["change_type"] = "never_treated"
        c["days_from_treatment"] = np.nan

        d = pd.concat([t, c], ignore_index=True)
        d["date"] = d["request_timestamp"].dt.normalize()
        ctrl = d["is_treated"] == 0
        d.loc[ctrl, ["post", "R_h", "W_h_plus", "W_h_minus", "delta_w_h"]] = 0
        d["post_R"] = d["post"] * d["R_h"]
        d["post_W_plus"] = d["post"] * d["W_h_plus"]
        d["post_W_minus"] = d["post"] * d["W_h_minus"]
        return d.dropna(subset=[outcome]).copy()

    def aggregate_dose_panel(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
        reg_cols = ["post_R", "post_W_plus", "post_W_minus"]
        agg_dict = {outcome: outcome_agg_func(outcome), "request_timestamp": "count"}
        for col in reg_cols:
            agg_dict[col] = "max"
        for col in [
            "is_treated",
            "R_h",
            "W_h_plus",
            "W_h_minus",
            "delta_w_h",
            "change_type",
            "treatment_date",
        ]:
            if col in d.columns:
                agg_dict[col] = "first"
        return (
            d.groupby(["hex", "date"], as_index=False)
            .agg(agg_dict)
            .rename(columns={"request_timestamp": "n_orders"})
        )

    return {
        "weight_lookup": weight_lookup,
        "build_dose_order_frame": build_dose_order_frame,
        "aggregate_dose_panel": aggregate_dose_panel,
        "valid_cohorts_all": valid_cohorts_all,
        "valid_cohorts_conversion": valid_cohorts_conversion,
    }


def support_from_hex_meta(weight_lookup: pd.DataFrame) -> pd.DataFrame:
    """Hexagon-level support including region_only (Δw=0) and never-treated absent."""
    wl = weight_lookup.copy()
    wl["delta_w_h"] = wl["delta_w_h"].astype(int)
    rows = (
        wl.groupby(["R_h", "delta_w_h", "change_type"], as_index=False)
        .agg(n_hexagons=("hex", "nunique"))
        .sort_values(["R_h", "delta_w_h", "change_type"])
    )
    return rows


def support_from_reg_panel(reg_panel: pd.DataFrame, outcome: str, cohort_spec: str) -> pd.DataFrame:
    rp = reg_panel.copy()
    rp["change_type"] = rp["change_type"].fillna("never_treated")
    rp.loc[rp["is_treated"] == 0, "change_type"] = "never_treated"
    rp["delta_w_h"] = rp["delta_w_h"].fillna(0).astype(int)
    rp["R_h"] = rp["R_h"].fillna(0).astype(int)

    hex_meta = (
        rp.groupby("hex", as_index=False)
        .agg(
            R_h=("R_h", "first"),
            delta_w_h=("delta_w_h", "first"),
            change_type=("change_type", "first"),
            is_treated=("is_treated", "max"),
        )
    )
    hex_counts = (
        hex_meta.groupby(["R_h", "delta_w_h", "change_type"], as_index=False)
        .agg(n_hexagons=("hex", "nunique"))
    )
    day_counts = (
        rp.groupby(["R_h", "delta_w_h", "change_type"], as_index=False)
        .agg(n_hex_day=("hex", "size"))
    )
    out = hex_counts.merge(
        day_counts, on=["R_h", "delta_w_h", "change_type"], how="outer"
    ).fillna(0)
    out["n_hexagons"] = out["n_hexagons"].astype(int)
    out["n_hex_day"] = out["n_hex_day"].astype(int)
    out["outcome"] = outcome
    out["cohort_specification"] = cohort_spec
    return out.sort_values(["outcome", "R_h", "delta_w_h", "change_type"])


def overlap_report(support: pd.DataFrame) -> pd.DataFrame:
    """Overlap of Δw across R_h within treated cells of estimation sample."""
    treated = support[support["change_type"] != "never_treated"].copy()
    rows = []
    for (outcome, cohort_spec), grp in treated.groupby(
        ["outcome", "cohort_specification"]
    ):
        for dw in sorted(grp["delta_w_h"].unique()):
            sub = grp[grp["delta_w_h"] == dw]
            n0 = int(sub.loc[sub["R_h"] == 0, "n_hexagons"].sum())
            n1 = int(sub.loc[sub["R_h"] == 1, "n_hexagons"].sum())
            d0 = int(sub.loc[sub["R_h"] == 0, "n_hex_day"].sum())
            d1 = int(sub.loc[sub["R_h"] == 1, "n_hex_day"].sum())
            rows.append(
                {
                    "outcome": outcome,
                    "cohort_specification": cohort_spec,
                    "delta_w_h": int(dw),
                    "n_hex_R0": n0,
                    "n_hex_R1": n1,
                    "n_hex_day_R0": d0,
                    "n_hex_day_R1": d1,
                    "has_overlap_hex": bool(n0 > 0 and n1 > 0),
                    "rare_cell": bool(min(n0, n1) > 0 and min(n0, n1) < 30),
                }
            )
    return pd.DataFrame(rows)


def change_type_delta_summary(support: pd.DataFrame) -> pd.DataFrame:
    treated = support[support["change_type"] != "never_treated"].copy()
    return (
        treated.groupby(
            ["outcome", "cohort_specification", "change_type", "delta_w_h"],
            as_index=False,
        )
        .agg(n_hexagons=("n_hexagons", "sum"), n_hex_day=("n_hex_day", "sum"))
        .sort_values(["outcome", "change_type", "delta_w_h"])
    )


def identification_notes(overlap: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    notes = []
    for (outcome, cohort_spec), grp in overlap.groupby(
        ["outcome", "cohort_specification"]
    ):
        nonzero = grp[grp["delta_w_h"] != 0]
        overlap_deltas = nonzero.loc[nonzero["has_overlap_hex"], "delta_w_h"].tolist()
        only_r0 = nonzero.loc[
            (nonzero["n_hex_R0"] > 0) & (nonzero["n_hex_R1"] == 0), "delta_w_h"
        ].tolist()
        only_r1 = nonzero.loc[
            (nonzero["n_hex_R1"] > 0) & (nonzero["n_hex_R0"] == 0), "delta_w_h"
        ].tolist()
        rare = nonzero.loc[nonzero["rare_cell"], "delta_w_h"].tolist()

        sub = support[
            (support["outcome"] == outcome)
            & (support["cohort_specification"] == cohort_spec)
            & (support["change_type"] != "never_treated")
        ]
        n_region_only = int(
            sub.loc[sub["change_type"] == "region_only", "n_hexagons"].sum()
        )
        n_wm_only = int(
            sub.loc[sub["change_type"] == "workmode_only", "n_hexagons"].sum()
        )
        n_both = int(
            sub.loc[sub["change_type"] == "region_and_workmode", "n_hexagons"].sum()
        )
        severity = "ok"
        if not overlap_deltas:
            severity = "weak_overlap"
        elif len(only_r1) >= 3 or len(rare) >= 3:
            severity = "partial_overlap"

        notes.append(
            {
                "outcome": outcome,
                "cohort_specification": cohort_spec,
                "n_hex_region_only": n_region_only,
                "n_hex_workmode_only": n_wm_only,
                "n_hex_region_and_workmode": n_both,
                "overlap_nonzero_delta_values": ",".join(map(str, overlap_deltas)),
                "delta_only_R0": ",".join(map(str, only_r0)),
                "delta_only_R1": ",".join(map(str, only_r1)),
                "rare_overlap_deltas_lt30": ",".join(map(str, rare)),
                "identification_flag": severity,
                "note": (
                    "beta_R separates from workmode doses via region_only (Δw=0, R=1) "
                    "plus overlapping nonzero Δw at R=0 and R=1; "
                    "weak/rare overlap weakens dose-conditional interpretation."
                ),
            }
        )
    return pd.DataFrame(notes)


def run(out_dir: Path, applications_path: Path, hexagons_path: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_samples(applications_path, hexagons_path)
    weight_lookup = prepared["weight_lookup"]
    build_dose_order_frame = prepared["build_dose_order_frame"]
    aggregate_dose_panel = prepared["aggregate_dose_panel"]

    hex_support = support_from_hex_meta(weight_lookup)
    hex_support.to_csv(out_dir / "dose_support_hex_metadata.csv", index=False)

    # Distinct Δw by change_type from metadata (same filter as dose treated universe)
    delta_by_type = (
        weight_lookup.groupby(["change_type", "delta_w_h"], as_index=False)
        .agg(n_hexagons=("hex", "nunique"))
        .sort_values(["change_type", "delta_w_h"])
    )
    delta_by_type.to_csv(out_dir / "dose_support_delta_by_change_type.csv", index=False)

    panel_rows = []
    for outcome in DOSE_OUTCOMES:
        for include_0727, cohort_spec in [
            (False, "outcome_specific"),
            (True, "inclusive_2022-07-27"),
        ]:
            d = build_dose_order_frame(outcome, include_2022_07_27=include_0727)
            reg_panel = aggregate_dose_panel(d, outcome)
            panel_rows.append(
                support_from_reg_panel(reg_panel, outcome, cohort_spec)
            )
            print(
                f"[OK] support panel {outcome} / {cohort_spec}: "
                f"hex-day={len(reg_panel):,}, hex={reg_panel['hex'].nunique():,}"
            )

    support = pd.concat(panel_rows, ignore_index=True)
    support.to_csv(out_dir / "dose_support_R_by_delta.csv", index=False)

    overlap = overlap_report(support)
    overlap.to_csv(out_dir / "dose_support_R_delta_overlap.csv", index=False)

    by_type = change_type_delta_summary(support)
    by_type.to_csv(out_dir / "dose_support_by_change_type_delta.csv", index=False)

    notes = identification_notes(overlap, support)
    notes.to_csv(out_dir / "dose_support_identification_flags.csv", index=False)

    # Compact primary table for thesis reference: sch_flg / outcome_specific
    primary = support[
        (support["outcome"] == "sch_flg")
        & (support["cohort_specification"] == "outcome_specific")
    ].copy()
    primary.to_csv(out_dir / "dose_support_R_by_delta_primary.csv", index=False)

    try:
        out_rel = out_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        out_rel = out_dir
    print("Saved diagnostics to", out_rel.as_posix())
    return {
        "hex_support": hex_support,
        "support": support,
        "overlap": overlap,
        "by_type": by_type,
        "notes": notes,
        "primary": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "final",
    )
    parser.add_argument(
        "--applications",
        type=Path,
        default=ROOT / "data" / "raw" / "application_dataset.csv",
    )
    parser.add_argument(
        "--hexagons",
        type=Path,
        default=ROOT / "data" / "raw" / "hexagons_dataset.csv",
    )
    args = parser.parse_args()
    run(args.out_dir, args.applications, args.hexagons)


if __name__ == "__main__":
    main()
