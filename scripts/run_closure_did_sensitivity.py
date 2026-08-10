# -*- coding: utf-8 -*-
"""
Sensitivity основной DiD к эмпирическому закрытию гексагонов.

Спецификации:
  A — исходная CORE-выборка
  B — CORE без empirical_closed
  C — CORE с достаточной активностью (не empirical_closed, не near_closed,
      достаточные pre/post окна)

Меняется только состав treated-выборки; FE/кластеризация/окна — как в основной
спецификации. Нулевые дни в конверсионную панель не добавляются.
"""

from __future__ import annotations

import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from last_mile.filter import CORE_CHANGE_TYPES, build_analysis_panel  # noqa: E402
from last_mile.outcomes import success_horizon_coverage  # noqa: E402
from last_mile.hex_activity import (  # noqa: E402
    EMPIRICAL_CLOSED,
    NEAR_CLOSED,
)

OUT_DIR = ROOT / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_ORDER = list(CORE_CHANGE_TYPES)
OUTCOMES_MAIN = ["sch_flg", "meet_flg", "success_within_20", "utlz_within_25"]
EXCLUDED_COHORT_FOR_CONVERSION = pd.Timestamp("2022-07-27")
UTILIZATION_OBSERVATION_END = pd.Timestamp("2022-10-18")
UTLZ_HORIZON_DAYS = 25
UTLZ_OUTCOME = "utlz_within_25"
OUTCOME_EVENT_DTTM = {"utlz_flg": "real_utilization_dttm"}
WIDE_VALUE_COLUMNS = ["coef", "se", "ci_low", "ci_high", "n_treated_hex"]


def build_closure_sensitivity_wide(results: pd.DataFrame) -> pd.DataFrame:
    """Return the public one-header comparison table without changing estimates."""

    required = {"outcome", "change_type", "specification", *WIDE_VALUE_COLUMNS}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"closure sensitivity results missing columns: {missing}")

    wide = results.pivot(
        index=["outcome", "change_type"],
        columns="specification",
        values=WIDE_VALUE_COLUMNS,
    )
    present_specs = list(dict.fromkeys(results["specification"].astype(str)))
    specifications = [spec for spec in ("A", "B", "C") if spec in present_specs]
    specifications.extend(sorted(set(present_specs) - set(specifications)))
    wide = wide.reindex(
        columns=pd.MultiIndex.from_product([WIDE_VALUE_COLUMNS, specifications])
    )
    wide.columns = [
        f"{metric}_spec_{specification}" for metric, specification in wide.columns
    ]
    return wide.reset_index()


def _neg_utilization_lag_mask(df: pd.DataFrame) -> pd.Series:
    return df["real_utilization_dttm"].notna() & (
        df["real_utilization_dttm"] < df["request_timestamp"]
    )


def prepare_utlz_orders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[~_neg_utilization_lag_mask(df)].copy()
    horizon = out["request_timestamp"] + pd.Timedelta(days=UTLZ_HORIZON_DAYS)
    out[UTLZ_OUTCOME] = (
        out["real_utilization_dttm"].notna() & (out["real_utilization_dttm"] <= horizon)
    ).astype(float)
    return out


def maturity_horizon_days(outcome: str, apps_t: pd.DataFrame, apps_c: pd.DataFrame) -> int:
    if outcome == "success_within_20":
        return 20
    if outcome in {UTLZ_OUTCOME, "utlz_flg"}:
        return int(UTLZ_HORIZON_DAYS)
    event_col = OUTCOME_EVENT_DTTM.get(outcome)
    if event_col is None or event_col not in apps_t.columns:
        return 0
    lag_df = pd.concat(
        [apps_t[["request_timestamp", event_col]], apps_c[["request_timestamp", event_col]]],
        ignore_index=True,
    ).dropna()
    if lag_df.empty:
        return 0
    lags = (lag_df[event_col] - lag_df["request_timestamp"]).dt.total_seconds() / 86400.0
    lags = lags[(lags >= 0) & np.isfinite(lags)]
    if lags.empty:
        return 0
    return int(np.ceil(lags.quantile(0.90)))


def filter_mature_requests(
    df: pd.DataFrame, outcome: str, h: int, obs_max: pd.Timestamp
) -> pd.DataFrame:
    if h <= 0:
        return df.copy()
    obs_end = (
        UTILIZATION_OBSERVATION_END
        if outcome in {UTLZ_OUTCOME, "utlz_flg", "success_within_20"}
        else obs_max
    )
    return df[df["request_timestamp"] + pd.Timedelta(days=h) <= obs_end].copy()


def week_name(w: int) -> str:
    return f"pre{-w}" if w < 0 else f"post{w}"


def _full_rank_event_columns(d: pd.DataFrame, reg_cols: list[str]) -> list[str]:
    active = []
    for col in reg_cols:
        if col not in d.columns:
            continue
        if d[col].sum() == 0 or d[col].nunique(dropna=False) < 2:
            continue
        active.append(col)
    kept = []
    for col in active:
        candidate = kept + [col]
        x = d[candidate].to_numpy(dtype=float)
        if np.linalg.matrix_rank(x) > len(kept):
            kept.append(col)
    return kept


def summarize_post_effect(coefs, support, res=None, min_supporting_cohorts=None):
    weight_col = "n_treated_orders"
    post = coefs.loc[coefs.index >= 0, ["coef"]].copy().join(support, how="left")
    post[weight_col] = post[weight_col].fillna(0)
    if "n_supporting_cohorts" not in post.columns:
        post["n_supporting_cohorts"] = np.nan
    usable = post[post["coef"].notna() & (post[weight_col] > 0)].copy()
    if min_supporting_cohorts is not None:
        usable = usable[usable["n_supporting_cohorts"] >= min_supporting_cohorts].copy()
    if usable.empty:
        return {
            "avg_post_effect_weighted": np.nan,
            "avg_post_se": np.nan,
            "avg_post_ci_low": np.nan,
            "avg_post_ci_high": np.nan,
            "post_weeks": [],
        }
    weights = usable[weight_col].to_numpy(dtype=float)
    a = weights / weights.sum()
    beta = usable["coef"].to_numpy(dtype=float)
    weighted = float(a @ beta)
    post_weeks = [int(w) for w in usable.index.tolist()]
    avg_se = ci_low = ci_high = np.nan
    if res is not None:
        post_cols = [week_name(w) for w in post_weeks]
        if not any(c not in res.params.index for c in post_cols):
            cov = res.cov.loc[post_cols, post_cols].to_numpy(dtype=float)
            avg_se = float(np.sqrt(max(float(a @ cov @ a), 0.0)))
            ci_low = weighted - 1.96 * avg_se
            ci_high = weighted + 1.96 * avg_se
    return {
        "avg_post_effect_weighted": weighted,
        "avg_post_se": avg_se,
        "avg_post_ci_low": ci_low,
        "avg_post_ci_high": ci_high,
        "post_weeks": post_weeks,
    }


def did_event_study(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    outcome: str,
    max_week: int = 8,
):
    name = week_name
    weeks = [w for w in range(-max_week, max_week + 1) if w != -1]
    t = treated.copy()
    t["rel_week"] = (t["days_from_treatment"] // 7).clip(-max_week, max_week)
    t["is_treated_es"] = 1
    c = control.copy()
    c["rel_week"] = np.nan
    c["is_treated_es"] = 0
    d = pd.concat([t, c], ignore_index=True)
    d["date"] = d["request_timestamp"].dt.normalize()
    for w in weeks:
        d[name(w)] = ((d["is_treated_es"] == 1) & (d["rel_week"] == w)).astype(int)
    all_reg_cols = [name(w) for w in weeks]
    d = d.dropna(subset=[outcome]).copy()

    treated_support_base = d[(d["is_treated_es"] == 1) & d["rel_week"].notna()].copy()
    treated_support_base["rel_week"] = treated_support_base["rel_week"].astype(int)
    n_orders = treated_support_base.groupby("rel_week")["request_timestamp"].count()
    n_hex = treated_support_base.groupby("rel_week")["hex"].nunique()
    n_hex_days = (
        treated_support_base[["rel_week", "hex", "date"]]
        .drop_duplicates()
        .groupby("rel_week")
        .size()
    )
    n_cohorts = treated_support_base.groupby("rel_week")["treatment_date"].nunique()
    support = pd.DataFrame(
        {
            "n_treated_orders": n_orders,
            "n_treated_hex": n_hex,
            "n_treated_hex_days": n_hex_days,
            "n_supporting_cohorts": n_cohorts,
        }
    ).reindex(weeks).fillna(0).astype(int)

    agg_dict = {outcome: "mean", "request_timestamp": "count"}
    agg_dict.update({col: "max" for col in all_reg_cols})
    reg_panel = (
        d.groupby(["hex", "date"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"request_timestamp": "n_orders"})
        .set_index(["hex", "date"])
        .sort_index()
    )
    reg_cols = _full_rank_event_columns(reg_panel, all_reg_cols)
    res = (
        PanelOLS(
            reg_panel[outcome],
            reg_panel[reg_cols],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
        )
        .fit(cov_type="clustered", cluster_entity=True, low_memory=True)
    )
    ci = res.conf_int()
    rows = []
    for w in weeks:
        col = name(w)
        if col in res.params.index:
            rows.append(
                {
                    "rel_week": w,
                    "coef": res.params[col],
                    "ci_low": ci.loc[col, "lower"],
                    "ci_high": ci.loc[col, "upper"],
                    "pval": res.pvalues[col],
                }
            )
        else:
            rows.append(
                {
                    "rel_week": w,
                    "coef": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "pval": np.nan,
                }
            )
    coefs = pd.DataFrame(rows).set_index("rel_week").sort_index()
    return res, coefs, support


def load_spec_filters() -> dict[str, set[str] | None]:
    cls = pd.read_csv(ROOT / "outputs" / "hex_empirical_closure_classification.csv")
    emp_closed = set(cls.loc[cls["empirical_status"] == EMPIRICAL_CLOSED, "hex"])
    # Spec C: keep hexes with sufficient pre activity and full post window,
    # not empirical_closed / near_closed.
    keep_c = set(
        cls.loc[
            cls["sufficient_pre_activity"]
            & cls["full_post_window"]
            & ~cls["empirical_status"].isin([EMPIRICAL_CLOSED, NEAR_CLOSED]),
            "hex",
        ]
    )
    # Also allow CORE hexes that are continues_active with full criteria
    # (already covered). For hexes not in classification (shouldn't happen for treated),
    # Spec C drops them.
    return {
        "A_core": None,  # no extra filter
        "B_core_ex_empirical_closed": emp_closed,
        "C_core_sufficient_activity": ("keep_only", keep_c),
    }


def main() -> None:
    print("Building analysis panel...")
    result = build_analysis_panel(
        applications_path=ROOT / "data" / "raw" / "application_dataset.csv",
        hexagons_path=ROOT / "data" / "raw" / "hexagons_dataset.csv",
        min_orders_per_hex=5,
    )
    apps_t = result["treated_orders"].copy()
    apps_c = result["control_orders"].copy()
    for df in (apps_t, apps_c):
        if "cohort" in df.columns and "treatment_date" not in df.columns:
            df["treatment_date"] = df["cohort"]
        if "days_from_treatment" not in df.columns and "treatment_date" in df.columns:
            if df["treatment_date"].notna().any():
                df["days_from_treatment"] = (
                    df["request_timestamp"] - df["treatment_date"]
                ).dt.days

    apps_t_utlz = prepare_utlz_orders(apps_t)
    apps_c_utlz = prepare_utlz_orders(apps_c)
    apps_t, _ = success_horizon_coverage(
        apps_t, horizons=(20,), observation_end=UTILIZATION_OBSERVATION_END
    )
    apps_c, _ = success_horizon_coverage(
        apps_c, horizons=(20,), observation_end=UTILIZATION_OBSERVATION_END
    )
    obs_max = max(apps_t["request_timestamp"].max(), apps_c["request_timestamp"].max())

    filters = load_spec_filters()
    # B excludes empirical_closed set; C keeps only keep_c
    emp_closed = filters["B_core_ex_empirical_closed"]
    keep_c = filters["C_core_sufficient_activity"][1]

    horizons = {
        o: maturity_horizon_days(o, apps_t, apps_c) for o in OUTCOMES_MAIN
    }
    # meet_flg maturity in notebook uses first_success? Check notebook OUTCOME_EVENT_DTTM
    # For sch/meet h=0 typically.

    rows = []
    specs = [
        ("A", "CORE исходная", None, "exclude"),
        ("B", "CORE без empirical_closed", emp_closed, "exclude"),
        ("C", "CORE достаточная активность", keep_c, "keep_only"),
    ]

    valid_all = set(pd.to_datetime(apps_t["treatment_date"].dropna().unique()))
    valid_conv = valid_all - {EXCLUDED_COHORT_FOR_CONVERSION}

    for spec_id, spec_label, hex_set, mode in specs:
        for outcome in OUTCOMES_MAIN:
            cohorts = valid_conv if outcome != "sch_flg" else valid_all
            h = horizons[outcome]
            src_t = apps_t_utlz if outcome == UTLZ_OUTCOME else apps_t
            src_c = apps_c_utlz if outcome == UTLZ_OUTCOME else apps_c
            control = filter_mature_requests(src_c, outcome, h, obs_max)
            min_cohorts = 2 if outcome == UTLZ_OUTCOME else None

            for ctype in TYPE_ORDER:
                treated = src_t[
                    (src_t["change_type"] == ctype)
                    & (src_t["treatment_date"].isin(cohorts))
                ].copy()
                if mode == "exclude" and hex_set:
                    treated = treated[~treated["hex"].isin(hex_set)].copy()
                elif mode == "keep_only":
                    treated = treated[treated["hex"].isin(hex_set)].copy()
                treated = filter_mature_requests(treated, outcome, h, obs_max)

                if treated.empty:
                    rows.append(
                        {
                            "specification": spec_id,
                            "specification_label": spec_label,
                            "outcome": outcome,
                            "change_type": ctype,
                            "coef": np.nan,
                            "se": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "p_value": np.nan,
                            "n_hexagons": 0,
                            "n_hex_day": 0,
                            "n_orders": 0,
                            "n_treated_hex": 0,
                            "n_control_hex": int(control["hex"].nunique()),
                            "filter_rule": f"{mode}:{spec_label}",
                            "status": "empty_treated",
                        }
                    )
                    continue

                print(f"Estimating {spec_id} | {outcome} × {ctype} ...")
                res, coefs, support = did_event_study(treated, control, outcome)
                post = summarize_post_effect(
                    coefs, support, res=res, min_supporting_cohorts=min_cohorts
                )
                est = post["avg_post_effect_weighted"]
                se = post["avg_post_se"]
                if pd.notna(est) and pd.notna(se) and float(se) > 0:
                    p_value = float(erfc(abs(float(est) / float(se)) / sqrt(2.0)))
                else:
                    p_value = np.nan
                n_hexagons = int(
                    res.model.dependent.dataframe.index.get_level_values(0).nunique()
                )
                rows.append(
                    {
                        "specification": spec_id,
                        "specification_label": spec_label,
                        "outcome": outcome,
                        "change_type": ctype,
                        "coef": est,
                        "se": se,
                        "ci_low": post["avg_post_ci_low"],
                        "ci_high": post["avg_post_ci_high"],
                        "p_value": p_value,
                        "n_hexagons": n_hexagons,
                        "n_hex_day": int(getattr(res, "nobs", np.nan)),
                        "n_orders": int(len(treated) + len(control)),
                        "n_treated_hex": int(treated["hex"].nunique()),
                        "n_control_hex": int(control["hex"].nunique()),
                        "n_treated_orders": int(len(treated)),
                        "filter_rule": f"{mode}:{spec_label}",
                        "status": "ok",
                    }
                )
                print(
                    f"  coef={est*100:+.2f} п.п. se={se*100:.2f} "
                    f"n_treated_hex={treated['hex'].nunique()}"
                )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "closure_did_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print("Saved", out_path)

    # Compact A/B/C comparison with one public CSV header row.
    wide = build_closure_sensitivity_wide(out)
    wide.to_csv(OUT_DIR / "closure_did_sensitivity_wide.csv", index=False)

    # Repeat all three sch_flg closure specifications without the problematic
    # 2022-07-27 cohort and export original/strict estimates side by side.
    strict_rows = []
    control_sch = apps_c.copy()
    for spec_id, spec_label, hex_set, mode in specs:
        for ctype in TYPE_ORDER:
            treated = apps_t[
                (apps_t["change_type"] == ctype)
                & apps_t["treatment_date"].isin(valid_conv)
            ].copy()
            if mode == "exclude" and hex_set:
                treated = treated[~treated["hex"].isin(hex_set)].copy()
            elif mode == "keep_only":
                treated = treated[treated["hex"].isin(hex_set)].copy()
            res, coefs, support = did_event_study(
                treated, control_sch, "sch_flg"
            )
            post = summarize_post_effect(coefs, support, res=res)
            strict_rows.append(
                {
                    "specification": spec_id,
                    "specification_label": spec_label,
                    "change_type": ctype,
                    "cohort_policy": "exclude_2022-07-27",
                    "estimate": post["avg_post_effect_weighted"],
                    "std_error": post["avg_post_se"],
                    "ci_lower": post["avg_post_ci_low"],
                    "ci_upper": post["avg_post_ci_high"],
                    "n_obs": int(res.nobs),
                    "n_treated_hexagons": int(treated["hex"].nunique()),
                }
            )
    original_sch = out[out["outcome"] == "sch_flg"].rename(
        columns={
            "coef": "estimate",
            "se": "std_error",
            "ci_low": "ci_lower",
            "ci_high": "ci_upper",
            "n_hex_day": "n_obs",
            "n_treated_hex": "n_treated_hexagons",
        }
    )
    original_sch = original_sch[
        [
            "specification",
            "specification_label",
            "change_type",
            "estimate",
            "std_error",
            "ci_lower",
            "ci_upper",
            "n_obs",
            "n_treated_hexagons",
        ]
    ].copy()
    original_sch["cohort_policy"] = "all_sch_cohorts"
    sch_robustness = pd.concat(
        [original_sch, pd.DataFrame(strict_rows)], ignore_index=True
    )
    sch_robustness.to_csv(OUT_DIR / "sch_robustness_summary.csv", index=False)
    print(out.groupby("specification")["status"].value_counts())


if __name__ == "__main__":
    main()
