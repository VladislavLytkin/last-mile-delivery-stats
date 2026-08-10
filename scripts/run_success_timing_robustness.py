# -*- coding: utf-8 -*-
"""Success-timing robustness: common-support DiD, multiple testing, long-tail.

Also appends corresponding cells to notebooks/success_timing_diagnostics.ipynb.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from last_mile.filter import build_analysis_panel  # noqa: E402
from last_mile.outcomes import success_horizon_coverage  # noqa: E402
from last_mile.plot_style import PALETTE, save_figure, style_axes, with_plot_style  # noqa: E402
from linearmodels.panel import PanelOLS  # noqa: E402

OUT_DIR = ROOT / "outputs" / "success_timing_diagnostics"
FIG_DIR = ROOT / "figures" / "success_timing_diagnostics"
EMP_FIG = ROOT / "figures" / "empirical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
EMP_FIG.mkdir(parents=True, exist_ok=True)

TYPE_ORDER = ["region_and_workmode", "region_only", "workmode_only"]
TYPE_LABELS_EN = {
    "region_and_workmode": "Region + work mode",
    "region_only": "Region only",
    "workmode_only": "Work mode only",
}
TYPE_COLORS = {
    "region_and_workmode": PALETTE["region_and_workmode"],
    "region_only": PALETTE["region_only"],
    "workmode_only": PALETTE["workmode_only"],
}
DID_HORIZONS = [3, 7, 14, 25]
SUCCESS_OBSERVATION_END = pd.Timestamp("2022-10-18")
EXCLUDED_COHORT_FOR_CONVERSION = pd.Timestamp("2022-07-27")
COMMON_MATURITY_H = 25


class EmptyWaldResult:
    pval = np.nan
    stat = np.nan


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


def summarize_post_effect(coefs, support, res=None) -> dict:
    weight_col = "n_treated_orders"
    post = coefs.loc[coefs.index >= 0, ["coef"]].copy().join(support, how="left")
    post[weight_col] = post[weight_col].fillna(0)
    usable = post[post["coef"].notna() & (post[weight_col] > 0)].copy()
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
    allowed_weeks: set[int] | None = None,
):
    name = week_name
    weeks = [w for w in range(-max_week, max_week + 1) if w != -1]
    if allowed_weeks is not None:
        # Keep all pre weeks in window; restrict post to intersection support.
        weeks = [w for w in weeks if w < 0 or w in allowed_weeks]

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
    support = (
        pd.DataFrame(
            {
                "n_treated_orders": n_orders,
                "n_treated_hex": n_hex,
                "n_treated_hex_days": n_hex_days,
                "n_supporting_cohorts": n_cohorts,
            }
        )
        .reindex(weeks)
        .fillna(0)
        .astype(int)
    )

    agg_dict = {outcome: "mean", "request_timestamp": "count"}
    agg_dict.update({col: "max" for col in all_reg_cols})
    reg_panel = (
        d.groupby(["hex", "date"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"request_timestamp": "n_orders"})
        .set_index(["hex", "date"])
        .sort_index()
    )
    if not reg_panel.index.is_unique:
        raise ValueError(f"non-unique hex×date for {outcome}")

    reg_cols = _full_rank_event_columns(reg_panel, all_reg_cols)
    if not reg_cols:
        raise ValueError(f"no estimable dummies for {outcome}")

    res = PanelOLS(
        reg_panel[outcome],
        reg_panel[reg_cols],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True, low_memory=True)

    ci = res.conf_int()
    rows = []
    for w in weeks:
        col = name(w)
        if col in res.params.index:
            rows.append(
                {
                    "rel_week": w,
                    "coef": float(res.params[col]),
                    "ci_low": float(ci.loc[col, "lower"]),
                    "ci_high": float(ci.loc[col, "upper"]),
                    "pval": float(res.pvalues[col]),
                    "estimated": True,
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
                    "estimated": False,
                }
            )
    coefs = pd.DataFrame(rows).set_index("rel_week").sort_index()
    pre_cols = [name(w) for w in weeks if w < -1 and name(w) in res.params.index]
    wald = (
        res.wald_test(formula=", ".join(f"{col} = 0" for col in pre_cols))
        if pre_cols
        else EmptyWaldResult()
    )
    meta = {
        "n_obs_hex_day": int(res.nobs),
        "n_hex": int(reg_panel.index.get_level_values(0).nunique()),
        "n_treated_hex": int(t["hex"].nunique()),
        "n_treated_orders": int(len(t)),
    }
    return res, coefs, wald, support, meta


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj_sorted = np.empty(m)
    running = 0.0
    for i in range(m):
        val = min(1.0, p[order[i]] * (m - i))
        running = max(running, val)
        adj_sorted[i] = running
    for i, orig in enumerate(order):
        out[orig] = adj_sorted[i]
    return out


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    q_sorted = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(1.0, p[order[i]] * m / rank)
        prev = min(prev, val)
        q_sorted[i] = prev
    for i, orig in enumerate(order):
        out[orig] = q_sorted[i]
    return out


def two_sided_pval(estimate: float, se: float) -> float:
    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0):
        return float("nan")
    z = abs(estimate / se)
    return float(2.0 * (1.0 - stats.norm.cdf(z)))


def load_apps():
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
        if "date" not in df.columns:
            df["date"] = df["request_timestamp"].dt.normalize()
        for col in ("request_timestamp", "first_success_dttm", "treatment_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    if "days_from_treatment" not in apps_t.columns:
        apps_t["days_from_treatment"] = (
            apps_t["request_timestamp"] - apps_t["treatment_date"]
        ).dt.days
    apps_t, _ = success_horizon_coverage(
        apps_t, horizons=DID_HORIZONS, observation_end=SUCCESS_OBSERVATION_END
    )
    apps_c, _ = success_horizon_coverage(
        apps_c, horizons=DID_HORIZONS, observation_end=SUCCESS_OBSERVATION_END
    )
    valid = {
        pd.Timestamp(x) for x in pd.to_datetime(apps_t["treatment_date"].dropna().unique())
    } - {EXCLUDED_COHORT_FOR_CONVERSION}
    return apps_t, apps_c, valid


def mature_mask(df: pd.DataFrame, h: int) -> pd.Series:
    return df["request_timestamp"] + pd.Timedelta(days=h) <= SUCCESS_OBSERVATION_END


def post_support_weeks(df: pd.DataFrame, max_week: int = 8) -> set[int]:
    rel = (df["days_from_treatment"] // 7).clip(-max_week, max_week)
    counts = rel.value_counts()
    return {int(w) for w, n in counts.items() if int(w) >= 0 and int(n) > 0}


def run_common_support(apps_t, apps_c, valid_cohorts):
    print("=== Common-support multi-horizon DiD ===")
    # Intersection of post weeks with support across change_types on H=25 mature sample
    # First compute per-type post weeks on common maturity sample; then per-type keep
    # intersection across horizons is automatic (same sample). Also intersect post weeks
    # that appear with support for ALL change types? User said for H=3,7,14,25 — same
    # event weeks. Per change_type estimation is separate, so intersection is across H
    # within type (= identical on mature_25).
    rows = []
    for ctype in TYPE_ORDER:
        t0 = apps_t[
            (apps_t["change_type"] == ctype)
            & (apps_t["treatment_date"].isin(valid_cohorts))
            & mature_mask(apps_t, COMMON_MATURITY_H)
        ].copy()
        c0 = apps_c.loc[mature_mask(apps_c, COMMON_MATURITY_H)].copy()
        common_post = post_support_weeks(t0)
        print(f"{ctype}: common mature sample treated={len(t0):,} control={len(c0):,} post_weeks={sorted(common_post)}")
        for H in DID_HORIZONS:
            outcome = f"success_within_{H}"
            print(f"  estimating {outcome} ...")
            try:
                res, coefs, wald, support, meta = did_event_study(
                    t0, c0, outcome, max_week=8, allowed_weeks=common_post
                )
                post = summarize_post_effect(coefs, support, res=res)
                rows.append(
                    {
                        "change_type": ctype,
                        "horizon_days": H,
                        "n_obs": meta["n_obs_hex_day"],
                        "n_hex": meta["n_hex"],
                        "n_treated_hex": meta["n_treated_hex"],
                        "pretrend_pvalue": float(getattr(wald, "pval", np.nan)),
                        "post_estimate_pp": 100.0 * post["avg_post_effect_weighted"]
                        if np.isfinite(post["avg_post_effect_weighted"])
                        else np.nan,
                        "post_se_pp": 100.0 * post["avg_post_se"]
                        if np.isfinite(post["avg_post_se"])
                        else np.nan,
                        "ci_low_pp": 100.0 * post["avg_post_ci_low"]
                        if np.isfinite(post["avg_post_ci_low"])
                        else np.nan,
                        "ci_high_pp": 100.0 * post["avg_post_ci_high"]
                        if np.isfinite(post["avg_post_ci_high"])
                        else np.nan,
                        "event_weeks_used": ",".join(str(w) for w in post["post_weeks"]),
                        "common_support_flag": True,
                        "common_post_weeks": ",".join(str(w) for w in sorted(common_post)),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                print(f"  SKIP {ctype} H={H}: {exc}")
                rows.append(
                    {
                        "change_type": ctype,
                        "horizon_days": H,
                        "n_obs": np.nan,
                        "n_hex": np.nan,
                        "n_treated_hex": np.nan,
                        "pretrend_pvalue": np.nan,
                        "post_estimate_pp": np.nan,
                        "post_se_pp": np.nan,
                        "ci_low_pp": np.nan,
                        "ci_high_pp": np.nan,
                        "event_weeks_used": "",
                        "common_support_flag": True,
                        "common_post_weeks": "",
                        "status": f"error:{type(exc).__name__}",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "success_horizon_did_common_support.csv", index=False)
    return out


@with_plot_style
def plot_common_support(cs: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, ctype in zip(axes, TYPE_ORDER):
        sub = cs.loc[(cs["change_type"] == ctype) & (cs["status"] == "ok")].sort_values(
            "horizon_days"
        )
        ax.set_title(TYPE_LABELS_EN[ctype])
        if sub.empty:
            continue
        yerr = np.vstack(
            [
                sub["post_estimate_pp"] - sub["ci_low_pp"],
                sub["ci_high_pp"] - sub["post_estimate_pp"],
            ]
        )
        ax.errorbar(
            sub["horizon_days"],
            sub["post_estimate_pp"],
            yerr=yerr,
            fmt="o-",
            color=TYPE_COLORS[ctype],
            ecolor=PALETTE["text_muted"],
            capsize=3,
            lw=1.4,
        )
        ax.axhline(0, color=PALETTE["zero"], lw=0.9)
        for _, r in sub.iterrows():
            if pd.notna(r["pretrend_pvalue"]) and r["pretrend_pvalue"] < 0.05:
                ax.text(
                    r["horizon_days"],
                    r["post_estimate_pp"],
                    "*",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
        ax.set_xlabel("Horizon H (days)")
        style_axes(ax)
    axes[0].set_ylabel("Weighted post DiD estimate (pp)")
    fig.suptitle(
        "Common-support multi-horizon event-study\n(* = pretrend p<0.05; not a significance star)",
        y=1.05,
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "multi_horizon_did_common_support.pdf")


def run_multiple_testing(natural: pd.DataFrame, common: pd.DataFrame):
    parts = []
    for label, df, est_col, se_col in [
        ("natural_maturity", natural, "DID_post_estimate_pp", "SE"),
        ("common_support", common, "post_estimate_pp", "post_se_pp"),
    ]:
        d = df.copy()
        if "status" in d.columns:
            d = d.loc[d["status"] == "ok"].copy()
        d["spec"] = label
        d["raw_pvalue"] = [
            two_sided_pval(e, s) for e, s in zip(d[est_col], d[se_col])
        ]
        parts.append(
            d[
                [
                    "spec",
                    "change_type",
                    "horizon_days",
                    est_col,
                    se_col,
                    "raw_pvalue",
                    "pretrend_pvalue",
                ]
            ].rename(columns={est_col: "post_estimate_pp", se_col: "post_se_pp"})
        )
    # Primary multiple-testing family: 12 natural-maturity post estimates
    primary = parts[0].copy()
    primary["holm_pvalue"] = holm_adjust(primary["raw_pvalue"].to_numpy())
    primary["bh_qvalue"] = bh_fdr(primary["raw_pvalue"].to_numpy())
    # Also attach common-support family (exploratory secondary)
    secondary = parts[1].copy()
    secondary["holm_pvalue"] = holm_adjust(secondary["raw_pvalue"].to_numpy())
    secondary["bh_qvalue"] = bh_fdr(secondary["raw_pvalue"].to_numpy())
    out = pd.concat([primary, secondary], ignore_index=True)
    out.to_csv(OUT_DIR / "success_horizon_multiple_testing.csv", index=False)
    return out


def run_long_tail(apps_t, apps_c, valid_cohorts):
    print("=== Long-tail diagnostic (>60 days) ===")
    apps = pd.concat([apps_t, apps_c], ignore_index=True)
    core = apps[
        apps["change_type"].isin(TYPE_ORDER)
        | (~apps["is_treated"])
    ].copy()
    core = core[
        (~core["is_treated"]) | core["treatment_date"].isin(valid_cohorts)
    ].copy()
    core["success_flg"] = pd.to_numeric(core["success_flg"], errors="coerce")
    delay = (
        (core["first_success_dttm"] - core["request_timestamp"]).dt.total_seconds()
        / 86400.0
    )
    valid = (
        (core["success_flg"] == 1)
        & core["first_success_dttm"].notna()
        & delay.ge(0)
    )
    core["success_delay_days"] = np.where(valid, delay, np.nan)
    succ = core.loc[core["success_delay_days"].notna()].copy()
    tail = succ.loc[succ["success_delay_days"] > 60].copy()
    cluster = succ.loc[succ["success_delay_days"].between(100, 110)].copy()

    rows = []
    def add(check, **kw):
        rows.append({"check": check, **kw})

    add(
        "n_valid_successes",
        n=int(len(succ)),
        share=1.0,
        value="",
    )
    add(
        "n_delay_gt_60",
        n=int(len(tail)),
        share=float(len(tail) / len(succ)) if len(succ) else np.nan,
        value="",
    )
    add(
        "n_delay_100_to_110",
        n=int(len(cluster)),
        share=float(len(cluster) / len(succ)) if len(succ) else np.nan,
        value="",
    )
    if len(tail):
        add("tail_min", n=None, share=None, value=float(tail["success_delay_days"].min()))
        add("tail_median", n=None, share=None, value=float(tail["success_delay_days"].median()))
        add("tail_p90", n=None, share=None, value=float(tail["success_delay_days"].quantile(0.9)))
        add("tail_max", n=None, share=None, value=float(tail["success_delay_days"].max()))

    # distributions
    for label, series in [
        ("request_month", tail["request_timestamp"].dt.to_period("M").astype(str)),
        ("success_month", tail["first_success_dttm"].dt.to_period("M").astype(str)),
        ("cohort", tail["treatment_date"].dt.strftime("%Y-%m-%d").fillna("control")),
        ("change_type", tail["change_type"].fillna("control")),
    ]:
        vc = series.value_counts(dropna=False)
        for k, v in vc.items():
            add(f"dist_{label}", n=int(v), share=float(v / len(tail)) if len(tail) else np.nan, value=str(k))

    if "is_treated" in tail.columns:
        # pre/post among treated
        tt = apps_t.merge(
            tail[["hex", "request_timestamp"]],
            on=["hex", "request_timestamp"],
            how="inner",
        )
        if "days_from_treatment" in tt.columns:
            pre = int((tt["days_from_treatment"] < 0).sum())
            post = int((tt["days_from_treatment"] >= 0).sum())
            add("treated_tail_pre", n=pre, share=pre / max(len(tt), 1), value="")
            add("treated_tail_post", n=post, share=post / max(len(tt), 1), value="")

    # Peak first_success dates
    if len(tail):
        peak = tail["first_success_dttm"].dt.normalize().value_counts().head(15)
        for dt, v in peak.items():
            add(
                "peak_first_success_date",
                n=int(v),
                share=float(v / len(tail)),
                value=str(pd.Timestamp(dt).date()),
            )
        # Near observation end
        near_end = int((tail["first_success_dttm"] >= SUCCESS_OBSERVATION_END - pd.Timedelta(days=7)).sum())
        add(
            "tail_success_within_7d_of_obs_end",
            n=near_end,
            share=near_end / len(tail),
            value="",
        )
        # Exact 100-110 cluster share of peaks
        if len(cluster):
            peak_c = cluster["first_success_dttm"].dt.normalize().value_counts().head(10)
            for dt, v in peak_c.items():
                add(
                    "cluster100_110_peak_first_success_date",
                    n=int(v),
                    share=float(v / len(cluster)),
                    value=str(pd.Timestamp(dt).date()),
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "success_long_tail_diagnostic.csv", index=False)

    # Honest conclusion
    peak_share = 0.0
    if len(tail):
        top = tail["first_success_dttm"].dt.normalize().value_counts().iloc[0]
        peak_share = float(top / len(tail))
    if peak_share >= 0.25:
        conclusion = (
            "data artifact suspected: a large share of long-delay successes "
            "concentrates on few first_success calendar dates"
        )
    elif len(cluster) / max(len(succ), 1) > 0.03 and peak_share < 0.15:
        conclusion = (
            "plausible delayed business process or mixed mechanism; "
            "available fields do not uniquely identify an administrative artifact"
        )
    else:
        conclusion = (
            "source cannot be determined from available data; "
            "long tail is retained only as descriptive context and is not used for causal claims"
        )
    (OUT_DIR / "success_long_tail_conclusion.txt").write_text(conclusion, encoding="utf-8")
    print("LONG-TAIL CONCLUSION:", conclusion)
    return out, conclusion


def write_final_summary(natural, common, mt, long_tail_conclusion):
    lines = [
        "## Final publication summary (success timing)",
        "",
        "### A. Descriptive evidence",
        "- Most successful meetings occur in the first days after application;",
        "- the delay distribution has a pronounced right tail;",
        "- conditional delay among successes is descriptive only (conditioned-on-outcome).",
        f"- Long-tail (>60d) diagnostic conclusion: {long_tail_conclusion}",
        "",
        "### B. Natural-maturity multi-horizon DiD",
    ]
    for ctype in TYPE_ORDER:
        lines.append(f"#### {ctype}")
        sub = natural.loc[natural["change_type"] == ctype]
        for _, r in sub.sort_values("horizon_days").iterrows():
            lines.append(
                f"- H={int(r['horizon_days'])}: {r['DID_post_estimate_pp']:.2f} pp "
                f"[{r['CI_low']:.2f}; {r['CI_high']:.2f}], pretrend p={r['pretrend_pvalue']:.3f}"
            )
    lines += ["", "### C. Common-support multi-horizon DiD"]
    for ctype in TYPE_ORDER:
        lines.append(f"#### {ctype}")
        sub = common.loc[common["change_type"] == ctype]
        for _, r in sub.sort_values("horizon_days").iterrows():
            lines.append(
                f"- H={int(r['horizon_days'])}: {r['post_estimate_pp']:.2f} pp "
                f"[{r['ci_low_pp']:.2f}; {r['ci_high_pp']:.2f}], pretrend p={r['pretrend_pvalue']:.3f}"
            )
    lines += [
        "",
        "### D. Multiple-testing diagnostic",
        "Individual horizon-specific p-values are exploratory because multiple "
        "change types and horizons are examined simultaneously.",
    ]
    nat_mt = mt.loc[mt["spec"] == "natural_maturity"]
    n_raw = int((nat_mt["raw_pvalue"] < 0.05).sum())
    n_holm = int((nat_mt["holm_pvalue"] < 0.05).sum())
    n_bh = int((nat_mt["bh_qvalue"] < 0.05).sum())
    lines.append(
        f"- Natural-maturity family (12 tests): raw p<0.05 in {n_raw}; "
        f"Holm-significant: {n_holm}; BH FDR q<0.05: {n_bh}."
    )
    if n_holm == 0 and n_bh == 0:
        lines.append(
            "- After multiple-testing correction, no individual post-effect remains significant."
        )
    lines += [
        "",
        "### E. Limitations",
        "- Horizon-specific maturity changes follow-up and post support.",
        "- Multiple testing across 3×4 specifications.",
        "- Conditional success-delay is not causal.",
        "- Common-support specification addresses the first limitation as robustness.",
        "",
        "### Interpretation rule",
        "Do not claim that changes accelerated/decelerated success unless the signal "
        "survives common support, pretrends, CI excluding 0, and multiple-testing.",
        "Preferred language: local short-horizon signal / exploratory evidence / "
        "no stable common effect.",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "final_publication_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return text


def append_notebook_cells():
    nb_path = ROOT / "notebooks" / "success_timing_diagnostics.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    marker = "COMMON-SUPPORT MULTI-HORIZON"
    if any(marker in "".join(c.get("source", [])) for c in nb["cells"]):
        print("Notebook already contains common-support cells; skipping append.")
        return

    def md(s):
        lines = s.strip("\n").splitlines(True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": lines}

    def code(s):
        lines = s.strip("\n").splitlines(True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": [],
            "source": lines,
        }

    new_cells = [
        md(
            """
## Robustness: common support, multiple testing, long-tail

Дополнительные checks перед публикационным использованием.
Запуск также воспроизводим через `scripts/run_success_timing_robustness.py`.
"""
        ),
        code(
            """
# ============================================================
# Robustness A: common-support multi-horizon DiD
# ============================================================
# На одной mature_H=25 выборке переоцениваются H=3,7,14,25.
# Post event weeks = support на этой выборке (общий для всех H).

from scripts.run_success_timing_robustness import (
    run_common_support,
    plot_common_support,
    run_multiple_testing,
    run_long_tail,
    write_final_summary,
)

# Reuse in-memory frames if notebook already loaded them
_common = run_common_support(apps_t, apps_c, VALID_COHORTS_CONVERSION)
display(_common)
plot_common_support(_common)
print('Saved common-support CSV/figure')
"""
        ),
        code(
            """
# ============================================================
# Robustness B: multiple testing (12 natural + 12 common)
# ============================================================
_natural = pd.read_csv(OUT_DIR / 'success_horizon_did.csv')
_mt = run_multiple_testing(_natural, _common)
display(_mt)
print('Saved success_horizon_multiple_testing.csv')
print(
    'NOTE: Individual horizon-specific p-values are exploratory because multiple '
    'change types and horizons are examined simultaneously.'
)
"""
        ),
        code(
            """
# ============================================================
# Robustness C: long-tail diagnostic (~100-110 days)
# ============================================================
_lt, _lt_conclusion = run_long_tail(apps_t, apps_c, VALID_COHORTS_CONVERSION)
display(_lt.head(40))
print('LONG-TAIL CONCLUSION:', _lt_conclusion)
"""
        ),
        code(
            """
# ============================================================
# Final publication summary A-E
# ============================================================
_final = write_final_summary(_natural, _common, _mt, _lt_conclusion)
from IPython.display import Markdown
display(Markdown(_final))
"""
        ),
    ]
    # Insert before final checklist markdown if present
    insert_at = len(nb["cells"]) - 2
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "Final checklist" in "".join(c.get("source", [])):
            insert_at = i
            break
        if c["cell_type"] == "markdown" and "## 11. Outputs" in "".join(c.get("source", [])):
            insert_at = i
            break
    nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Appended {len(new_cells)} cells to {nb_path}")


def copy_publication_figures():
    for name in [
        "cumulative_success_curves.pdf",
        "cumulative_success_curves.png",
        "multi_horizon_did_common_support.pdf",
        "multi_horizon_did_common_support.png",
    ]:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, EMP_FIG / name)
            print("Copied", src.name, "-> figures/empirical/")


def main():
    natural_path = OUT_DIR / "success_horizon_did.csv"
    if not natural_path.exists():
        raise SystemExit("Missing success_horizon_did.csv — run the notebook DiD loop first.")
    natural = pd.read_csv(natural_path)

    apps_t, apps_c, valid = load_apps()
    common = run_common_support(apps_t, apps_c, valid)
    plot_common_support(common)
    mt = run_multiple_testing(natural, common)
    _, lt_conclusion = run_long_tail(apps_t, apps_c, valid)
    write_final_summary(natural, common, mt, lt_conclusion)
    copy_publication_figures()
    append_notebook_cells()
    print("DONE")


if __name__ == "__main__":
    main()
