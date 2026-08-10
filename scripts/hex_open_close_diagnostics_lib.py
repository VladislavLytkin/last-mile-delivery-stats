# -*- coding: utf-8 -*-
"""
Diagnostic library: work_mode open/close vs actual applications.

Does NOT modify DiD treatment assignment or production change_type rules.
Reuses last_mile.io loaders and last_mile.filter.classify_hexagons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from last_mile.filter import (
    EXCLUDED_COHORTS,
    INACTIVE_REGION,
    STUDY_END,
    STUDY_START,
    classify_hexagons,
    resolve_hexagon_treatments,
)
from last_mile.io import load_applications, load_hexagons

# Event-study calendar around treatment (clipped to observable app range).
EVENT_PAD_DAYS = 42
HORIZONS = (7, 14, 21, 28)
PRIMARY_HORIZON = 14
MIN_PRE_APPS = 5
MIN_PRE_ACTIVE_DAYS = 3
HEATMAP_MAX_HEX = 80
RANDOM_SEED = 42

STRUCTURAL_ORDER = [
    "explicit_closed",
    "explicit_opened",
    "remained_closed",
    "remained_active",
    "work_mode_missing",
]

EMPIRICAL_ORDER = [
    "empirical_closed",
    "empirical_opened",
    "remained_active",
    "unclear",
    "censored",
]


def project_paths(root: Path) -> dict[str, Path]:
    out = root / "outputs" / "hex_open_close"
    fig = root / "figures" / "hex_open_close"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return {"out": out, "fig": fig, "root": root}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_applications(), load_hexagons()


def work_mode_semantics(hexagons: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Section 1: inspect work_mode_old / work_mode_new without assumptions."""
    rows = []
    for col in ("work_mode_old", "work_mode_new"):
        s = hexagons[col]
        vc = s.value_counts(dropna=False).rename("n").reset_index()
        vc = vc.rename(columns={vc.columns[0]: "value"})
        vc["field"] = col
        vc["share"] = vc["n"] / len(hexagons)
        rows.append(vc)
        summary = pd.DataFrame(
            [
                {
                    "field": col,
                    "n": int(len(s)),
                    "n_nan": int(s.isna().sum()),
                    "n_unique": int(s.nunique(dropna=True)),
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                    "dtype": str(s.dtype),
                    "unique_values": sorted(s.dropna().unique().tolist()),
                }
            ]
        )
        rows.append(summary.assign(_kind="summary"))

    counts = pd.concat(
        [r for r in rows if "_kind" not in r.columns], ignore_index=True
    )
    summaries = pd.concat(
        [r.drop(columns=["_kind"]) for r in rows if "_kind" in r.columns],
        ignore_index=True,
    )
    examples = hexagons[
        [
            "hex",
            "treatment_date",
            "subject",
            "region_id_old",
            "region_id_new",
            "work_mode_old",
            "work_mode_new",
        ]
    ].copy()
    # Stratified examples by work_mode transitions of interest
    parts = []
    masks = {
        "wm_to_zero": (examples["work_mode_old"] > 0) & (examples["work_mode_new"] == 0),
        "wm_from_zero": (examples["work_mode_old"] == 0) & (examples["work_mode_new"] > 0),
        "wm_both_zero": (examples["work_mode_old"] == 0) & (examples["work_mode_new"] == 0),
        "wm_both_pos": (examples["work_mode_old"] > 0) & (examples["work_mode_new"] > 0),
    }
    for label, m in masks.items():
        sub = examples.loc[m].head(5).assign(example_group=label)
        parts.append(sub)
    examples_out = pd.concat(parts, ignore_index=True) if parts else examples.head(20)

    # Cross-check with region inactivity sentinel used by classify_hexagons
    cross = (
        hexagons.assign(
            region_old_inactive=hexagons["region_id_old"] == INACTIVE_REGION,
            region_new_inactive=hexagons["region_id_new"] == INACTIVE_REGION,
            wm_old_zero=hexagons["work_mode_old"] == 0,
            wm_new_zero=hexagons["work_mode_new"] == 0,
        )
        .groupby(
            ["wm_old_zero", "wm_new_zero", "region_old_inactive", "region_new_inactive"],
            dropna=False,
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return {
        "value_counts": counts,
        "summaries": summaries,
        "examples": examples_out,
        "wm_vs_region_cross": cross,
    }


def classify_structural_work_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Structural classification from work_mode only.
    NaN / unexpected values are not coerced to zero.
    """
    out = df.copy()
    old = out["work_mode_old"]
    new = out["work_mode_new"]

    expected = set(range(8))
    valid = old.notna() & new.notna()
    unexpected = valid & (~old.isin(expected) | ~new.isin(expected))

    cat = pd.Series("work_mode_missing", index=out.index, dtype=object)
    cat.loc[~valid] = "work_mode_missing"
    cat.loc[unexpected] = "work_mode_unexpected"
    usable = valid & ~unexpected

    cat.loc[usable & (old > 0) & (new == 0)] = "explicit_closed"
    cat.loc[usable & (old == 0) & (new > 0)] = "explicit_opened"
    cat.loc[usable & (old == 0) & (new == 0)] = "remained_closed"
    cat.loc[usable & (old > 0) & (new > 0)] = "remained_active"

    out["structural_wm"] = cat
    return out


def prepare_hex_universe(
    applications: pd.DataFrame,
    hexagons: pd.DataFrame,
    *,
    min_orders_per_hex: int | None = None,
    apps_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build unique-hex metadata for diagnostics.

    Parameters
    ----------
    min_orders_per_hex :
        If None, do NOT apply the DiD sample threshold (>=5).
        If int, apply the same filter as build_analysis_panel.
    apps_only :
        Restrict hexagon rows to hexes that appear in applications.
        Full hexagon file has ~3.6M rows; app-linked subset is the relevant universe
        for application-based diagnostics.
    """
    hexagons_cls = classify_hexagons(hexagons)
    if apps_only:
        app_hexes = set(applications["hex"].unique())
        hexagons_cls = hexagons_cls[hexagons_cls["hex"].isin(app_hexes)].copy()

    resolved = resolve_hexagon_treatments(hexagons_cls)
    hex_meta = resolved["single_treatment"].copy()
    hex_meta = classify_structural_work_mode(hex_meta)

    apps = applications.copy()
    apps["request_timestamp"] = pd.to_datetime(apps["request_timestamp"], errors="coerce")
    # Observable application calendar (do not invent days beyond data)
    app_date_min = apps["request_timestamp"].min().normalize()
    app_date_max = apps["request_timestamp"].max().normalize()

    # Optionally clip to study window for consistency with DiD diagnostics,
    # but keep raw span for censoring flags.
    apps_study = apps[
        (apps["request_timestamp"] >= STUDY_START)
        & (apps["request_timestamp"] <= STUDY_END)
    ].copy()

    order_counts = apps_study.groupby("hex").size().rename("n_orders_study")
    hex_meta = hex_meta.merge(order_counts, on="hex", how="left")
    hex_meta["n_orders_study"] = hex_meta["n_orders_study"].fillna(0).astype(int)

    if min_orders_per_hex is not None:
        hex_meta = hex_meta[hex_meta["n_orders_study"] >= min_orders_per_hex].copy()

    # Exclude DiD-excluded late cohort from treated diagnostics by default
    # (keep them labelled for transparency)
    hex_meta["excluded_cohort"] = hex_meta["treatment_date"].isin(EXCLUDED_COHORTS)

    treated = hex_meta[hex_meta["is_treated"].fillna(False)].copy()

    return {
        "hex_meta": hex_meta.reset_index(drop=True),
        "treated_meta": treated.reset_index(drop=True),
        "multi_treatment": resolved["multi_treatment"],
        "app_date_min": app_date_min,
        "app_date_max": app_date_max,
        "apps_study": apps_study,
    }


def build_daily_apps(apps_study: pd.DataFrame) -> pd.DataFrame:
    tmp = apps_study.copy()
    tmp["date"] = pd.to_datetime(tmp["request_timestamp"]).dt.normalize()
    daily = (
        tmp.groupby(["hex", "date"], as_index=False)
        .size()
        .rename(columns={"size": "n_apps"})
    )
    return daily


def build_event_panel(
    treated_meta: pd.DataFrame,
    daily_apps: pd.DataFrame,
    *,
    app_date_min: pd.Timestamp,
    app_date_max: pd.Timestamp,
    pad_days: int = EVENT_PAD_DAYS,
) -> pd.DataFrame:
    """
    hex × calendar_date panel around treatment in [-pad, +pad],
    clipped to observable application dates. Missing hex-days filled with 0.
    """
    meta = treated_meta[
        [
            "hex",
            "treatment_date",
            "structural_wm",
            "change_type",
            "subject",
            "work_mode_old",
            "work_mode_new",
            "region_id_old",
            "region_id_new",
            "n_orders_study",
            "excluded_cohort",
        ]
    ].copy()
    meta["treatment_date"] = pd.to_datetime(meta["treatment_date"])
    meta = meta[meta["treatment_date"].notna()].copy()

    offsets = np.arange(-pad_days, pad_days + 1, dtype=int)
    meta = meta.assign(_k=1)
    off = pd.DataFrame({"rel_day": offsets, "_k": 1})
    panel = meta.merge(off, on="_k", how="inner").drop(columns="_k")
    panel["date"] = panel["treatment_date"] + pd.to_timedelta(panel["rel_day"], unit="D")
    panel = panel[
        (panel["date"] >= app_date_min) & (panel["date"] <= app_date_max)
    ].copy()
    panel["post"] = panel["rel_day"] >= 0

    panel = panel.merge(daily_apps, on=["hex", "date"], how="left")
    panel["n_apps"] = panel["n_apps"].fillna(0).astype(int)
    return panel.reset_index(drop=True)


def _window_mask(rel: pd.Series, start: int, end: int) -> pd.Series:
    return (rel >= start) & (rel <= end)


def _max_consecutive_zeros(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in values:
        if v == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _zero_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, length) for consecutive zero runs."""
    runs: list[tuple[int, int]] = []
    start = None
    length = 0
    for i, v in enumerate(values):
        if v == 0:
            if start is None:
                start = i
                length = 1
            else:
                length += 1
        else:
            if start is not None:
                runs.append((start, length))
            start = None
            length = 0
    if start is not None:
        runs.append((start, length))
    return runs


def compute_window_metrics(
    panel: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Pre/post activity metrics for multiple horizons per hex."""
    rows: list[dict] = []
    for hex_id, g in panel.groupby("hex", sort=False):
        g = g.sort_values("rel_day")
        tdate = g["treatment_date"].iloc[0]
        meta = {
            "hex": hex_id,
            "treatment_date": tdate,
            "structural_wm": g["structural_wm"].iloc[0],
            "change_type": g["change_type"].iloc[0],
            "subject": g["subject"].iloc[0],
            "work_mode_old": g["work_mode_old"].iloc[0],
            "work_mode_new": g["work_mode_new"].iloc[0],
            "n_orders_study": int(g["n_orders_study"].iloc[0]),
            "excluded_cohort": bool(g["excluded_cohort"].iloc[0]),
            "panel_rel_min": int(g["rel_day"].min()),
            "panel_rel_max": int(g["rel_day"].max()),
            "left_censored": bool(g["rel_day"].min() > -EVENT_PAD_DAYS),
            "right_censored": bool(g["rel_day"].max() < EVENT_PAD_DAYS),
        }

        for w in horizons:
            pre = g[_window_mask(g["rel_day"], -w, -1)]
            post = g[_window_mask(g["rel_day"], 0, w - 1)]
            full_pre = len(pre) >= w
            full_post = len(post) >= w

            pre_apps = int(pre["n_apps"].sum()) if len(pre) else 0
            post_apps = int(post["n_apps"].sum()) if len(post) else 0
            pre_active = int((pre["n_apps"] > 0).sum()) if len(pre) else 0
            post_active = int((post["n_apps"] > 0).sum()) if len(post) else 0
            pre_days = int(len(pre))
            post_days = int(len(post))

            meta[f"full_pre_{w}"] = full_pre
            meta[f"full_post_{w}"] = full_post
            meta[f"pre_apps_{w}"] = pre_apps
            meta[f"post_apps_{w}"] = post_apps
            meta[f"pre_active_days_{w}"] = pre_active
            meta[f"post_active_days_{w}"] = post_active
            meta[f"pre_active_share_{w}"] = (
                pre_active / pre_days if pre_days else np.nan
            )
            meta[f"post_active_share_{w}"] = (
                post_active / post_days if post_days else np.nan
            )
            meta[f"pre_avg_apps_day_{w}"] = (
                pre_apps / pre_days if pre_days else np.nan
            )
            meta[f"post_avg_apps_day_{w}"] = (
                post_apps / post_days if post_days else np.nan
            )
            meta[f"pre_median_apps_day_{w}"] = (
                float(pre["n_apps"].median()) if len(pre) else np.nan
            )
            meta[f"post_median_apps_day_{w}"] = (
                float(post["n_apps"].median()) if len(post) else np.nan
            )
            meta[f"pre_max_consecutive_zero_{w}"] = _max_consecutive_zeros(
                pre["n_apps"].to_numpy(dtype=int) if len(pre) else np.array([])
            )
            meta[f"post_max_consecutive_zero_{w}"] = _max_consecutive_zeros(
                post["n_apps"].to_numpy(dtype=int) if len(post) else np.array([])
            )
            # Strict consecutive zero blocks used by empirical rules
            meta[f"pre_all_zero_{w}"] = bool(full_pre and pre_apps == 0)
            meta[f"post_all_zero_{w}"] = bool(full_post and post_apps == 0)
            meta[f"pre_sufficient_activity_{w}"] = bool(
                full_pre
                and pre_apps >= MIN_PRE_APPS
                and pre_active >= MIN_PRE_ACTIVE_DAYS
            )
            meta[f"post_sufficient_activity_{w}"] = bool(
                full_post
                and post_apps >= MIN_PRE_APPS
                and post_active >= MIN_PRE_ACTIVE_DAYS
            )

        # Timing relative to treatment using full available panel
        active = g.loc[g["n_apps"] > 0, "rel_day"]
        if len(active):
            meta["last_active_rel_day"] = int(active.max())
            meta["first_active_rel_day"] = int(active.min())
            meta["last_active_day_pre_or_near"] = int(
                active[active < 0].max() if (active < 0).any() else active.max()
            )
            post_active_rel = active[active >= 0]
            meta["first_active_day_post"] = (
                int(post_active_rel.min()) if len(post_active_rel) else np.nan
            )
        else:
            meta["last_active_rel_day"] = np.nan
            meta["first_active_rel_day"] = np.nan
            meta["last_active_day_pre_or_near"] = np.nan
            meta["first_active_day_post"] = np.nan

        rows.append(meta)
    return pd.DataFrame(rows)


def classify_empirical(
    metrics: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """
    Empirical open/close for a given consecutive-zero horizon.

    empirical_closed_W:
      pre sufficiently active AND post W consecutive calendar days all zero.
    empirical_opened_W:
      pre W consecutive days all zero AND post sufficiently active.
    Censored if required windows are incomplete.
    """
    m = metrics.copy()
    w = horizon
    full_pre = m[f"full_pre_{w}"]
    full_post = m[f"full_post_{w}"]
    closed_ok = (
        full_pre
        & full_post
        & m[f"pre_sufficient_activity_{w}"]
        & m[f"post_all_zero_{w}"]
    )
    opened_ok = (
        full_pre
        & full_post
        & m[f"pre_all_zero_{w}"]
        & m[f"post_sufficient_activity_{w}"]
    )
    # Remained active: pre activity and post not all-zero (and not opened rule)
    remained = (
        full_pre
        & full_post
        & m[f"pre_sufficient_activity_{w}"]
        & ~m[f"post_all_zero_{w}"]
        & ~opened_ok
    )

    status = pd.Series("unclear", index=m.index, dtype=object)
    status.loc[~full_pre | ~full_post] = "censored"
    status.loc[closed_ok] = "empirical_closed"
    status.loc[opened_ok] = "empirical_opened"
    status.loc[remained] = "remained_active"
    # opened_ok takes precedence over remained if both (shouldn't overlap)

    col = f"empirical_{w}"
    m[col] = status
    m[f"empirical_closed_{w}"] = closed_ok
    m[f"empirical_opened_{w}"] = opened_ok
    return m


def classify_permanent_and_temporary(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    horizon: int = PRIMARY_HORIZON,
    min_post_observability: int = 28,
) -> pd.DataFrame:
    """
    Permanent closure / temporary shutdown using post-treatment zero runs.
    Right-censoring: if not enough calendar after treatment, permanent=False/unknown.
    """
    rows = []
    met = metrics.set_index("hex")
    for hex_id, g in panel.groupby("hex", sort=False):
        g = g.sort_values("rel_day")
        row = met.loc[hex_id]
        post = g[g["rel_day"] >= 0].copy()
        pre = g[_window_mask(g["rel_day"], -horizon, -1)]
        max_rel = int(g["rel_day"].max())
        can_claim_permanent = max_rel >= (min_post_observability - 1)

        pret_ok = bool(
            len(pre) >= horizon
            and int(pre["n_apps"].sum()) >= MIN_PRE_APPS
            and int((pre["n_apps"] > 0).sum()) >= MIN_PRE_ACTIVE_DAYS
        )

        post_vals = post["n_apps"].to_numpy(dtype=int)
        runs = _zero_runs(post_vals)
        # longest post zero run
        longest = max((r[1] for r in runs), default=0)
        # zero run starting at day 0 (index 0 of post)
        leading = runs[0][1] if runs and runs[0][0] == 0 else 0

        active_post = post.loc[post["n_apps"] > 0, "rel_day"]
        resumed = False
        if leading >= horizon and len(active_post):
            # activity after a long initial zero run
            resumed = bool(active_post.min() >= horizon)

        # permanent: pret activity, some near-treatment stop, no apps until end
        # of available post panel, and enough post observability
        no_apps_after_start = int(post["n_apps"].sum()) == 0 if len(post) else False
        # allow short spillover: last pre activity near 0, then forever zero from first post zero
        forever_zero_from = None
        if len(post_vals):
            # first index after which ALL remaining are zero and stay zero to end
            # i.e. suffix of zeros
            if post_vals[-1] == 0:
                # find start of final zero suffix
                i = len(post_vals) - 1
                while i >= 0 and post_vals[i] == 0:
                    i -= 1
                forever_zero_from = i + 1  # index in post
            else:
                forever_zero_from = None

        permanent = False
        permanent_status = "not_permanent"
        if not can_claim_permanent:
            permanent_status = "right_censored_cannot_claim"
        elif pret_ok and forever_zero_from is not None:
            suffix_len = len(post_vals) - forever_zero_from
            if forever_zero_from <= horizon and suffix_len >= horizon and no_apps_after_start:
                permanent = True
                permanent_status = "empirical_permanent_closure"
            elif forever_zero_from <= horizon and suffix_len >= horizon:
                # stopped after brief post spillover
                permanent = True
                permanent_status = "empirical_permanent_closure_after_spillover"

        temporary = bool(
            pret_ok
            and leading >= horizon
            and resumed
            and can_claim_permanent
        )

        rows.append(
            {
                "hex": hex_id,
                "structural_wm": row["structural_wm"],
                "change_type": row["change_type"],
                "pretreatment_active": pret_ok,
                "post_observed_days": int(len(post)),
                "max_rel_day": max_rel,
                "can_claim_permanent": can_claim_permanent,
                "leading_post_zero_run": int(leading),
                "longest_post_zero_run": int(longest),
                "n_post_zero_runs": int(len(runs)),
                "resumed_after_zero_run": resumed,
                "empirical_permanent_closure": permanent,
                "permanent_status": permanent_status,
                "temporary_shutdown": temporary,
                "post_total_apps": int(post["n_apps"].sum()) if len(post) else 0,
            }
        )
    return pd.DataFrame(rows)


def lifetime_activity_diagnostics(
    apps_study: pd.DataFrame,
    hex_meta: pd.DataFrame,
    *,
    study_start: pd.Timestamp = STUDY_START,
    study_end: pd.Timestamp = STUDY_END,
) -> pd.DataFrame:
    """
    Full-study-span activity profile per hex (not only around treatment).

    Uses active-day gaps (no full Cartesian panel) for speed.
    Distinguishes never_active / intermittent / closed-like / opened-like.
    """
    daily = build_daily_apps(apps_study)
    daily = daily[daily["hex"].isin(hex_meta["hex"])].copy()
    daily = daily.sort_values(["hex", "date"])

    span_days = int((study_end - study_start).days) + 1
    rows: list[dict] = []
    present = set(daily["hex"].unique())

    for hex_id, g in daily.groupby("hex", sort=False):
        dates = pd.to_datetime(g["date"]).to_numpy()
        apps = g["n_apps"].to_numpy(dtype=int)
        total_apps = int(apps.sum())
        total_active_days = int(len(dates))
        first_d = pd.Timestamp(dates[0])
        last_d = pd.Timestamp(dates[-1])

        if len(dates) == 1:
            longest_between = 0
            n_resume = 0
        else:
            gaps = np.diff(dates.astype("datetime64[D]").astype(int)) - 1
            longest_between = int(gaps.max()) if len(gaps) else 0
            n_resume = int((gaps > 0).sum())

        after_last = int((study_end - last_d).days)
        before_first = int((first_d - study_start).days)

        if longest_between == 0 and after_last == 0 and before_first == 0:
            pattern = "active_throughout"
        elif before_first > 7 and after_last <= 7 and longest_between < 14:
            pattern = "opened_like"
        elif after_last >= 14 and longest_between < after_last:
            pattern = "closed_like"
        elif n_resume >= 2 and longest_between >= 7:
            pattern = "intermittent_or_low_demand"
        elif total_apps < 5:
            pattern = "low_demand"
        else:
            pattern = "temporarily_inactive_or_mixed"

        rows.append(
            {
                "hex": hex_id,
                "first_app_date": first_d,
                "last_app_date": last_d,
                "total_apps": total_apps,
                "total_active_days": total_active_days,
                "longest_zero_run_between_active": longest_between,
                "longest_zero_run_after_last_active": after_last,
                "n_resume_events": n_resume,
                "lifetime_pattern": pattern,
            }
        )

    # Hexes in meta with zero apps in study window
    for hex_id in hex_meta["hex"]:
        if hex_id not in present:
            rows.append(
                {
                    "hex": hex_id,
                    "first_app_date": pd.NaT,
                    "last_app_date": pd.NaT,
                    "total_apps": 0,
                    "total_active_days": 0,
                    "longest_zero_run_between_active": 0,
                    "longest_zero_run_after_last_active": span_days,
                    "n_resume_events": 0,
                    "lifetime_pattern": "never_active",
                }
            )

    out = pd.DataFrame(rows)
    out = out.merge(
        hex_meta[
            [
                "hex",
                "treatment_date",
                "structural_wm",
                "change_type",
                "is_treated",
                "n_orders_study",
            ]
        ],
        on="hex",
        how="left",
    )
    return out


def low_demand_false_closure_risk(
    metrics: pd.DataFrame,
    perm_tmp: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """
    Among hexes that resume activity after treatment, probability of having
    a W-day zero run — stratified by pre-volume. Measures false-closure risk.
    """
    m = metrics.merge(
        perm_tmp[["hex", "leading_post_zero_run", "longest_post_zero_run", "temporary_shutdown", "resumed_after_zero_run"]],
        on="hex",
        how="left",
    )
    # pre volume: use 28-day pre apps if available else study orders
    bins = [-np.inf, 4, 9, 24, 49, np.inf]
    labels = ["1-4", "5-9", "10-24", "25-49", "50+"]
    # For stratification use pre_apps_28 among those with full_pre_28; else n_orders_study
    if "pre_apps_28" in m.columns:
        base_vol = m["pre_apps_28"].where(m.get("full_pre_28", True), m["n_orders_study"])
    else:
        base_vol = m["n_orders_study"]
    m["pre_volume_bin"] = pd.cut(base_vol, bins=bins, labels=labels)

    # Restrict to hexes that eventually resume (have post apps somewhere in panel)
    resumed = m["post_apps_28"] > 0 if "post_apps_28" in m.columns else m["n_orders_study"] > 0

    rows = []
    for w in horizons:
        sub = m.loc[resumed & m[f"full_post_{w}"]].copy()
        sub["had_zero_run_w"] = sub["longest_post_zero_run"] >= w
        for b, g in sub.groupby("pre_volume_bin", observed=False):
            rows.append(
                {
                    "horizon": w,
                    "pre_volume_bin": str(b),
                    "n_hex_resumed": int(len(g)),
                    "n_with_zero_run_ge_w": int(g["had_zero_run_w"].sum()),
                    "share_with_zero_run_ge_w": float(g["had_zero_run_w"].mean())
                    if len(g)
                    else np.nan,
                    "n_temporary_shutdown_flag": int(g["temporary_shutdown"].sum()),
                }
            )
    return pd.DataFrame(rows)


def structural_vs_empirical_table(
    metrics: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    col = f"empirical_{horizon}"
    ct = (
        metrics.groupby(["structural_wm", col], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={col: "empirical"})
    )
    ct["row_pct"] = ct["n"] / ct.groupby("structural_wm")["n"].transform("sum")
    ct["col_pct"] = ct["n"] / ct.groupby("empirical")["n"].transform("sum")
    return ct


def agreement_rate(metrics: pd.DataFrame, horizon: int) -> dict:
    """Agreement between structural wm closed/opened and empirical labels."""
    col = f"empirical_{horizon}"
    m = metrics.copy()
    # Restrict to clear structural closed/opened
    closed_s = m["structural_wm"] == "explicit_closed"
    opened_s = m["structural_wm"] == "explicit_opened"
    n_closed = int(closed_s.sum())
    n_opened = int(opened_s.sum())
    closed_agree = int((closed_s & (m[col] == "empirical_closed")).sum())
    opened_agree = int((opened_s & (m[col] == "empirical_opened")).sum())
    # Hidden closures: remained_active structurally but empirical closed
    hidden = int(
        ((m["structural_wm"] == "remained_active") & (m[col] == "empirical_closed")).sum()
    )
    # wm_new==0 but apps continue: explicit_closed but not empirical_closed and post apps
    wm0_but_apps = int(
        (
            closed_s
            & (m[col] != "empirical_closed")
            & (m[f"post_apps_{horizon}"] > 0)
        ).sum()
    )
    return {
        "horizon": horizon,
        "n_explicit_closed": n_closed,
        "n_explicit_opened": n_opened,
        "n_explicit_closed_empirical_closed": closed_agree,
        "share_closed_confirmed": closed_agree / n_closed if n_closed else np.nan,
        "n_explicit_opened_empirical_opened": opened_agree,
        "share_opened_confirmed": opened_agree / n_opened if n_opened else np.nan,
        "n_hidden_empirical_closed_among_remained_active": hidden,
        "n_explicit_closed_but_post_apps": wm0_but_apps,
        "n_empirical_closed": int((m[col] == "empirical_closed").sum()),
        "n_empirical_opened": int((m[col] == "empirical_opened").sum()),
    }


def activity_change_timing(metrics: pd.DataFrame) -> pd.DataFrame:
    """Timing of last/first active day vs treatment for structural groups."""
    out = metrics[
        [
            "hex",
            "structural_wm",
            "change_type",
            "treatment_date",
            "last_active_day_pre_or_near",
            "first_active_day_post",
            "last_active_rel_day",
            "first_active_rel_day",
        ]
    ].copy()
    # For closures: gap = last active - treatment (usually <=0 if closed at t)
    out["closure_timing_rel"] = out["last_active_day_pre_or_near"]
    out["opening_timing_rel"] = out["first_active_day_post"]
    return out


def suspicious_hexes(metrics: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> pd.DataFrame:
    col = f"empirical_{horizon}"
    m = metrics.copy()
    flags = []

    a = m[
        (m["structural_wm"] == "explicit_closed")
        & (m[col] != "empirical_closed")
        & (m[f"post_apps_{horizon}"] > 0)
    ].copy()
    a["suspicion"] = "wm_closed_but_apps_continue"
    flags.append(a)

    b = m[
        (m["structural_wm"] == "explicit_opened")
        & (m[col] != "empirical_opened")
        & (m[f"pre_apps_{horizon}"] > 0)
    ].copy()
    b["suspicion"] = "wm_opened_but_pre_apps_exist_or_post_weak"
    flags.append(b)

    c = m[
        (m["structural_wm"] == "remained_active") & (m[col] == "empirical_closed")
    ].copy()
    c["suspicion"] = "hidden_empirical_closure"
    flags.append(c)

    d = m[
        (m["work_mode_new"] == 0)
        & (m[f"post_apps_{horizon}"] > 0)
        & (m["structural_wm"].isin(["explicit_closed", "remained_closed"]))
    ].copy()
    d["suspicion"] = "work_mode_new_0_but_post_apps"
    flags.append(d)

    if not flags:
        return pd.DataFrame()
    out = pd.concat(flags, ignore_index=True)
    return out.drop_duplicates(subset=["hex", "suspicion"])


def change_type_vs_structural(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["change_type", "structural_wm"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )


def event_time_aggregates(
    panel: pd.DataFrame,
    groups: Iterable[str] = ("explicit_closed", "explicit_opened", "remained_active"),
    *,
    min_rel: int = -28,
    max_rel: int = 28,
) -> pd.DataFrame:
    sub = panel[
        panel["structural_wm"].isin(list(groups))
        & (panel["rel_day"] >= min_rel)
        & (panel["rel_day"] <= max_rel)
    ].copy()
    sub["active"] = (sub["n_apps"] > 0).astype(int)
    agg = (
        sub.groupby(["structural_wm", "rel_day"], as_index=False)
        .agg(
            mean_n_apps=("n_apps", "mean"),
            active_share=("active", "mean"),
            n_hex=("hex", "nunique"),
            n_cells=("hex", "size"),
        )
        .sort_values(["structural_wm", "rel_day"])
    )
    return agg


def heatmap_matrix(
    panel: pd.DataFrame,
    *,
    structural_group: str,
    mode: str = "closed",
    max_hex: int = HEATMAP_MAX_HEX,
    min_rel: int = -28,
    max_rel: int = 28,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    Return hex × rel_day matrix of active indicators for heatmap.
    Sorted by last active (closed) or first active (opened).
    """
    sub = panel[
        (panel["structural_wm"] == structural_group)
        & (panel["rel_day"] >= min_rel)
        & (panel["rel_day"] <= max_rel)
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.Index([])

    # ranking key
    if mode == "closed":
        key = (
            sub.loc[sub["n_apps"] > 0]
            .groupby("hex")["rel_day"]
            .max()
            .rename("sort_key")
        )
    else:
        key = (
            sub.loc[sub["n_apps"] > 0]
            .groupby("hex")["rel_day"]
            .min()
            .rename("sort_key")
        )
    # volume for sampling
    vol = sub.groupby("hex")["n_apps"].sum().rename("vol")
    ranking = pd.concat([key, vol], axis=1)
    ranking["sort_key"] = ranking["sort_key"].fillna(99 if mode == "closed" else -99)

    if len(ranking) > max_hex:
        # prefer higher pre-volume around treatment
        top = ranking.nlargest(max_hex, "vol")
        ranking = top

    ranking = ranking.sort_values("sort_key")
    hex_order = ranking.index

    mat = (
        sub[sub["hex"].isin(hex_order)]
        .pivot_table(index="hex", columns="rel_day", values="n_apps", aggfunc="sum")
        .reindex(hex_order)
        .fillna(0)
    )
    return mat, hex_order


def threshold_sensitivity_summary(metrics: pd.DataFrame, horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    rows = []
    status_by_h = {}
    for h in horizons:
        col = f"empirical_{h}"
        status_by_h[h] = metrics.set_index("hex")[col]
        vc = metrics[col].value_counts()
        agr = agreement_rate(metrics, h)
        rows.append(
            {
                **agr,
                "n_unclear": int(vc.get("unclear", 0)),
                "n_censored": int(vc.get("censored", 0)),
                "n_remained_active_emp": int(vc.get("remained_active", 0)),
            }
        )
    sens = pd.DataFrame(rows)

    # classification changes across thresholds among non-censored
    primary = status_by_h[PRIMARY_HORIZON]
    change_rows = []
    for h, series in status_by_h.items():
        common = primary.index.intersection(series.index)
        # comparable where neither censored
        mask = (primary.loc[common] != "censored") & (series.loc[common] != "censored")
        idx = common[mask]
        n_change = int((primary.loc[idx] != series.loc[idx]).sum())
        change_rows.append(
            {
                "horizon": h,
                "n_comparable_vs_primary14": int(len(idx)),
                "n_status_changes_vs_primary14": n_change,
                "share_status_changes_vs_primary14": n_change / len(idx) if len(idx) else np.nan,
            }
        )
    sens = sens.merge(pd.DataFrame(change_rows), on="horizon", how="left")
    return sens


def build_summary_table(
    structural_counts: pd.DataFrame,
    sens: pd.DataFrame,
    perm_tmp: pd.DataFrame,
    *,
    n_treated: int,
    min_orders_note: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_treated_hex_diagnosed": n_treated,
                "min_orders_filter": min_orders_note,
                "n_explicit_closed": int(
                    structural_counts.loc[
                        structural_counts["structural_wm"] == "explicit_closed", "n"
                    ].sum()
                )
                if len(structural_counts)
                else 0,
                "n_explicit_opened": int(
                    structural_counts.loc[
                        structural_counts["structural_wm"] == "explicit_opened", "n"
                    ].sum()
                )
                if len(structural_counts)
                else 0,
                "n_empirical_closed_14": int(
                    sens.loc[sens["horizon"] == 14, "n_empirical_closed"].iloc[0]
                )
                if (sens["horizon"] == 14).any()
                else np.nan,
                "n_empirical_opened_14": int(
                    sens.loc[sens["horizon"] == 14, "n_empirical_opened"].iloc[0]
                )
                if (sens["horizon"] == 14).any()
                else np.nan,
                "share_closed_confirmed_14": float(
                    sens.loc[sens["horizon"] == 14, "share_closed_confirmed"].iloc[0]
                )
                if (sens["horizon"] == 14).any()
                else np.nan,
                "share_opened_confirmed_14": float(
                    sens.loc[sens["horizon"] == 14, "share_opened_confirmed"].iloc[0]
                )
                if (sens["horizon"] == 14).any()
                else np.nan,
                "n_temporary_shutdown": int(perm_tmp["temporary_shutdown"].sum()),
                "n_permanent_closure": int(perm_tmp["empirical_permanent_closure"].sum()),
                "primary_horizon": PRIMARY_HORIZON,
                "admin_closed_rule": "region_id_new == -100 (classify_hexagons)",
                "admin_opened_rule": "region_id_old == -100 & region_id_new != -100",
            }
        ]
    )
