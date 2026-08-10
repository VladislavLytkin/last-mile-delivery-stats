"""
Диагностика фактической активности и эмпирического закрытия гексагонов.

Полная панель hex × date включает дни без заявок и используется только
для диагностики активности. Нулевые дни не интерпретируются как нулевые
конверсии и не подменяют основную конверсионную панель DiD.

Эмпирическая классификация закрытия — post-treatment диагностика и не
заменяет административный change_type / treatment assignment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .filter import (
    CLOSED,
    CORE_CHANGE_TYPES,
    EXCLUDED_COHORTS,
    STUDY_END,
    STUDY_START,
    attach_hex_metadata,
    classify_hexagons,
    exclude_cohorts,
    filter_study_window,
    get_cohort_map,
    resolve_hexagon_treatments,
)
from .io import load_applications, load_hexagons

# Diagnostic thresholds (not universal business rules).
PRIMARY_WINDOW_DAYS = 28
MIN_PRE_ORDERS = 5
MIN_PRE_ACTIVE_DAYS = 3
NEAR_CLOSED_RATIO = 0.05
MAX_NEAR_CLOSED_ACTIVE_DAYS = 1

EMPIRICAL_CLOSED = "empirical_closed"
NEAR_CLOSED = "near_closed"
SPARSE_PRE_ACTIVITY = "sparse_pre_activity"
CONTINUES_ACTIVE = "continues_active"
INSUFFICIENT_POST_WINDOW = "insufficient_post_window"

META_COLS = [
    "treatment_date",
    "change_type",
    "is_treated",
    "subject",
    "region_id_old",
    "region_id_new",
    "work_mode_old",
    "work_mode_new",
    "cohort",
]

__all__ = [
    "PRIMARY_WINDOW_DAYS",
    "MIN_PRE_ORDERS",
    "MIN_PRE_ACTIVE_DAYS",
    "NEAR_CLOSED_RATIO",
    "MAX_NEAR_CLOSED_ACTIVE_DAYS",
    "EMPIRICAL_CLOSED",
    "NEAR_CLOSED",
    "SPARSE_PRE_ACTIVITY",
    "CONTINUES_ACTIVE",
    "INSUFFICIENT_POST_WINDOW",
    "build_hex_activity_panel",
    "validate_hex_activity_panel",
    "compute_closure_window_metrics",
    "classify_empirical_activity",
    "run_closure_threshold_sensitivity",
    "build_metadata_vs_empirical_crosstab",
    "summarize_comparison_groups",
    "build_event_time_activity",
    "prepare_analysis_hex_universe",
]


def prepare_analysis_hex_universe(
    applications: pd.DataFrame,
    hexagons: pd.DataFrame,
    *,
    min_orders_per_hex: int = 5,
    excluded_cohorts: Optional[set[pd.Timestamp]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Готовит единый набор анализируемых гексагонов (как в build_analysis_panel).

    Multi-treatment исключаются явно через resolve_hexagon_treatments.
    Возвращает hex-метаданные (1 строка на hex), заявки в окне и счётчики.
    """
    if excluded_cohorts is None:
        excluded_cohorts = EXCLUDED_COHORTS

    hexagons_cls = classify_hexagons(hexagons)
    resolved = resolve_hexagon_treatments(hexagons_cls)
    hexagons_single = resolved["single_treatment"]
    cohort_map = get_cohort_map(hexagons_single)
    if not cohort_map["hex"].is_unique:
        raise RuntimeError("prepare_analysis_hex_universe: hex не уникален")

    apps_windowed = filter_study_window(applications)
    apps_meta = attach_hex_metadata(apps_windowed, cohort_map)
    apps_clean = exclude_cohorts(apps_meta, excluded=excluded_cohorts)

    hex_order_counts = apps_clean.groupby("hex")["request_timestamp"].count()
    viable_hexes = hex_order_counts[hex_order_counts >= min_orders_per_hex].index
    apps_clean = apps_clean[apps_clean["hex"].isin(viable_hexes)].copy()

    meta_cols = ["hex"] + [c for c in META_COLS if c in hexagons_single.columns]
    # cohort уже есть в classify; treatment_date — исходная дата воздействия
    if "treatment_date" not in hexagons_single.columns and "cohort" in hexagons_single.columns:
        hexagons_single = hexagons_single.copy()
        hexagons_single["treatment_date"] = hexagons_single["cohort"]

    hex_meta = hexagons_single.loc[
        hexagons_single["hex"].isin(viable_hexes), meta_cols
    ].copy()
    if not hex_meta["hex"].is_unique:
        raise RuntimeError(
            "prepare_analysis_hex_universe: после фильтра остались дубли hex. "
            "Проверьте multi-treatment resolution."
        )

    return {
        "hex_meta": hex_meta.reset_index(drop=True),
        "apps_clean": apps_clean,
        "cohort_map": cohort_map,
        "multi_treatment": resolved["multi_treatment"],
        "treatment_diagnostics": resolved["diagnostics"],
        "hex_order_counts": hex_order_counts.loc[viable_hexes].rename("n_orders_total"),
    }


def build_hex_activity_panel(
    applications: pd.DataFrame | None = None,
    hexagons: pd.DataFrame | None = None,
    *,
    applications_path: str | Path | None = None,
    hexagons_path: str | Path | None = None,
    min_orders_per_hex: int = 5,
    study_start: pd.Timestamp = STUDY_START,
    study_end: pd.Timestamp = STUDY_END,
    excluded_cohorts: Optional[set[pd.Timestamp]] = None,
    hex_meta: pd.DataFrame | None = None,
    apps_clean: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Строит полную панель активности: все анализируемые hex × все календарные дни.

    Дни без заявок присутствуют со значением n_orders = 0.
    Конверсионные исходы сюда не добавляются.
    """
    if hex_meta is None or apps_clean is None:
        if applications is None:
            applications = load_applications(applications_path)
        if hexagons is None:
            hexagons = load_hexagons(hexagons_path)
        prepared = prepare_analysis_hex_universe(
            applications,
            hexagons,
            min_orders_per_hex=min_orders_per_hex,
            excluded_cohorts=excluded_cohorts,
        )
        hex_meta = prepared["hex_meta"]
        apps_clean = prepared["apps_clean"]

    hex_meta = hex_meta.copy()
    hex_meta["treatment_date"] = pd.to_datetime(hex_meta["treatment_date"], errors="coerce")
    if "cohort" not in hex_meta.columns:
        hex_meta["cohort"] = hex_meta["treatment_date"]

    dates = pd.date_range(study_start, study_end, freq="D")
    hex_ids = hex_meta["hex"].unique()
    grid = pd.MultiIndex.from_product(
        [hex_ids, dates], names=["hex", "date"]
    ).to_frame(index=False)

    apps = apps_clean.copy()
    apps["date"] = pd.to_datetime(apps["request_timestamp"]).dt.normalize()
    daily = (
        apps.groupby(["hex", "date"], as_index=False)
        .size()
        .rename(columns={"size": "n_orders"})
    )

    panel = grid.merge(daily, on=["hex", "date"], how="left")
    panel["n_orders"] = panel["n_orders"].fillna(0).astype(int)

    attach_cols = ["hex"] + [c for c in META_COLS if c in hex_meta.columns]
    panel = panel.merge(hex_meta[attach_cols], on="hex", how="left", validate="m:1")

    validate_hex_activity_panel(panel, apps_clean)
    return panel


def validate_hex_activity_panel(
    panel: pd.DataFrame,
    apps_clean: pd.DataFrame,
) -> None:
    """Проверяет целостность полной панели активности."""
    if panel.duplicated(subset=["hex", "date"]).any():
        raise ValueError("hex_activity_panel: пара (hex, date) не уникальна")
    if panel["n_orders"].isna().any():
        raise ValueError("hex_activity_panel: пропуски в n_orders")
    if (panel["n_orders"] < 0).any():
        raise ValueError("hex_activity_panel: отрицательные n_orders")
    if (panel["n_orders"] == 0).sum() == 0:
        raise ValueError("hex_activity_panel: нет нулевых дней — панель неполная")

    apps = apps_clean.copy()
    apps["date"] = pd.to_datetime(apps["request_timestamp"]).dt.normalize()
    date_min = pd.to_datetime(panel["date"]).min()
    date_max = pd.to_datetime(panel["date"]).max()
    hex_set = set(panel["hex"])
    apps_in = apps[
        apps["hex"].isin(hex_set)
        & (apps["date"] >= date_min)
        & (apps["date"] <= date_max)
    ]
    n_apps = len(apps_in)
    n_panel = int(panel["n_orders"].sum())
    if n_panel != n_apps:
        raise ValueError(
            f"hex_activity_panel: сумма n_orders={n_panel} != числу заявок "
            f"в окне панели={n_apps}"
        )

    # Нет размножения заявок: daily counts должны совпадать с исходными.
    daily_src = apps_in.groupby(["hex", "date"]).size().astype(int)
    daily_panel = (
        panel.loc[panel["n_orders"] > 0]
        .set_index(["hex", "date"])["n_orders"]
        .astype(int)
        .sort_index()
    )
    daily_src = daily_src.reindex(daily_panel.index).astype(int)
    if not np.array_equal(daily_src.to_numpy(), daily_panel.to_numpy()):
        raise ValueError("hex_activity_panel: расхождение daily counts после merge")


def _safe_ratio(numer: float, denom: float) -> float:
    """Безопасное отношение: NaN при нулевом знаменателе (не подменять нулём)."""
    if denom is None or not np.isfinite(denom) or denom == 0:
        return np.nan
    if numer is None or not np.isfinite(numer):
        return np.nan
    return float(numer) / float(denom)


def _max_consecutive_zeros(values: np.ndarray) -> int:
    """Максимальная длина подряд идущих нулей."""
    best = 0
    cur = 0
    for v in values:
        if v == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def compute_closure_window_metrics(
    activity_panel: pd.DataFrame,
    *,
    window_days: int,
    treated_only: bool = True,
) -> pd.DataFrame:
    """
    Для каждого treated-гексагона считает pre/post метрики в симметричном окне.

    Окно обрезается границами доступного периода наблюдения в панели.
    """
    df = activity_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["treatment_date"] = pd.to_datetime(df["treatment_date"], errors="coerce")

    if treated_only:
        df = df[df["is_treated"].fillna(False)].copy()

    rows: list[dict] = []
    for hex_id, g in df.groupby("hex", sort=False):
        g = g.sort_values("date")
        tdate = g["treatment_date"].iloc[0]
        if pd.isna(tdate):
            continue

        pre_start = tdate - pd.Timedelta(days=window_days)
        pre_end = tdate - pd.Timedelta(days=1)
        post_start = tdate
        post_end = tdate + pd.Timedelta(days=window_days - 1)

        pre = g[(g["date"] >= pre_start) & (g["date"] <= pre_end)]
        post = g[(g["date"] >= post_start) & (g["date"] <= post_end)]

        pre_orders = int(pre["n_orders"].sum()) if len(pre) else 0
        post_orders = int(post["n_orders"].sum()) if len(post) else 0
        pre_obs = int(len(pre))
        post_obs = int(len(post))
        pre_active = int((pre["n_orders"] > 0).sum()) if len(pre) else 0
        post_active = int((post["n_orders"] > 0).sum()) if len(post) else 0

        active_dates = g.loc[g["n_orders"] > 0, "date"]
        last_req = active_dates.max() if len(active_dates) else pd.NaT
        post_active_dates = post.loc[post["n_orders"] > 0, "date"]
        first_post = post_active_dates.min() if len(post_active_dates) else pd.NaT

        full_post_window = post_obs >= window_days
        sufficient_pre = (pre_orders >= MIN_PRE_ORDERS) and (
            pre_active >= MIN_PRE_ACTIVE_DAYS
        )

        rows.append(
            {
                "hex": hex_id,
                "window_days": int(window_days),
                "treatment_date": tdate,
                "change_type": g["change_type"].iloc[0],
                "subject": g["subject"].iloc[0] if "subject" in g.columns else np.nan,
                "cohort": g["cohort"].iloc[0] if "cohort" in g.columns else tdate,
                "pre_observed_days": pre_obs,
                "post_observed_days": post_obs,
                "pre_orders": pre_orders,
                "post_orders": post_orders,
                "pre_mean_orders_per_day": (
                    pre_orders / pre_obs if pre_obs > 0 else np.nan
                ),
                "post_mean_orders_per_day": (
                    post_orders / post_obs if post_obs > 0 else np.nan
                ),
                "pre_active_days": pre_active,
                "post_active_days": post_active,
                "pre_zero_share": (
                    1.0 - pre_active / pre_obs if pre_obs > 0 else np.nan
                ),
                "post_zero_share": (
                    1.0 - post_active / post_obs if post_obs > 0 else np.nan
                ),
                "post_to_pre_orders_ratio": _safe_ratio(post_orders, pre_orders),
                "last_request_date": last_req,
                "last_request_relative_day": (
                    (last_req - tdate).days if pd.notna(last_req) else np.nan
                ),
                "first_post_request_date": first_post,
                "first_post_request_relative_day": (
                    (first_post - tdate).days if pd.notna(first_post) else np.nan
                ),
                "max_consecutive_zero_days_post": _max_consecutive_zeros(
                    post["n_orders"].to_numpy(dtype=int) if len(post) else np.array([])
                ),
                "full_post_window": bool(full_post_window),
                "sufficient_pre_activity": bool(sufficient_pre),
            }
        )

    return pd.DataFrame(rows)


def classify_empirical_activity(
    metrics: pd.DataFrame,
    *,
    window_days: int = PRIMARY_WINDOW_DAYS,
    min_pre_orders: int = MIN_PRE_ORDERS,
    min_pre_active_days: int = MIN_PRE_ACTIVE_DAYS,
    near_closed_ratio: float = NEAR_CLOSED_RATIO,
    max_near_closed_active_days: int = MAX_NEAR_CLOSED_ACTIVE_DAYS,
) -> pd.DataFrame:
    """
    Прозрачная диагностическая классификация эмпирической активности.

    Не заменяет административный change_type.
    """
    m = metrics[metrics["window_days"] == window_days].copy()
    if m.empty:
        raise ValueError(f"Нет метрик для window_days={window_days}")

    statuses: list[str] = []
    for _, row in m.iterrows():
        full_post = bool(row["full_post_window"]) and int(row["post_observed_days"]) >= window_days
        suff_pre = (
            int(row["pre_orders"]) >= min_pre_orders
            and int(row["pre_active_days"]) >= min_pre_active_days
        )
        post_orders = int(row["post_orders"])
        ratio = row["post_to_pre_orders_ratio"]
        post_active = int(row["post_active_days"])

        if not full_post:
            status = INSUFFICIENT_POST_WINDOW
        elif not suff_pre:
            status = SPARSE_PRE_ACTIVITY
        elif post_orders == 0:
            status = EMPIRICAL_CLOSED
        elif (
            pd.notna(ratio)
            and ratio <= near_closed_ratio
            and post_active <= max_near_closed_active_days
            and post_orders > 0
        ):
            status = NEAR_CLOSED
        else:
            status = CONTINUES_ACTIVE
        statuses.append(status)

    m["empirical_status"] = statuses
    m["class_window_days"] = window_days
    m["class_min_pre_orders"] = min_pre_orders
    m["class_min_pre_active_days"] = min_pre_active_days
    m["class_near_closed_ratio"] = near_closed_ratio
    m["class_max_near_closed_active_days"] = max_near_closed_active_days
    return m.reset_index(drop=True)


def run_closure_threshold_sensitivity(
    metrics_by_window: pd.DataFrame,
    *,
    window_grid: Sequence[int] = (14, 28, 56),
    min_pre_orders_grid: Sequence[int] = (3, 5, 10),
    near_closed_ratio_grid: Sequence[float] = (0.0, 0.05, 0.10),
) -> pd.DataFrame:
    """
    Сетка чувствительности диагностической классификации.

    Для каждой комбинации параметров считает число гексагонов по статусам.
    """
    rows: list[dict] = []
    status_by_key: dict[tuple, pd.Series] = {}

    for w in window_grid:
        base = metrics_by_window[metrics_by_window["window_days"] == w]
        if base.empty:
            continue
        for min_pre in min_pre_orders_grid:
            for ratio in near_closed_ratio_grid:
                cls = classify_empirical_activity(
                    metrics_by_window,
                    window_days=w,
                    min_pre_orders=min_pre,
                    near_closed_ratio=ratio,
                )
                key = (w, min_pre, ratio)
                status_by_key[key] = cls.set_index("hex")["empirical_status"]
                counts = cls["empirical_status"].value_counts()
                row = {
                    "window_days": w,
                    "min_pre_orders": min_pre,
                    "near_closed_ratio": ratio,
                    "n_hexagons": int(cls["hex"].nunique()),
                    "n_empirical_closed": int(counts.get(EMPIRICAL_CLOSED, 0)),
                    "n_near_closed": int(counts.get(NEAR_CLOSED, 0)),
                    "n_continues_active": int(counts.get(CONTINUES_ACTIVE, 0)),
                    "n_sparse_pre_activity": int(counts.get(SPARSE_PRE_ACTIVITY, 0)),
                    "n_insufficient_post_window": int(
                        counts.get(INSUFFICIENT_POST_WINDOW, 0)
                    ),
                }
                rows.append(row)

    sens = pd.DataFrame(rows)

    # Pairwise agreement of empirical_closed labels across primary window variants
    primary = status_by_key.get((PRIMARY_WINDOW_DAYS, MIN_PRE_ORDERS, NEAR_CLOSED_RATIO))
    if primary is not None and not sens.empty:
        agree_cols = []
        for key, series in status_by_key.items():
            if key == (PRIMARY_WINDOW_DAYS, MIN_PRE_ORDERS, NEAR_CLOSED_RATIO):
                continue
            common = primary.index.intersection(series.index)
            if len(common) == 0:
                continue
            agree = (
                (primary.loc[common] == EMPIRICAL_CLOSED)
                == (series.loc[common] == EMPIRICAL_CLOSED)
            ).mean()
            agree_cols.append(
                {
                    "compare_window_days": key[0],
                    "compare_min_pre_orders": key[1],
                    "compare_near_closed_ratio": key[2],
                    "agreement_empirical_closed": float(agree),
                    "n_status_changes_vs_primary": int(
                        (primary.loc[common] != series.loc[common]).sum()
                    ),
                }
            )
        sens.attrs["agreement"] = pd.DataFrame(agree_cols)

    return sens


def build_metadata_vs_empirical_crosstab(
    classification: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tab административного change_type и эмпирического статуса."""
    ct = (
        classification.groupby(["change_type", "empirical_status"], dropna=False)
        .size()
        .rename("n_hexagons")
        .reset_index()
    )
    return ct.sort_values(["change_type", "empirical_status"]).reset_index(drop=True)


def summarize_comparison_groups(
    classification: pd.DataFrame,
    activity_panel: pd.DataFrame,
    *,
    example_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """
    Четыре ключевые группы сопоставления metadata vs empirical.

    1) closed ∩ empirical_closed
    2) closed, но активность продолжается
    3) CORE ∩ empirical_closed
    4) CORE ∩ continues_active
    """
    cls = classification.copy()
    panel = activity_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["treatment_date"] = pd.to_datetime(panel["treatment_date"], errors="coerce")

    def _group_summary(mask: pd.Series, label: str) -> pd.DataFrame:
        sub = cls.loc[mask].copy()
        hexes = set(sub["hex"])
        if not hexes:
            return pd.DataFrame(
                [
                    {
                        "group": label,
                        "n_hexagons": 0,
                        "pre_orders": 0,
                        "post_orders": 0,
                        "cohort_distribution": "",
                        "change_type_distribution": "",
                        "example_hexes": "",
                    }
                ]
            )

        g_panel = panel[panel["hex"].isin(hexes)].copy()
        pre_orders = int(
            g_panel.loc[g_panel["date"] < g_panel["treatment_date"], "n_orders"].sum()
        )
        post_orders = int(
            g_panel.loc[g_panel["date"] >= g_panel["treatment_date"], "n_orders"].sum()
        )
        cohort_dist = (
            sub.groupby(sub["treatment_date"].astype(str))["hex"]
            .nunique()
            .sort_index()
            .to_dict()
        )
        type_dist = sub.groupby("change_type")["hex"].nunique().to_dict()
        examples = ",".join(map(str, sub["hex"].head(example_n).tolist()))
        return pd.DataFrame(
            [
                {
                    "group": label,
                    "n_hexagons": int(sub["hex"].nunique()),
                    "pre_orders": pre_orders,
                    "post_orders": post_orders,
                    "cohort_distribution": str(cohort_dist),
                    "change_type_distribution": str(type_dist),
                    "example_hexes": examples,
                }
            ]
        )

    is_closed = cls["change_type"] == CLOSED
    is_core = cls["change_type"].isin(CORE_CHANGE_TYPES)
    is_emp_closed = cls["empirical_status"] == EMPIRICAL_CLOSED
    is_continues = cls["empirical_status"] == CONTINUES_ACTIVE
    # «заявки продолжаются» для admin-closed: не empirical_closed при наличии post_orders
    admin_still_active = is_closed & (cls["post_orders"] > 0)

    summaries = pd.concat(
        [
            _group_summary(is_closed & is_emp_closed, "admin_closed_and_empirical_closed"),
            _group_summary(admin_still_active, "admin_closed_but_orders_continue"),
            _group_summary(is_core & is_emp_closed, "core_and_empirical_closed"),
            _group_summary(is_core & is_continues, "core_and_continues_active"),
        ],
        ignore_index=True,
    )

    core_emp = cls.loc[is_core & is_emp_closed].copy()
    admin_active = cls.loc[admin_still_active].copy()

    return {
        "group_summary": summaries,
        "core_empirically_closed": core_emp.reset_index(drop=True),
        "administratively_closed_but_active": admin_active.reset_index(drop=True),
    }


def build_event_time_activity(
    activity_panel: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    min_rel: int = -56,
    max_rel: int = 56,
    bootstrap_reps: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Event-time агрегаты активности по группам (mean/median orders, active share, CI).

    Группы: administratively closed, CORE, empirical_closed, continues_active.
    """
    panel = activity_panel.copy()
    panel = panel[panel["is_treated"].fillna(False)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["treatment_date"] = pd.to_datetime(panel["treatment_date"], errors="coerce")
    panel["relative_day"] = (panel["date"] - panel["treatment_date"]).dt.days
    panel = panel[
        (panel["relative_day"] >= min_rel) & (panel["relative_day"] <= max_rel)
    ].copy()
    panel["active_day"] = (panel["n_orders"] > 0).astype(int)

    cls = classification[["hex", "empirical_status"]].drop_duplicates(subset=["hex"])
    panel = panel.merge(cls, on="hex", how="left")

    group_defs = {
        "admin_closed": panel["change_type"] == CLOSED,
        "core": panel["change_type"].isin(CORE_CHANGE_TYPES),
        "empirical_closed": panel["empirical_status"] == EMPIRICAL_CLOSED,
        "continues_active": panel["empirical_status"] == CONTINUES_ACTIVE,
    }

    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for gname, mask in group_defs.items():
        sub = panel.loc[mask]
        if sub.empty:
            continue
        hex_ids = sub["hex"].unique()
        for rel_day, g in sub.groupby("relative_day"):
            mean_orders = float(g["n_orders"].mean())
            median_orders = float(g["n_orders"].median())
            active_share = float(g["active_day"].mean())
            n_hex = int(g["hex"].nunique())

            # Bootstrap CI by hexagon: resample hexes, then mean active_day
            if bootstrap_reps > 0 and n_hex >= 2:
                hex_day = (
                    g.groupby("hex")["active_day"].mean().reindex(hex_ids).fillna(0.0)
                )
                boots = []
                for _ in range(bootstrap_reps):
                    sample = rng.choice(hex_day.to_numpy(), size=len(hex_day), replace=True)
                    boots.append(float(np.mean(sample)))
                lo, hi = np.quantile(boots, [0.025, 0.975])
            else:
                lo, hi = np.nan, np.nan

            rows.append(
                {
                    "group": gname,
                    "relative_day": int(rel_day),
                    "mean_n_orders": mean_orders,
                    "median_n_orders": median_orders,
                    "active_share": active_share,
                    "active_share_ci_low": float(lo),
                    "active_share_ci_high": float(hi),
                    "n_observed_hexagons": n_hex,
                }
            )

    return pd.DataFrame(rows).sort_values(["group", "relative_day"]).reset_index(drop=True)


def selective_attrition_diagnostics(
    activity_panel: pd.DataFrame,
    *,
    window_days: int = PRIMARY_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Риск селективного исчезновения treated из заявочной панели.

    Для treated: доля гексагонов с post_orders==0 в окне после treatment.
    Для control: календарные «псевдо-окна» по датам когорт treated
    (фиксированное правило: каждая never-treated строка оценивается
    относительно каждой cohort date, затем усредняется по когортам).
    """
    panel = activity_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["treatment_date"] = pd.to_datetime(panel["treatment_date"], errors="coerce")

    treated = panel[panel["is_treated"].fillna(False)].copy()
    control = panel[~panel["is_treated"].fillna(False)].copy()

    cohort_dates = sorted(treated["treatment_date"].dropna().unique())
    rows: list[dict] = []

    # Treated by change_type / cohort
    for keys, g in treated.groupby(["change_type", "treatment_date"], dropna=False):
        change_type, tdate = keys
        post = g[
            (g["date"] >= tdate)
            & (g["date"] <= tdate + pd.Timedelta(days=window_days - 1))
        ]
        post_sum = post.groupby("hex")["n_orders"].sum()
        n_hex = int(post_sum.shape[0])
        n_zero = int((post_sum == 0).sum())
        rows.append(
            {
                "sample": "treated",
                "change_type": change_type,
                "cohort": tdate,
                "window_days": window_days,
                "n_hexagons": n_hex,
                "n_zero_post_orders": n_zero,
                "share_zero_post_orders": n_zero / n_hex if n_hex else np.nan,
                "rule": "relative_to_own_treatment_date",
            }
        )

    # Control aligned to each cohort date (transparent, non-random)
    for tdate in cohort_dates:
        post = control[
            (control["date"] >= tdate)
            & (control["date"] <= tdate + pd.Timedelta(days=window_days - 1))
        ]
        post_sum = post.groupby("hex")["n_orders"].sum()
        n_hex = int(post_sum.shape[0])
        n_zero = int((post_sum == 0).sum())
        rows.append(
            {
                "sample": "control_cohort_aligned",
                "change_type": "never_treated",
                "cohort": tdate,
                "window_days": window_days,
                "n_hexagons": n_hex,
                "n_zero_post_orders": n_zero,
                "share_zero_post_orders": n_zero / n_hex if n_hex else np.nan,
                "rule": "aligned_to_treated_cohort_date",
            }
        )

    return pd.DataFrame(rows)


def save_activity_panel(
    panel: pd.DataFrame,
    out_dir: str | Path,
    *,
    max_csv_rows: int = 2_500_000,
) -> dict[str, Path]:
    """Сохраняет панель: CSV если размер приемлем, иначе parquet + CSV-summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    summary = (
        panel.groupby("hex", as_index=False)
        .agg(
            n_days=("date", "count"),
            n_orders=("n_orders", "sum"),
            n_zero_days=("n_orders", lambda s: int((s == 0).sum())),
            n_active_days=("n_orders", lambda s: int((s > 0).sum())),
            change_type=("change_type", "first"),
            is_treated=("is_treated", "first"),
            treatment_date=("treatment_date", "first"),
        )
    )
    summary_path = out_dir / "hex_activity_daily_panel_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    if len(panel) <= max_csv_rows:
        csv_path = out_dir / "hex_activity_daily_panel.csv"
        panel.to_csv(csv_path, index=False)
        paths["panel"] = csv_path
    else:
        pq_path = out_dir / "hex_activity_daily_panel.parquet"
        try:
            panel.to_parquet(pq_path, index=False)
            paths["panel"] = pq_path
        except Exception:
            # Fallback without pyarrow: chunked CSV of non-zero + summary only
            nz = panel[panel["n_orders"] > 0]
            nz_path = out_dir / "hex_activity_daily_panel_nonzero_days.csv"
            nz.to_csv(nz_path, index=False)
            paths["panel_nonzero"] = nz_path

    return paths
