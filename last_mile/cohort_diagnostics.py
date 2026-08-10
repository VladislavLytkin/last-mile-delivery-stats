"""
last_mile/cohort_diagnostics.py

Диагностика состава когорт по типам изменения логистической конфигурации
(change_type) — предоценочная проверка перед запуском event-study DiD.

Исключения по исходам:
  - opened / closed → structurally invalid для DiD, исключаются всегда
  - когорта 2022-07-27:
      * сохраняется для sch_flg
      * исключается для meet_flg, success_flg и 25-дневной утилизации
        (utlz_within_25); для meet_flg исключение связано с нарушением
        parallel trends (p ≈ 0.0168)

Явные ключи отчёта:
  composition_sch, composition_post_meeting, composition_meet,
  composition_success, composition_utilization

Алиас composition_conversion (= composition_post_meeting) сохранён
для обратной совместимости со старыми ноутбуками. Название «conversion»
не означает, что исключение относится только к success/utilization.
"""

from __future__ import annotations

import pandas as pd

from .filter import CORE_CHANGE_TYPES

# Когорта с нарушением parallel trends для post-meeting исходов.
# Для sch_flg остаётся валидной. Для meet_flg: pretrend p ≈ 0.0168.
EXCLUDED_COHORT_FOR_CONVERSION = pd.Timestamp("2022-07-27")
EXCLUDED_COHORT_FOR_POST_MEETING = EXCLUDED_COHORT_FOR_CONVERSION
EXCLUDED_COHORT_FOR_MEET = EXCLUDED_COHORT_FOR_CONVERSION
EXCLUDED_COHORT_FOR_SUCCESS = EXCLUDED_COHORT_FOR_CONVERSION
EXCLUDED_COHORT_FOR_UTILIZATION = EXCLUDED_COHORT_FOR_CONVERSION

MIN_HEXAGONS_THRESHOLD = 5


def _pivot_hex_counts(hex_level: pd.DataFrame) -> pd.DataFrame:
    if hex_level.empty:
        return pd.DataFrame()
    return (
        hex_level.groupby(["cohort", "change_type"])["hex"]
        .nunique()
        .unstack(fill_value=0)
        .astype(int)
    )


def _ensure_order_cohort_columns(
    orders: pd.DataFrame,
    cohort_map: pd.DataFrame,
) -> pd.DataFrame:
    if "hex" not in orders.columns:
        raise ValueError(
            "treated_orders должен содержать колонку 'hex', "
            "чтобы присоединить cohort/change_type из cohort_map."
        )

    needed_cols = [
        col for col in ["cohort", "change_type"] if col not in orders.columns
    ]
    if not needed_cols:
        return orders

    cohort_lookup = cohort_map[["hex", *needed_cols]].drop_duplicates(subset=["hex"])
    return orders.merge(cohort_lookup, on="hex", how="left")


def _pivot_order_counts(
    orders: pd.DataFrame,
    cohort_map: pd.DataFrame,
) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()

    orders = _ensure_order_cohort_columns(orders, cohort_map)
    return (
        orders.groupby(["cohort", "change_type"])["hex"]
        .count()
        .unstack(fill_value=0)
        .astype(int)
        .rename_axis(None, axis=1)
    )


def _small_cohorts(composition: pd.DataFrame) -> pd.DataFrame:
    if composition.empty:
        return composition
    return composition[composition.sum(axis=1) < MIN_HEXAGONS_THRESHOLD]


def cohort_composition_report(
    cohort_map: pd.DataFrame,
    treated_orders: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Строит набор диагностических таблиц состава когорт по типам treatment.

    Ключи:
      composition_full
      composition_sch              — CORE types, когорта 2022-07-27 сохранена
      composition_post_meeting     — CORE types, 2022-07-27 исключена
      composition_meet             — алиас composition_post_meeting
      composition_success          — алиас composition_post_meeting
      composition_utilization      — алиас composition_post_meeting
      composition_conversion       — legacy-алиас composition_post_meeting
    """
    required_cols = {"hex", "cohort", "change_type", "is_treated"}
    missing = required_cols - set(cohort_map.columns)
    if missing:
        raise ValueError(
            f"cohort_map не содержит обязательных колонок: {missing}. "
            f"Ожидается результат get_cohort_map()."
        )

    if treated_orders is not None and not treated_orders.empty:
        orders_with_meta = _ensure_order_cohort_columns(
            treated_orders.copy(), cohort_map
        )
        hex_level = (
            orders_with_meta[["hex", "cohort", "change_type"]]
            .drop_duplicates(subset=["hex"])
            .reset_index(drop=True)
        )
    else:
        hex_level = cohort_map.loc[
            cohort_map["is_treated"], ["hex", "cohort", "change_type"]
        ].copy()

    composition_full = _pivot_hex_counts(hex_level)

    hex_level_sch = hex_level[hex_level["change_type"].isin(CORE_CHANGE_TYPES)].copy()
    composition_sch = _pivot_hex_counts(hex_level_sch)

    # meet_flg / success_flg / utlz_within_25: исключаем 2022-07-27
    hex_level_post = hex_level_sch[
        hex_level_sch["cohort"] != EXCLUDED_COHORT_FOR_POST_MEETING
    ].copy()
    composition_post_meeting = _pivot_hex_counts(hex_level_post)

    small_cohorts_sch = _small_cohorts(composition_sch)
    small_cohorts_post = _small_cohorts(composition_post_meeting)

    report: dict[str, pd.DataFrame] = {
        "composition_full": composition_full,
        "composition_sch": composition_sch,
        "composition_post_meeting": composition_post_meeting,
        "composition_meet": composition_post_meeting,
        "composition_success": composition_post_meeting,
        "composition_utilization": composition_post_meeting,
        # Legacy alias — не означает «только success/utilization»
        "composition_conversion": composition_post_meeting,
        "small_cohorts_sch": small_cohorts_sch,
        "small_cohorts_post_meeting": small_cohorts_post,
        "small_cohorts_conversion": small_cohorts_post,
        "composition_conv": composition_post_meeting,
        "small_cohorts_conv": small_cohorts_post,
    }

    if treated_orders is not None and not treated_orders.empty:
        orders = _ensure_order_cohort_columns(treated_orders.copy(), cohort_map)
        orders_full = _pivot_order_counts(orders, cohort_map)
        orders_sch_src = orders[orders["change_type"].isin(CORE_CHANGE_TYPES)].copy()
        orders_sch = _pivot_order_counts(orders_sch_src, cohort_map)
        orders_post_src = orders_sch_src[
            orders_sch_src["cohort"] != EXCLUDED_COHORT_FOR_POST_MEETING
        ].copy()
        orders_post = _pivot_order_counts(orders_post_src, cohort_map)

        report["orders_full"] = orders_full
        report["orders_sch"] = orders_sch
        report["orders_post_meeting"] = orders_post
        report["orders_meet"] = orders_post
        report["orders_success"] = orders_post
        report["orders_utilization"] = orders_post
        report["orders_conversion"] = orders_post
        report["orders_conv"] = orders_post

    return report


def print_cohort_composition_report(report: dict[str, pd.DataFrame]) -> None:
    """Печатает отчёт cohort_composition_report() в читаемом виде."""
    print("=" * 70)
    print("ПОЛНАЯ КОМПОЗИЦИЯ КОГОРТ (все change_type, n_hexagons)")
    print("=" * 70)
    print(report["composition_full"])

    print("\n" + "=" * 70)
    print("КОМПОЗИЦИЯ ДЛЯ sch_flg (CORE_CHANGE_TYPES; 2022-07-27 сохранена)")
    print("=" * 70)
    print(report["composition_sch"])

    print("\n" + "=" * 70)
    print(
        "КОМПОЗИЦИЯ ДЛЯ meet_flg / success_flg / utlz_within_25 "
        f"(исключена когорта {EXCLUDED_COHORT_FOR_POST_MEETING.date()}; "
        "для meet_flg — нарушение parallel trends, p ≈ 0.0168)"
    )
    print("=" * 70)
    print(report["composition_post_meeting"])

    print(f"\nМалые когорты (< {MIN_HEXAGONS_THRESHOLD} гексагонов) для sch_flg:")
    print(
        report["small_cohorts_sch"]
        if not report["small_cohorts_sch"].empty
        else "  (нет)"
    )

    print(
        f"\nМалые когорты (< {MIN_HEXAGONS_THRESHOLD} гексагонов) "
        "для meet / success / utilization:"
    )
    print(
        report["small_cohorts_post_meeting"]
        if not report["small_cohorts_post_meeting"].empty
        else "  (нет)"
    )

    if "orders_full" in report:
        print("\n" + "=" * 70)
        print("ЧИСЛО ЗАЯВОК ПО КОГОРТАМ И ТИПАМ: full")
        print("=" * 70)
        print(report["orders_full"])

        print("\n" + "=" * 70)
        print("ЧИСЛО ЗАЯВОК ПО КОГОРТАМ И ТИПАМ: sch_flg")
        print("=" * 70)
        print(report["orders_sch"])

        print("\n" + "=" * 70)
        print("ЧИСЛО ЗАЯВОК: meet_flg / success_flg / utlz_within_25")
        print("=" * 70)
        print(report["orders_post_meeting"])
