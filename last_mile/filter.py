"""
last_mile/filter.py

Пайплайн фильтрации и построения аналитической панели для оценки
эффекта изменений логистической конфигурации методом TWFE DiD.

Единица наблюдения: (hex, date) — гексагон-день.
Период исследования: 2022-04-01 — 2022-10-18 (включительно).
Исключаемая когорта: treatment_date == 2022-10-19 (правоцензурирование).

Политика multi-treatment:
  гексагоны с несколькими различными ненулевыми treatment_date исключаются
  из основной DiD-выборки; диагностика сохраняется отдельно.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io import load_applications, load_hexagons

STUDY_START = pd.Timestamp("2022-04-01")
STUDY_END = pd.Timestamp("2022-10-18")

EXCLUDED_COHORTS: set[pd.Timestamp] = {pd.Timestamp("2022-10-19")}

INACTIVE_REGION = -100

REGION_CHANGE = "region_only"
WORKMODE_CHANGE = "workmode_only"
REGION_AND_WORKMODE_CHANGE = "region_and_workmode"
OPENED = "opened"
CLOSED = "closed"
NEVER_ACTIVE = "never_active"
OTHER = "other"

CORE_CHANGE_TYPES = [REGION_CHANGE, WORKMODE_CHANGE, REGION_AND_WORKMODE_CHANGE]

# Re-export loaders for backwards compatibility with `from last_mile.filter import ...`
__all__ = [
    "load_applications",
    "load_hexagons",
    "classify_hexagons",
    "resolve_hexagon_treatments",
    "validate_single_treatment_hexagons",
    "get_cohort_map",
    "filter_study_window",
    "attach_hex_metadata",
    "exclude_cohorts",
    "compute_time_variables",
    "aggregate_to_hex_day",
    "build_analysis_panel",
    "build_did_samples",
    "CORE_CHANGE_TYPES",
    "STUDY_START",
    "STUDY_END",
    "EXCLUDED_COHORTS",
]


def classify_hexagons(hexagons: pd.DataFrame) -> pd.DataFrame:
    """
    Классифицирует каждый гексагон по типу воздействия и когорте.

    Активность зоны определяется сентинелом region_id == INACTIVE_REGION (-100),
    а не NaN: в данных region_id всегда заполнен, неактивная зона кодируется -100.

    Логика change_type:
      - был неактивен и остался неактивен        -> 'never_active'
      - был неактивен, стал активен              -> 'opened'
      - был активен, стал неактивен              -> 'closed'
      - активен, изменился только region_id      -> 'region_only'
      - активен, изменился только work_mode      -> 'workmode_only'
      - активен, изменились и регион, и режим     -> 'region_and_workmode'
      - прочие случаи                            -> 'other'

    NaT в treatment_date -> контрольная группа (never-treated).
    """
    df = hexagons.copy()

    active_old = df["region_id_old"] != INACTIVE_REGION
    active_new = df["region_id_new"] != INACTIVE_REGION
    same_region = df["region_id_old"] == df["region_id_new"]
    wm_changed = df["work_mode_old"] != df["work_mode_new"]

    df["change_type"] = OTHER
    df.loc[(~active_old) & (~active_new), "change_type"] = NEVER_ACTIVE
    df.loc[(~active_old) & active_new, "change_type"] = OPENED
    df.loc[active_old & (~active_new), "change_type"] = CLOSED
    df.loc[active_old & active_new & (~same_region) & (~wm_changed), "change_type"] = REGION_CHANGE
    df.loc[active_old & active_new & same_region & wm_changed, "change_type"] = WORKMODE_CHANGE
    df.loc[active_old & active_new & (~same_region) & wm_changed, "change_type"] = (
        REGION_AND_WORKMODE_CHANGE
    )

    df["cohort"] = df["treatment_date"]
    df["is_treated"] = df["treatment_date"].notna()

    return df


def resolve_hexagon_treatments(
    hexagons: pd.DataFrame,
    *,
    treatment_col: str = "treatment_date",
    hex_col: str = "hex",
    metadata_cols: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Явно разрешает multi-treatment гексагоны.

    Политика основной оценки:
      - оставляются только single-treatment гексагоны (0 или 1 уникальная
        ненулевая treatment_date);
      - гексагоны с несколькими различными ненулевыми treatment_date
        исключаются из основной DiD-выборки;
      - повторяющиеся одинаковые строки с одной и той же treatment_date
        не считаются отдельными воздействиями.

    Returns
    -------
    dict:
        'single_treatment' : строки гексагонов, допустимые для основной оценки
                             (по одной строке на hex после дедупликации
                             идентичных treatment_date)
        'multi_treatment'  : диагностическая таблица исключённых гексагонов
        'metadata_conflicts': строки с противоречивыми old/new metadata при
                              одной treatment_date
        'diagnostics'      : сводка по hex (n_unique_dates, policy, ...)
        'summary'          : агрегированная воспроизводимая сводка resolution
    """
    if hex_col not in hexagons.columns:
        raise ValueError(f"resolve_hexagon_treatments: нет колонки '{hex_col}'")
    if treatment_col not in hexagons.columns:
        raise ValueError(f"resolve_hexagon_treatments: нет колонки '{treatment_col}'")

    df = hexagons.copy()
    df[treatment_col] = pd.to_datetime(df[treatment_col], errors="coerce")
    if metadata_cols is None:
        metadata_cols = [
            "region_id_old",
            "region_id_new",
            "work_mode_old",
            "work_mode_new",
        ]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    # Одинаковая дата не делает строки взаимозаменяемыми: old/new metadata
    # являются частью определения воздействия. Любое расхождение сохраняется
    # отдельно и исключается из канонической карты, а не разрешается keep="first".
    treated_for_conflicts = df[df[treatment_col].notna()].copy()
    if metadata_cols and not treated_for_conflicts.empty:
        metadata_nunique = (
            treated_for_conflicts.groupby(
                [hex_col, treatment_col], dropna=False
            )[metadata_cols]
            .nunique(dropna=False)
        )
        conflicting_keys = metadata_nunique.index[
            metadata_nunique.gt(1).any(axis=1)
        ]
        conflict_key_frame = conflicting_keys.to_frame(index=False)
        metadata_conflicts = treated_for_conflicts.merge(
            conflict_key_frame,
            on=[hex_col, treatment_col],
            how="inner",
        ).sort_values([hex_col, treatment_col, *metadata_cols])
    else:
        metadata_conflicts = df.iloc[0:0].copy()
    conflict_hexes = set(metadata_conflicts[hex_col])

    # Never-treated: все treatment_date пусты.
    # Single-treatment: ровно одна уникальная ненулевая дата (дубликаты той же даты OK).
    # Multi-treatment: ≥2 различных ненулевых дат.
    treated_rows = df[df[treatment_col].notna()].copy()

    n_unique = (
        treated_rows.groupby(hex_col)[treatment_col]
        .nunique()
        .rename("n_unique_treatment_dates")
    )
    n_rows = treated_rows.groupby(hex_col).size().rename("n_rows_with_treatment")

    never_hexes = set(df.loc[df[treatment_col].isna(), hex_col]) - set(n_unique.index)
    diagnostics = pd.concat([n_unique, n_rows], axis=1).reset_index()
    diagnostics["policy"] = np.where(
        diagnostics["n_unique_treatment_dates"] > 1,
        "exclude_multi_treatment",
        "keep_single_treatment",
    )
    diagnostics.loc[
        diagnostics[hex_col].isin(conflict_hexes)
        & diagnostics["policy"].eq("keep_single_treatment"),
        "policy",
    ] = "exclude_same_date_metadata_conflict"

    if never_hexes:
        never_diag = pd.DataFrame(
            {
                hex_col: list(never_hexes),
                "n_unique_treatment_dates": 0,
                "n_rows_with_treatment": 0,
                "policy": "keep_never_treated",
            }
        )
        diagnostics = pd.concat([diagnostics, never_diag], ignore_index=True)

    multi_hexes = set(
        diagnostics.loc[
            diagnostics["policy"] == "exclude_multi_treatment", hex_col
        ]
    )
    keep_hexes = set(df[hex_col]) - multi_hexes - conflict_hexes

    multi_treatment = (
        df[df[hex_col].isin(multi_hexes)]
        .sort_values([hex_col, treatment_col])
        .reset_index(drop=True)
    )

    # Для keep допустимы только полностью идентичные строки. Нельзя
    # дедуплицировать по (hex, date), поскольку это скрывает metadata conflicts.
    keep_df = df[df[hex_col].isin(keep_hexes)].copy()
    single_treatment = (
        keep_df.drop_duplicates(keep="first")
        .sort_values([hex_col, treatment_col])
        .reset_index(drop=True)
    )

    # После дедупликации одинаковых дат на hex должна остаться ≤1 ненулевая дата.
    check = (
        single_treatment[single_treatment[treatment_col].notna()]
        .groupby(hex_col)[treatment_col]
        .nunique()
    )
    bad = check[check > 1]
    if len(bad) > 0:
        raise RuntimeError(
            "resolve_hexagon_treatments: после фильтрации остались multi-treatment "
            f"гексагоны: {bad.index.tolist()[:10]}"
        )

    # Для never-treated также запрещён неявный выбор metadata.
    never_mask = single_treatment[treatment_col].isna()
    never_part = single_treatment.loc[never_mask]
    treated_part = single_treatment.loc[~never_mask]
    single_treatment = pd.concat([treated_part, never_part], ignore_index=True)

    if not single_treatment[hex_col].is_unique:
        dupes = single_treatment.loc[
            single_treatment[hex_col].duplicated(keep=False), hex_col
        ].unique()
        raise RuntimeError(
            "resolve_hexagon_treatments: после удаления точных дублей остались "
            "неоднозначные строки metadata. Silent deduplication запрещена. "
            f"Примеры hex: {list(dupes[:10])}"
        )

    summary = (
        diagnostics.groupby("policy", dropna=False)
        .agg(n_hexagons=(hex_col, "nunique"))
        .reset_index()
    )
    summary["n_input_hexagons"] = int(df[hex_col].nunique())
    summary["n_canonical_hexagons"] = int(single_treatment[hex_col].nunique())

    return {
        "single_treatment": single_treatment,
        "multi_treatment": multi_treatment,
        "metadata_conflicts": metadata_conflicts.reset_index(drop=True),
        "diagnostics": diagnostics.sort_values(hex_col).reset_index(drop=True),
        "summary": summary.sort_values("policy").reset_index(drop=True),
    }


def validate_single_treatment_hexagons(
    hexagons: pd.DataFrame,
    *,
    treatment_col: str = "treatment_date",
    hex_col: str = "hex",
) -> dict[str, pd.DataFrame]:
    """Алиас resolve_hexagon_treatments для явной семантики валидации."""
    return resolve_hexagon_treatments(
        hexagons, treatment_col=treatment_col, hex_col=hex_col
    )


def get_cohort_map(hexagons_classified: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает таблицу соответствия hex -> (cohort, change_type, is_treated).

    Ожидает, что multi-treatment уже разрешён через resolve_hexagon_treatments().
    Требует уникальности hex; неявный выбор «первой строки» запрещён.
    """
    cols = [
        "hex",
        "cohort",
        "change_type",
        "is_treated",
        "treatment_date",
        "region_id_old",
        "region_id_new",
        "work_mode_old",
        "work_mode_new",
        "subject",
    ]
    available = [c for c in cols if c in hexagons_classified.columns]
    out = hexagons_classified[available].copy()

    if not out["hex"].is_unique:
        dupes = out.loc[out["hex"].duplicated(keep=False), "hex"].unique()
        raise ValueError(
            "get_cohort_map: колонка 'hex' не уникальна. Сначала вызовите "
            f"resolve_hexagon_treatments(). Примеры дублей: {list(dupes[:10])}"
        )
    return out.reset_index(drop=True)


def filter_study_window(applications: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только заявки внутри окна [STUDY_START, STUDY_END]."""
    mask = (
        (applications["request_timestamp"] >= STUDY_START)
        & (applications["request_timestamp"] <= STUDY_END)
    )
    filtered = applications.loc[mask].copy()
    assert len(filtered) > 0, (
        f"После фильтрации по окну {STUDY_START.date()}–{STUDY_END.date()} "
        "не осталось наблюдений. Проверьте формат request_timestamp."
    )
    return filtered


def attach_hex_metadata(
    applications: pd.DataFrame,
    cohort_map: pd.DataFrame,
) -> pd.DataFrame:
    """Присоединяет к заявкам метаданные гексагона."""
    merged = applications.merge(cohort_map, on="hex", how="inner")
    n_dropped = len(applications) - len(merged)
    if n_dropped > 0:
        print(
            f"[INFO] attach_hex_metadata: отброшено {n_dropped} заявок "
            "из гексагонов, не представленных в hexagons_dataset."
        )
    return merged


def exclude_cohorts(
    df: pd.DataFrame,
    excluded: set[pd.Timestamp] = EXCLUDED_COHORTS,
) -> pd.DataFrame:
    """Исключает когорты из EXCLUDED_COHORTS. Never-treated не затрагиваются."""
    mask_excluded = df["cohort"].isin(excluded)
    n_excluded = mask_excluded.sum()
    if n_excluded > 0:
        print(
            f"[INFO] exclude_cohorts: исключено {n_excluded} заявок "
            f"из когорт {sorted(str(c.date()) for c in excluded)}."
        )
    return df.loc[~mask_excluded].copy()


def compute_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет date, t_available, t_utilization."""
    df = df.copy()
    df["date"] = df["request_timestamp"].dt.normalize()

    if "start_interval" in df.columns:
        delta = df["start_interval"] - df["request_timestamp"]
        df["t_available"] = delta.dt.total_seconds() / 86400.0
        df.loc[df["t_available"] < 0, "t_available"] = np.nan

    if "real_utilization_dttm" in df.columns:
        delta_u = df["real_utilization_dttm"] - df["request_timestamp"]
        df["t_utilization"] = delta_u.dt.total_seconds() / 86400.0
        df.loc[df["t_utilization"] < 0, "t_utilization"] = np.nan

    return df


def aggregate_to_hex_day(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует заявки до уровня (hex, date)."""
    outcome_cols = {
        "sch_flg": "mean",
        "meet_flg": "mean",
        "success_flg": "mean",
        "utlz_flg": "mean",
        "utlz_within_25": "mean",
    }
    time_cols = {}
    if "t_available" in df.columns:
        time_cols["t_available"] = "median"
    if "t_utilization" in df.columns:
        time_cols["t_utilization"] = "median"

    agg_dict = {"request_timestamp": "count"}
    agg_dict.update({k: v for k, v in outcome_cols.items() if k in df.columns})
    agg_dict.update(time_cols)

    panel = (
        df.groupby(["hex", "date", "cohort", "change_type", "is_treated"], dropna=False)
        .agg(agg_dict)
        .rename(columns={"request_timestamp": "n_orders"})
        .reset_index()
    )

    days_from_cohort = (panel["date"] - panel["cohort"]).dt.days
    panel["weeks_since_treatment"] = (days_from_cohort // 7).astype("Int64")

    return panel


def build_analysis_panel(
    applications_path: str | Path,
    hexagons_path: str | Path,
    min_orders_per_hex: int = 5,
    excluded_cohorts: Optional[set[pd.Timestamp]] = None,
    multi_treatment_diagnostics_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Строит аналитическую панель для TWFE DiD оценивания.

    Multi-treatment гексагоны исключаются до построения cohort_map.
    """
    if excluded_cohorts is None:
        excluded_cohorts = EXCLUDED_COHORTS

    applications = load_applications(applications_path)
    hexagons = load_hexagons(hexagons_path)

    hexagons_cls = classify_hexagons(hexagons)
    resolved = resolve_hexagon_treatments(hexagons_cls)
    hexagons_single = resolved["single_treatment"]
    multi_diag = resolved["multi_treatment"]
    metadata_conflicts = resolved["metadata_conflicts"]
    treatment_diag = resolved["diagnostics"]

    n_multi = treatment_diag["policy"].eq("exclude_multi_treatment").sum()
    print(
        f"[INFO] resolve_hexagon_treatments: исключено {n_multi} multi-treatment "
        f"гексагонов; сохранено {hexagons_single['hex'].nunique()} single/never."
    )

    if multi_treatment_diagnostics_path is None:
        multi_treatment_diagnostics_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "diagnostics"
            / "multi_treatment_hexagons.csv"
        )
    diag_path = Path(multi_treatment_diagnostics_path)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    multi_diag.to_csv(diag_path, index=False)
    treatment_diag.to_csv(
        diag_path.with_name("multi_treatment_hexagon_summary.csv"), index=False
    )
    metadata_conflicts.to_csv(
        diag_path.with_name("same_date_metadata_conflicts.csv"), index=False
    )
    resolved["summary"].to_csv(
        diag_path.with_name("treatment_resolution_summary.csv"), index=False
    )
    repo_root = Path(__file__).resolve().parents[1]
    try:
        diag_display = diag_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        diag_display = diag_path.name
    print(f"[INFO] multi-treatment diagnostics: {diag_display}")

    cohort_map = get_cohort_map(hexagons_single)
    if not cohort_map["hex"].is_unique:
        raise RuntimeError("build_analysis_panel: cohort_map['hex'] не уникален")

    apps_windowed = filter_study_window(applications)
    apps_meta = attach_hex_metadata(apps_windowed, cohort_map)
    apps_clean = exclude_cohorts(apps_meta, excluded=excluded_cohorts)
    apps_clean = compute_time_variables(apps_clean)

    hex_order_counts = apps_clean.groupby("hex")["request_timestamp"].count()
    viable_hexes = hex_order_counts[hex_order_counts >= min_orders_per_hex].index
    apps_clean = apps_clean[apps_clean["hex"].isin(viable_hexes)].copy()
    print(
        f"[INFO] build_analysis_panel: "
        f"после порогового фильтра (≥{min_orders_per_hex} заявок) "
        f"осталось {apps_clean['hex'].nunique()} уникальных гексагонов."
    )

    treated_orders = apps_clean[apps_clean["is_treated"]].copy()
    control_orders = apps_clean[~apps_clean["is_treated"]].copy()
    panel = aggregate_to_hex_day(apps_clean)

    _print_summary(panel, cohort_map, excluded_cohorts)

    return {
        "panel": panel,
        "treated_orders": treated_orders,
        "control_orders": control_orders,
        "cohort_map": cohort_map,
        "multi_treatment": multi_diag,
        "metadata_conflicts": metadata_conflicts,
        "treatment_diagnostics": treatment_diag,
        "treatment_resolution_summary": resolved["summary"],
    }


def _print_summary(
    panel: pd.DataFrame,
    cohort_map: pd.DataFrame,
    excluded_cohorts: set[pd.Timestamp],
) -> None:
    n_hex_total = panel["hex"].nunique()
    n_hex_treated = panel.loc[panel["is_treated"], "hex"].nunique()
    n_hex_control = panel.loc[~panel["is_treated"], "hex"].nunique()
    n_obs = len(panel)
    date_min = panel["date"].min()
    date_max = panel["date"].max()

    treated_panel = panel[
        panel["is_treated"]
        & panel["cohort"].notna()
        & ~panel["cohort"].isin(excluded_cohorts)
    ]
    cohort_counts = treated_panel.groupby("cohort")["hex"].nunique().sort_index()

    print("\n" + "=" * 60)
    print("СВОДКА АНАЛИТИЧЕСКОЙ ПАНЕЛИ")
    print("=" * 60)
    print(f"  Период:              {date_min.date()} - {date_max.date()}")
    print(f"  Наблюдений (hex×day):{n_obs:>10,}")
    print(f"  Гексагонов всего:    {n_hex_total:>10,}")
    print(f"  - трактуемых:        {n_hex_treated:>10,}")
    print(f"  - never-treated:     {n_hex_control:>10,}")
    print("\n  Когорты (включённые):")
    for cohort_date, count in cohort_counts.items():
        print(f"    {cohort_date.date()}  →  {count} гексагонов")
    print(
        f"\n  Исключённые когорты: "
        f"{sorted(str(c.date()) for c in excluded_cohorts)}"
    )
    print("=" * 60 + "\n")


def build_did_samples(
    applications_path: str | Path,
    hexagons_path: str | Path,
    multi_treatment_diagnostics_path: str | Path | None = None,
) -> dict:
    """
    Готовит выборки уровня заявки для предоценочной диагностики DiD.

    Гексагоны с несколькими различными treatment_date исключаются явно
    через resolve_hexagon_treatments().
    """
    applications = load_applications(applications_path)
    hexagons = load_hexagons(hexagons_path)
    if "Unnamed: 0" in hexagons.columns:
        hexagons = hexagons.drop(columns="Unnamed: 0")

    hexagons = classify_hexagons(hexagons)
    resolved = resolve_hexagon_treatments(hexagons)
    hexagons_single = resolved["single_treatment"]
    multi_diag = resolved["multi_treatment"]
    metadata_conflicts = resolved["metadata_conflicts"]

    if multi_treatment_diagnostics_path is None:
        multi_treatment_diagnostics_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "diagnostics"
            / "multi_treatment_hexagons.csv"
        )
    diag_path = Path(multi_treatment_diagnostics_path)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    multi_diag.to_csv(diag_path, index=False)
    metadata_conflicts.to_csv(
        diag_path.with_name("same_date_metadata_conflicts.csv"), index=False
    )
    resolved["diagnostics"].to_csv(
        diag_path.with_name("multi_treatment_hexagon_summary.csv"), index=False
    )
    resolved["summary"].to_csv(
        diag_path.with_name("treatment_resolution_summary.csv"), index=False
    )

    obs_min = applications["request_timestamp"].min()
    obs_max = applications["request_timestamp"].max()

    never_treated = hexagons_single.loc[hexagons_single["treatment_date"].isna(), "hex"]
    control_hexes = set(never_treated) & set(applications["hex"])

    treated_meta = hexagons_single.loc[
        hexagons_single["treatment_date"].notna(),
        ["hex", "treatment_date", "change_type"],
    ]
    if not treated_meta["hex"].is_unique:
        raise RuntimeError("build_did_samples: treated hex_meta не уникален по hex")

    hex_meta = treated_meta.reset_index(drop=True)
    single_change = set(hex_meta["hex"])

    apps_t = applications[applications["hex"].isin(single_change)].merge(
        hex_meta, on="hex", how="left"
    )
    apps_t["days_from_treatment"] = (
        apps_t["request_timestamp"] - apps_t["treatment_date"]
    ).dt.days

    apps_c = applications[applications["hex"].isin(control_hexes)].copy()

    print("Период наблюдений:", obs_min.date(), "-", obs_max.date())
    print("treated-заявок:", len(apps_t), "| control-заявок:", len(apps_c))
    print(
        "treated-гексагонов с заявками:",
        apps_t["hex"].nunique(),
        "| control-гексагонов:",
        apps_c["hex"].nunique(),
    )
    print(
        "[INFO] multi-treatment исключено:",
        resolved["diagnostics"]["policy"].eq("exclude_multi_treatment").sum(),
    )

    return {
        "apps_t": apps_t,
        "apps_c": apps_c,
        "hex_meta": hex_meta,
        "control_hexes": control_hexes,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "hexagons": hexagons_single,
        "multi_treatment": multi_diag,
        "metadata_conflicts": metadata_conflicts,
        "treatment_diagnostics": resolved["diagnostics"],
        "treatment_resolution_summary": resolved["summary"],
    }
