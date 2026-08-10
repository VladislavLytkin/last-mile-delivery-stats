from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

APPLICATION_COLS = [
    "request_timestamp",
    "type",
    "sch_flg",
    "meet_flg",
    "success_flg",
    "utlz_flg",
    "real_utilization_dttm",
    "planned_start_date",
    "first_success_dttm",
    "start_interval",
    "hex",
]

HEXAGON_COLS = [
    "hex",
    "subject",
    "treatment_date",
    "region_id_old",
    "region_id_new",
    "work_mode_old",
    "work_mode_new",
]

DATE_COLS_APPLICATIONS = [
    "request_timestamp",
    "real_utilization_dttm",
    "planned_start_date",
    "first_success_dttm",
    "start_interval",
]

DATE_COLS_HEXAGONS = ["treatment_date"]


def _require_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: отсутствуют обязательные колонки: {missing}. "
            f"Доступны: {list(df.columns)}"
        )


def load_applications(path: str | Path | None = None) -> pd.DataFrame:
    """
    Загружает датасет заявок.

    Параметры чтения:
      - sep=';'
      - parse_dates для временных колонок
      - low_memory=False
    """
    path = Path(path) if path is not None else RAW_DIR / "application_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"applications: файл не найден: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        usecols=lambda c: c in APPLICATION_COLS,
        parse_dates=DATE_COLS_APPLICATIONS,
        low_memory=False,
    )
    _require_columns(df, ["hex", "request_timestamp"], "applications")

    for col in DATE_COLS_APPLICATIONS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def load_hexagons(path: str | Path | None = None) -> pd.DataFrame:
    """
    Загружает датасет гексагонов с метаданными изменений.
    """
    path = Path(path) if path is not None else RAW_DIR / "hexagons_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"hexagons: файл не найден: {path}")

    df = pd.read_csv(
        path,
        usecols=lambda c: c in HEXAGON_COLS,
        parse_dates=DATE_COLS_HEXAGONS,
        low_memory=False,
    )
    _require_columns(df, ["hex", "treatment_date"], "hexagons")

    if df["treatment_date"].dtype == object:
        df["treatment_date"] = pd.to_datetime(df["treatment_date"], errors="coerce")

    return df
