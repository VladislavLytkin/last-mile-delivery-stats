"""Canonical outcome definitions used by the empirical notebooks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_success_within_horizon(
    applications: pd.DataFrame,
    *,
    horizon_days: int = 20,
    observation_end: pd.Timestamp | str,
    request_col: str = "request_timestamp",
    success_time_col: str = "first_success_dttm",
) -> pd.DataFrame:
    """Create a horizon-specific success outcome and an explicit maturity flag."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    required = {request_col, success_time_col}
    missing = required - set(applications.columns)
    if missing:
        raise ValueError(f"missing columns for success horizon: {sorted(missing)}")

    out = applications.copy()
    request = pd.to_datetime(out[request_col], errors="coerce")
    success = pd.to_datetime(out[success_time_col], errors="coerce")
    observation_end = pd.Timestamp(observation_end)
    if observation_end.tz is not None:
        observation_end = observation_end.tz_localize(None)

    horizon = pd.Timedelta(days=horizon_days)
    maturity_col = f"mature_success_{horizon_days}"
    outcome_col = f"success_within_{horizon_days}"
    out[maturity_col] = request.notna() & (request + horizon <= observation_end)
    nonnegative_lag = success.isna() | (success >= request)
    out[outcome_col] = (
        success.notna()
        & nonnegative_lag
        & (success <= request + horizon)
    ).astype("Int8")
    out.loc[~out[maturity_col], outcome_col] = pd.NA
    return out


def success_horizon_coverage(
    applications: pd.DataFrame,
    *,
    horizons: Iterable[int] = (7, 14, 20, 25, 30),
    observation_end: pd.Timestamp | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return horizon-specific outcomes and reproducible maturity diagnostics."""
    enriched = applications.copy()
    rows: list[dict[str, float | int | str]] = []
    n_total = len(enriched)
    for horizon in horizons:
        enriched = add_success_within_horizon(
            enriched,
            horizon_days=int(horizon),
            observation_end=observation_end,
        )
        mature_col = f"mature_success_{int(horizon)}"
        outcome_col = f"success_within_{int(horizon)}"
        mature = enriched[mature_col]
        rows.append(
            {
                "horizon_days": int(horizon),
                "observation_end": str(pd.Timestamp(observation_end).date()),
                "n_requests_total": n_total,
                "n_requests_mature": int(mature.sum()),
                "mature_share": float(mature.mean()) if n_total else float("nan"),
                "n_success_within_horizon": int(
                    enriched.loc[mature, outcome_col].sum()
                ),
            }
        )
    return enriched, pd.DataFrame(rows)
