# -*- coding: utf-8 -*-
"""Generate Russian-labeled *_ru figures for success timing diagnostics.

Does not re-estimate DiD. Descriptive curves are recomputed with the same
definitions as the notebook; DiD panels are drawn from saved CSVs so
coefficients/CIs stay identical.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from last_mile.filter import build_analysis_panel  # noqa: E402
from last_mile.outcomes import success_horizon_coverage  # noqa: E402
from last_mile.plot_style import (  # noqa: E402
    PALETTE,
    save_figure,
    style_axes,
    with_plot_style,
)

OUT_DIR = ROOT / "outputs" / "success_timing_diagnostics"
FIG_DIR = ROOT / "figures" / "success_timing_diagnostics"
EMP_FIG = ROOT / "figures" / "empirical"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EMP_FIG.mkdir(parents=True, exist_ok=True)

TYPE_ORDER = ["region_and_workmode", "region_only", "workmode_only"]
TYPE_LABELS_RU = {
    "region_and_workmode": "Регион + режим работы",
    "region_only": "Только регион",
    "workmode_only": "Только режим работы",
}
TYPE_COLORS = {
    "region_and_workmode": PALETTE["region_and_workmode"],
    "region_only": PALETTE["region_only"],
    "workmode_only": PALETTE["workmode_only"],
}
REF_DAYS = [3, 7, 14, 25]
CURVE_DAYS = list(range(0, 26))
DID_HORIZONS = [3, 7, 14, 25]
EVENT_WEEK_MIN, EVENT_WEEK_MAX = -8, 8
SUCCESS_OBSERVATION_END = pd.Timestamp("2022-10-18")
EXCLUDED_COHORT_FOR_CONVERSION = pd.Timestamp("2022-07-27")
BUCKET_LABELS = ["0–1", "1–3", "3–5", "5–7", "7–10", "10–14", "14–20", "20–25"]
DELAY_BUCKETS = [
    (0, 1, "0–1"),
    (1, 3, "1–3"),
    (3, 5, "3–5"),
    (5, 7, "5–7"),
    (7, 10, "7–10"),
    (10, 14, "10–14"),
    (14, 20, "14–20"),
    (20, 25, "20–25"),
]


def add_ref_lines(ax, days=REF_DAYS):
    for d in days:
        ax.axvline(d, color=PALETTE["grid"], linewidth=0.9, linestyle=":", zorder=0)


def load_core_frame():
    print("Loading analysis panel for descriptive RU figures...")
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
        pd.Timestamp(x)
        for x in pd.to_datetime(apps_t["treatment_date"].dropna().unique())
    } - {EXCLUDED_COHORT_FOR_CONVERSION}

    apps_all = pd.concat([apps_t, apps_c], ignore_index=True)
    apps_core = apps_all[
        apps_all["change_type"].isin(TYPE_ORDER) | (~apps_all["is_treated"])
    ].copy()
    apps_core = apps_core[
        (~apps_core["is_treated"])
        | apps_core["treatment_date"].isin(valid)
    ].copy()
    apps_core["is_post"] = (
        apps_core["is_treated"]
        & apps_core["treatment_date"].notna()
        & (apps_core["request_timestamp"].dt.normalize() >= apps_core["treatment_date"])
    )
    apps_core["period"] = np.where(
        ~apps_core["is_treated"],
        "control",
        np.where(apps_core["is_post"], "post", "pre"),
    )
    delay = (
        (apps_core["first_success_dttm"] - apps_core["request_timestamp"]).dt.total_seconds()
        / 86400.0
    )
    valid_ok = (
        (pd.to_numeric(apps_core["success_flg"], errors="coerce") == 1)
        & apps_core["first_success_dttm"].notna()
        & delay.ge(0)
    )
    apps_core["success_delay_days"] = np.where(valid_ok, delay, np.nan)
    return apps_core


def cumulative_success_curve(df: pd.DataFrame, days=CURVE_DAYS) -> pd.DataFrame:
    request = pd.to_datetime(df["request_timestamp"], errors="coerce")
    success = pd.to_datetime(df["first_success_dttm"], errors="coerce")
    delay_days = (success - request).dt.total_seconds() / 86400.0
    valid_success = success.notna() & request.notna() & delay_days.ge(0)
    rows = []
    for H in days:
        horizon = pd.Timedelta(days=int(H))
        mature = request.notna() & (request + horizon <= SUCCESS_OBSERVATION_END)
        outcome = valid_success & (success <= request + horizon)
        n_eligible = int(mature.sum())
        n_success = int((mature & outcome).sum())
        rate = n_success / n_eligible if n_eligible else np.nan
        rows.append(
            {
                "horizon_days": H,
                "n_eligible": n_eligible,
                "cum_success_rate": rate,
            }
        )
    return pd.DataFrame(rows)


@with_plot_style
def plot_hist_ru(succ: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax = axes[0]
    ax.hist(
        succ["success_delay_days"],
        bins=40,
        color=PALETTE["treated"],
        edgecolor="white",
        linewidth=0.4,
    )
    add_ref_lines(ax)
    ax.set_xlabel("Дни до успешной встречи")
    ax.set_ylabel("Количество заявок")
    ax.set_title("Распределение времени до успешной встречи")
    style_axes(ax)

    ax = axes[1]
    clipped = succ.loc[succ["success_delay_days"].between(0, 25), "success_delay_days"]
    ax.hist(
        clipped,
        bins=np.arange(0, 26.5, 1.0),
        color=PALETTE["treated"],
        edgecolor="white",
        linewidth=0.4,
        histtype="stepfilled",
        alpha=0.85,
    )
    add_ref_lines(ax)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Дни до успешной встречи (0–25)")
    ax.set_ylabel("Количество заявок")
    ax.set_title("Распределение в пределах 25 дней")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "hist_success_delay_days_ru.pdf")


@with_plot_style
def plot_ecdf_ru(succ: pd.DataFrame, treated_succ: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for period, color, ls, lab in [
        ("pre", PALETTE["control"], "--", "До изменения"),
        ("post", PALETTE["treated"], "-", "После изменения"),
    ]:
        x = np.sort(
            treated_succ.loc[treated_succ["period"] == period, "success_delay_days"]
            .dropna()
            .to_numpy()
        )
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, linestyle=ls, label=f"{lab} (n={len(x)})")
    add_ref_lines(ax)
    ax.set_xlim(left=0)
    ax.set_xlabel("Дни до успешной встречи")
    ax.set_ylabel("Накопленная доля")
    ax.set_title(
        "Эмпирическая функция распределения времени до успеха:\nдо и после изменения"
    )
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "ecdf_success_delay_pre_post_ru.pdf")

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for ctype in TYPE_ORDER:
        x = np.sort(
            succ.loc[succ["change_type"] == ctype, "success_delay_days"].dropna().to_numpy()
        )
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(
            x,
            y,
            where="post",
            color=TYPE_COLORS[ctype],
            label=f"{TYPE_LABELS_RU[ctype]} (n={len(x)})",
        )
    add_ref_lines(ax)
    ax.set_xlim(left=0)
    ax.set_xlabel("Дни до успешной встречи")
    ax.set_ylabel("Накопленная доля")
    ax.set_title(
        "Эмпирическая функция распределения времени до успеха\nпо типу изменения"
    )
    ax.legend(loc="lower right", fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "ecdf_success_delay_by_change_type_ru.pdf")


@with_plot_style
def plot_cum_curves_ru(apps_core: pd.DataFrame):
    treated_core = apps_core.loc[apps_core["change_type"].isin(TYPE_ORDER)]
    curve_overall = cumulative_success_curve(apps_core)
    curve_pre = cumulative_success_curve(treated_core.loc[treated_core["period"] == "pre"])
    curve_post = cumulative_success_curve(treated_core.loc[treated_core["period"] == "post"])
    curve_by_type = {
        ctype: cumulative_success_curve(apps_core.loc[apps_core["change_type"] == ctype])
        for ctype in TYPE_ORDER
    }
    diff = pd.read_csv(OUT_DIR / "success_cumcurve_pre_post_diff.csv")

    # Sanity: recomputed pre/post rates match saved CSV
    merged = diff.merge(
        curve_pre.rename(columns={"cum_success_rate": "pre_check"})[
            ["horizon_days", "pre_check"]
        ],
        on="horizon_days",
    ).merge(
        curve_post.rename(columns={"cum_success_rate": "post_check"})[
            ["horizon_days", "post_check"]
        ],
        on="horizon_days",
    )
    assert np.allclose(merged["pre_rate"], merged["pre_check"], equal_nan=True, atol=1e-10)
    assert np.allclose(merged["post_rate"], merged["post_check"], equal_nan=True, atol=1e-10)
    print("[CHECK] cumulative pre/post rates match saved CSV")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.15))
    ax = axes[0]
    ax.plot(
        curve_overall["horizon_days"],
        100 * curve_overall["cum_success_rate"],
        color=PALETTE["zero"],
        label="Все наблюдения",
        linewidth=1.8,
    )
    ax.plot(
        curve_pre["horizon_days"],
        100 * curve_pre["cum_success_rate"],
        color=PALETTE["control"],
        linestyle="--",
        label="До изменения",
    )
    ax.plot(
        curve_post["horizon_days"],
        100 * curve_post["cum_success_rate"],
        color=PALETTE["treated"],
        label="После изменения",
    )
    for ctype in TYPE_ORDER:
        c = curve_by_type[ctype]
        if c["n_eligible"].max() < 50:
            continue
        ax.plot(
            c["horizon_days"],
            100 * c["cum_success_rate"],
            color=TYPE_COLORS[ctype],
            alpha=0.85,
            label=TYPE_LABELS_RU[ctype],
        )
    add_ref_lines(ax)
    ax.set_xlabel("Дни с момента заявки")
    ax.set_ylabel("Накопленная вероятность успешной встречи, %")
    ax.set_title("Накопленная вероятность успешной встречи")
    ax.legend(loc="lower right", fontsize=7.5)
    style_axes(ax)

    ax = axes[1]
    ax.plot(diff["horizon_days"], diff["diff_pp"], color=PALETTE["treated"], lw=1.8)
    if diff["diff_ci_low"].notna().any():
        ax.fill_between(
            diff["horizon_days"],
            diff["diff_ci_low"],
            diff["diff_ci_high"],
            color=PALETTE["treated"],
            alpha=0.18,
            label="95% бутстрэп-ДИ (B=200)",
        )
    ax.axhline(0, color=PALETTE["zero"], lw=0.9)
    add_ref_lines(ax)
    ax.set_xlabel("Дни с момента заявки")
    ax.set_ylabel("После − до, п.п.")
    ax.set_title("Разница после и до изменения: CORE-гексагоны")
    ax.legend(loc="best", fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "cumulative_success_curves_ru.pdf")


@with_plot_style
def plot_event_week_ru(ev_table: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5))

    ax = axes[0, 0]
    ax.plot(
        ev_table["event_week"],
        ev_table["median_days_to_success"],
        color=PALETTE["treated"],
        marker="o",
        ms=4,
        label="Медиана",
    )
    ax.fill_between(
        ev_table["event_week"],
        ev_table["p25_days_to_success"],
        ev_table["p75_days_to_success"],
        color=PALETTE["treated"],
        alpha=0.18,
        label="25–75-й процентили",
    )
    ax.axvline(-0.5, color=PALETTE["grid"], ls=":", lw=1)
    ax.set_title("Медианное время до успешной встречи\nпо неделям относительно изменения")
    ax.set_xlabel("Неделя относительно изменения")
    ax.set_ylabel("Дни до успешной встречи")
    ax.legend(fontsize=8)
    style_axes(ax)

    ax = axes[0, 1]
    ax.bar(ev_table["event_week"], ev_table["n_successful"], color=PALETTE["treated"], width=0.8)
    for _, r in ev_table.iterrows():
        ax.text(
            r["event_week"],
            r["n_successful"],
            str(int(r["n_applications"])),
            ha="center",
            va="bottom",
            fontsize=7,
            color=PALETTE["text_muted"],
        )
    ax.set_title("Число успешных заявок\nпо неделям относительно изменения")
    ax.set_xlabel("Неделя относительно изменения")
    ax.set_ylabel("Число успешных заявок")
    style_axes(ax)

    ax = axes[1, 0]
    for H, color, lab in [
        (7, PALETTE["region_only"], "В течение 7 дней"),
        (14, PALETTE["workmode_only"], "В течение 14 дней"),
        (25, PALETTE["region_and_workmode"], "В течение 25 дней"),
    ]:
        ax.plot(
            ev_table["event_week"],
            100 * ev_table[f"cum_success_{H}"],
            marker="o",
            ms=3.5,
            color=color,
            label=lab,
        )
    ax.axvline(-0.5, color=PALETTE["grid"], ls=":", lw=1)
    ax.set_title("Вероятность успешной встречи\nпо неделям относительно изменения")
    ax.set_xlabel("Неделя относительно изменения")
    ax.set_ylabel("Доля успешных встреч среди доступных заявок, %")
    ax.legend(fontsize=8)
    style_axes(ax)

    ax = axes[1, 1]
    ax.bar(
        ev_table["event_week"] - 0.2,
        ev_table["n_eligible_7"],
        width=0.2,
        color=PALETTE["region_only"],
        label="Доступно для H=7",
    )
    ax.bar(
        ev_table["event_week"],
        ev_table["n_eligible_14"],
        width=0.2,
        color=PALETTE["workmode_only"],
        label="Доступно для H=14",
    )
    ax.bar(
        ev_table["event_week"] + 0.2,
        ev_table["n_eligible_25"],
        width=0.2,
        color=PALETTE["region_and_workmode"],
        label="Доступно для H=25",
    )
    ax.set_title("Размер доступной выборки\nпо неделям относительно изменения")
    ax.set_xlabel("Неделя относительно изменения")
    ax.set_ylabel("Число доступных наблюдений")
    ax.legend(fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "event_week_success_diagnostics_ru.pdf")


def delay_bucket(days: float):
    if pd.isna(days):
        return None
    for lo, hi, lab in DELAY_BUCKETS:
        if lo <= days < hi or (hi == 25 and lo <= days <= hi):
            return lab
    return None


@with_plot_style
def plot_heatmap_ru(ev: pd.DataFrame):
    weeks = list(range(EVENT_WEEK_MIN, EVENT_WEEK_MAX + 1))

    def make_matrix(df):
        s = df.loc[df["success_delay_days"].notna()].copy()
        s = s.loc[s["event_week"].between(EVENT_WEEK_MIN, EVENT_WEEK_MAX)]
        s["bucket"] = s["success_delay_days"].map(delay_bucket)
        s = s.loc[s["bucket"].notna()]
        denom = s.groupby("event_week").size().reindex(weeks, fill_value=0)
        mat = (
            s.groupby(["event_week", "bucket"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=weeks, columns=BUCKET_LABELS, fill_value=0)
        )
        share = mat.div(denom.replace(0, np.nan), axis=0)
        return share, denom

    panels = [("Все CORE", ev)]
    for ctype in TYPE_ORDER:
        sub = ev.loc[ev["change_type"] == ctype]
        if sub["success_delay_days"].notna().sum() < 30:
            continue
        panels.append((TYPE_LABELS_RU[ctype], sub))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (title, sub) in zip(axes, panels):
        share, denom = make_matrix(sub)
        vmax = np.nanmax(share.to_numpy())
        im = ax.imshow(
            share.to_numpy(dtype=float),
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=vmax if np.isfinite(vmax) else 1,
            origin="lower",
        )
        ax.set_xticks(range(len(BUCKET_LABELS)))
        ax.set_xticklabels(BUCKET_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(weeks)))
        ax.set_yticklabels([str(w) for w in weeks])
        ax.set_title(f"{title}\n(знаменатель — успешные заявки)")
        ax.set_xlabel("Интервал времени до успешной встречи, дней")
        ax.set_ylabel("Неделя относительно изменения")
        ax.text(
            0.0,
            1.02,
            f"min n={int(denom.min())}, max n={int(denom.max())}",
            transform=ax.transAxes,
            fontsize=7,
            color=PALETTE["text_muted"],
        )
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="Доля успешных встреч")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "heatmap_event_week_delay_buckets_ru.pdf")


@with_plot_style
def plot_multi_horizon_ru():
    natural = pd.read_csv(OUT_DIR / "success_horizon_did.csv")
    cs_path = OUT_DIR / "success_horizon_did_common_support.csv"
    use_common = cs_path.exists()
    if use_common:
        df = pd.read_csv(cs_path)
        est_col, lo_col, hi_col, pt_col = (
            "post_estimate_pp",
            "ci_low_pp",
            "ci_high_pp",
            "pretrend_pvalue",
        )
        status_ok = df["status"].eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)
        out_name = "multi_horizon_did_common_support_ru.pdf"
        title = "Multi-horizon DiD на общей выборке наблюдений"
    else:
        df = natural
        est_col, lo_col, hi_col, pt_col = (
            "DID_post_estimate_pp",
            "CI_low",
            "CI_high",
            "pretrend_pvalue",
        )
        status_ok = df["status"].eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)
        out_name = "multi_horizon_did_comparison_ru.pdf"
        title = "DiD-оценки успешной встречи для разных временных горизонтов"

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for ax, ctype in zip(axes, TYPE_ORDER):
        sub = df.loc[(df["change_type"] == ctype) & status_ok].sort_values("horizon_days")
        ax.set_title(TYPE_LABELS_RU[ctype])
        if sub.empty:
            continue
        yerr = np.vstack(
            [
                sub[est_col] - sub[lo_col],
                sub[hi_col] - sub[est_col],
            ]
        )
        ax.errorbar(
            sub["horizon_days"],
            sub[est_col],
            yerr=yerr,
            fmt="o-",
            color=TYPE_COLORS[ctype],
            ecolor=PALETTE["text_muted"],
            capsize=3,
            lw=1.4,
        )
        ax.axhline(0, color=PALETTE["zero"], lw=0.9)
        for _, r in sub.iterrows():
            if pd.notna(r[pt_col]) and r[pt_col] < 0.05:
                ax.text(
                    r["horizon_days"],
                    r[est_col],
                    "*",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
        ax.set_xlabel("Горизонт H, дней")
        style_axes(ax)
    axes[0].set_ylabel("Средняя post-DiD-оценка, п.п.")
    fig.suptitle(
        f"{title}\n(* — тест предтрендов: p < 0,05; не означает значимость эффекта treatment)",
        y=1.08,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, FIG_DIR / out_name)

    # Always also emit natural-maturity RU comparison for completeness
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for ax, ctype in zip(axes, TYPE_ORDER):
        sub = natural.loc[
            (natural["change_type"] == ctype)
            & (natural["status"] == "ok" if "status" in natural.columns else True)
        ].sort_values("horizon_days")
        ax.set_title(TYPE_LABELS_RU[ctype])
        if sub.empty:
            continue
        yerr = np.vstack(
            [
                sub["DID_post_estimate_pp"] - sub["CI_low"],
                sub["CI_high"] - sub["DID_post_estimate_pp"],
            ]
        )
        ax.errorbar(
            sub["horizon_days"],
            sub["DID_post_estimate_pp"],
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
                    r["DID_post_estimate_pp"],
                    "*",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
        ax.set_xlabel("Горизонт H, дней")
        style_axes(ax)
    axes[0].set_ylabel("Средняя post-DiD-оценка, п.п.")
    fig.suptitle(
        "DiD-оценки успешной встречи для разных временных горизонтов\n"
        "(* — тест предтрендов: p < 0,05; не означает значимость эффекта treatment)",
        y=1.08,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "multi_horizon_did_comparison_ru.pdf")
    return use_common


@with_plot_style
def plot_event_study_grid_ru():
    es = pd.read_csv(OUT_DIR / "success_event_study.csv")
    fig, axes = plt.subplots(
        len(DID_HORIZONS),
        len(TYPE_ORDER),
        figsize=(12.0, 2.75 * len(DID_HORIZONS)),
        sharex=True,
        sharey=True,
    )
    for i, H in enumerate(DID_HORIZONS):
        for j, ctype in enumerate(TYPE_ORDER):
            ax = axes[i, j]
            sub = es.loc[
                (es["horizon_days"] == H) & (es["change_type"] == ctype)
            ].sort_values("rel_week")
            if sub.empty:
                ax.set_axis_off()
                continue
            ax.axhline(0, color=PALETTE["zero"], lw=0.8)
            ax.axvline(-0.5, color=PALETTE["grid"], ls=":", lw=0.9)
            ax.fill_between(
                sub["rel_week"],
                100 * sub["ci_low"],
                100 * sub["ci_high"],
                color=TYPE_COLORS[ctype],
                alpha=0.15,
            )
            ax.plot(
                sub["rel_week"],
                100 * sub["coef"],
                color=TYPE_COLORS[ctype],
                marker="o",
                ms=3,
            )
            ax.scatter([-1], [0], color=PALETTE["zero"], zorder=5, s=18)
            if i == 0:
                ax.set_title(TYPE_LABELS_RU[ctype], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"H={H}\nп.п.", fontsize=8)
            if i == len(DID_HORIZONS) - 1:
                ax.set_xlabel("Неделя относительно изменения")
            style_axes(ax)
    fig.suptitle("Event-study оценки по горизонту успешной встречи", y=1.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "event_study_by_horizon_ru.pdf")


def copy_publication_ru(use_common: bool):
    targets = [
        "cumulative_success_curves_ru.pdf",
        "cumulative_success_curves_ru.png",
    ]
    if use_common:
        targets += [
            "multi_horizon_did_common_support_ru.pdf",
            "multi_horizon_did_common_support_ru.png",
        ]
    else:
        targets += [
            "multi_horizon_did_comparison_ru.pdf",
            "multi_horizon_did_comparison_ru.png",
        ]
    import shutil

    for name in targets:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, EMP_FIG / name)
            print("Copied", name, "-> figures/empirical/")


def append_notebook_note():
    import json
    import uuid

    nb_path = ROOT / "notebooks" / "success_timing_diagnostics.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    marker = "localize_success_timing_figures"
    if any(marker in "".join(c.get("source", [])) for c in nb["cells"]):
        print("Notebook already references RU localization script.")
        return

    def md(s):
        lines = s.strip("\n").splitlines(True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": lines,
        }

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

    cells = [
        md(
            """
## Русская локализация графиков (только подписи)

Английские PDF/PNG не перезаписываются. Русские версии сохраняются как `*_ru.pdf` /
`*_ru.png` скриптом `scripts/localize_success_timing_figures.py`.
Числа, CI и выборки не пересчитываются заново для DiD (берутся из CSV);
описательные кривые пересчитываются той же формулой и сверяются с
`success_cumcurve_pre_post_diff.csv`.
"""
        ),
        code(
            """
# Regenerates *_ru figures without changing DiD estimates.
# Uncomment to re-run locally:
# %run ../scripts/localize_success_timing_figures.py
print("RU figures are produced by scripts/localize_success_timing_figures.py")
print("Expected publication files:")
for name in [
    "cumulative_success_curves_ru.pdf",
    "multi_horizon_did_common_support_ru.pdf",
]:
    p = FIG_DIR / name
    print(" ", "OK" if p.exists() else "MISSING", p)
"""
        ),
    ]
    insert_at = len(nb["cells"]) - 2
    for i, c in enumerate(nb["cells"]):
        if "## 11. Outputs" in "".join(c.get("source", [])):
            insert_at = i
            break
    nb["cells"] = nb["cells"][:insert_at] + cells + nb["cells"][insert_at:]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Appended RU localization note cells to notebook")


def main():
    apps_core = load_core_frame()
    succ = apps_core.loc[apps_core["success_delay_days"].notna()].copy()
    treated_succ = succ.loc[succ["change_type"].isin(TYPE_ORDER)].copy()
    treated_core = apps_core.loc[apps_core["change_type"].isin(TYPE_ORDER)].copy()
    treated_core["application_date"] = treated_core["request_timestamp"].dt.normalize()
    treated_core["event_day"] = (
        treated_core["application_date"] - treated_core["treatment_date"]
    ).dt.days
    treated_core["event_week"] = np.floor(treated_core["event_day"] / 7.0).astype("Int64")
    ev = treated_core.loc[
        treated_core["event_week"].between(EVENT_WEEK_MIN, EVENT_WEEK_MAX)
    ].copy()

    plot_hist_ru(succ)
    plot_ecdf_ru(succ, treated_succ)
    plot_cum_curves_ru(apps_core)

    ev_table = pd.read_csv(OUT_DIR / "success_event_week_descriptives.csv")
    plot_event_week_ru(ev_table)
    plot_heatmap_ru(ev)
    use_common = plot_multi_horizon_ru()
    plot_event_study_grid_ru()
    copy_publication_ru(use_common)
    append_notebook_note()

    print("Created RU figures:")
    for p in sorted(FIG_DIR.glob("*_ru.pdf")):
        print(" ", p.name)
    print("use_common_support_publication:", use_common)


if __name__ == "__main__":
    main()
