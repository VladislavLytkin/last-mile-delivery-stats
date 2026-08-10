# -*- coding: utf-8 -*-
"""
Полная диагностика фактического закрытия гексагонов.

Пишет CSV в outputs/ и графики в figures/empirical/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from last_mile.filter import CORE_CHANGE_TYPES, CLOSED, STUDY_END, STUDY_START  # noqa: E402
from last_mile.hex_activity import (  # noqa: E402
    EMPIRICAL_CLOSED,
    PRIMARY_WINDOW_DAYS,
    MIN_PRE_ORDERS,
    MIN_PRE_ACTIVE_DAYS,
    NEAR_CLOSED_RATIO,
    MAX_NEAR_CLOSED_ACTIVE_DAYS,
    build_event_time_activity,
    build_hex_activity_panel,
    build_metadata_vs_empirical_crosstab,
    classify_empirical_activity,
    compute_closure_window_metrics,
    prepare_analysis_hex_universe,
    run_closure_threshold_sensitivity,
    save_activity_panel,
    selective_attrition_diagnostics,
    summarize_comparison_groups,
)
from last_mile.io import load_applications, load_hexagons  # noqa: E402
from last_mile.plot_style import (  # noqa: E402
    PALETTE,
    save_figure,
    style_axes,
    with_plot_style,
)

OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures" / "empirical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_LABELS = {
    "admin_closed": "Административно closed",
    "core": "CORE",
    "empirical_closed": "Эмпирически закрытые",
    "continues_active": "Активность продолжается",
}
GROUP_COLORS = {
    "admin_closed": "#7B8794",
    "core": PALETTE["region_and_workmode"],
    "empirical_closed": "#B85C38",
    "continues_active": PALETTE["region_only"],
}


def main() -> None:
    print("Loading data...")
    applications = load_applications()
    hexagons = load_hexagons()

    print("Preparing analysis universe...")
    prepared = prepare_analysis_hex_universe(applications, hexagons)
    hex_meta = prepared["hex_meta"]
    apps_clean = prepared["apps_clean"]
    print(
        f"  hexes={hex_meta['hex'].nunique()}, "
        f"apps={len(apps_clean)}, "
        f"treated={int(hex_meta['is_treated'].sum())}"
    )

    print("Building hex_activity_panel...")
    hex_activity_panel = build_hex_activity_panel(
        hex_meta=hex_meta,
        apps_clean=apps_clean,
        study_start=STUDY_START,
        study_end=STUDY_END,
    )
    print(
        f"  rows={len(hex_activity_panel):,}, "
        f"zero_days={(hex_activity_panel['n_orders']==0).sum():,}, "
        f"sum_orders={int(hex_activity_panel['n_orders'].sum()):,}"
    )
    panel_paths = save_activity_panel(hex_activity_panel, OUT_DIR)
    print("  saved:", {k: str(v) for k, v in panel_paths.items()})

    print("Window metrics 14/28/56...")
    metrics_list = [
        compute_closure_window_metrics(hex_activity_panel, window_days=w)
        for w in (14, 28, 56)
    ]
    metrics_by_window = pd.concat(metrics_list, ignore_index=True)
    metrics_by_window.to_csv(
        OUT_DIR / "hex_closure_diagnostics_by_window.csv", index=False
    )

    print("Primary empirical classification (28d)...")
    # Diagnostic thresholds — not universal business rules.
    print(
        f"  PRIMARY_WINDOW_DAYS={PRIMARY_WINDOW_DAYS}, "
        f"MIN_PRE_ORDERS={MIN_PRE_ORDERS}, "
        f"MIN_PRE_ACTIVE_DAYS={MIN_PRE_ACTIVE_DAYS}, "
        f"NEAR_CLOSED_RATIO={NEAR_CLOSED_RATIO}, "
        f"MAX_NEAR_CLOSED_ACTIVE_DAYS={MAX_NEAR_CLOSED_ACTIVE_DAYS}"
    )
    classification = classify_empirical_activity(
        metrics_by_window, window_days=PRIMARY_WINDOW_DAYS
    )
    classification.to_csv(
        OUT_DIR / "hex_empirical_closure_classification.csv", index=False
    )
    print(classification["empirical_status"].value_counts().to_string())

    print("Sensitivity grid...")
    sensitivity = run_closure_threshold_sensitivity(metrics_by_window)
    sensitivity.to_csv(OUT_DIR / "closure_threshold_sensitivity.csv", index=False)
    agreement = sensitivity.attrs.get("agreement")
    if agreement is not None and not agreement.empty:
        agreement.to_csv(
            OUT_DIR / "closure_threshold_sensitivity_agreement.csv", index=False
        )

    print("Metadata vs empirical...")
    crosstab = build_metadata_vs_empirical_crosstab(classification)
    crosstab.to_csv(OUT_DIR / "metadata_vs_empirical_closure.csv", index=False)

    groups = summarize_comparison_groups(classification, hex_activity_panel)
    groups["group_summary"].to_csv(
        OUT_DIR / "metadata_vs_empirical_group_summary.csv", index=False
    )
    groups["core_empirically_closed"].to_csv(
        OUT_DIR / "core_empirically_closed_hexagons.csv", index=False
    )
    groups["administratively_closed_but_active"].to_csv(
        OUT_DIR / "administratively_closed_but_active_hexagons.csv", index=False
    )
    print(groups["group_summary"].to_string(index=False))

    print("Selective attrition...")
    attrition = selective_attrition_diagnostics(hex_activity_panel)
    attrition.to_csv(OUT_DIR / "hex_selective_attrition_diagnostics.csv", index=False)

    print("Event-time aggregates...")
    event_time = build_event_time_activity(
        hex_activity_panel, classification, bootstrap_reps=200
    )
    event_time.to_csv(OUT_DIR / "hex_activity_event_time.csv", index=False)

    print("Figures...")
    _make_figures(hex_activity_panel, classification, event_time, sensitivity)

    # Headline counts for thesis sync
    n_admin_closed = int((classification["change_type"] == CLOSED).sum())
    n_admin_emp = int(
        (
            (classification["change_type"] == CLOSED)
            & (classification["empirical_status"] == EMPIRICAL_CLOSED)
        ).sum()
    )
    n_admin_active = int(
        (
            (classification["change_type"] == CLOSED)
            & (classification["post_orders"] > 0)
        ).sum()
    )
    n_core_emp = int(
        (
            classification["change_type"].isin(CORE_CHANGE_TYPES)
            & (classification["empirical_status"] == EMPIRICAL_CLOSED)
        ).sum()
    )
    headline = pd.DataFrame(
        [
            {
                "n_admin_closed": n_admin_closed,
                "n_admin_closed_empirical_closed": n_admin_emp,
                "n_admin_closed_orders_continue": n_admin_active,
                "n_core_empirical_closed": n_core_emp,
                "n_treated_classified": int(classification["hex"].nunique()),
                "primary_window_days": PRIMARY_WINDOW_DAYS,
                "min_pre_orders": MIN_PRE_ORDERS,
                "min_pre_active_days": MIN_PRE_ACTIVE_DAYS,
                "near_closed_ratio": NEAR_CLOSED_RATIO,
            }
        ]
    )
    headline.to_csv(OUT_DIR / "hex_closure_headline_counts.csv", index=False)
    print("\nHEADLINE")
    print(headline.to_string(index=False))
    print("Done.")


@with_plot_style
def _make_figures(panel, classification, event_time, sensitivity) -> None:
    # 1) event-time mean orders
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for gname, g in event_time.groupby("group"):
        ax.plot(
            g["relative_day"],
            g["mean_n_orders"],
            label=GROUP_LABELS.get(gname, gname),
            color=GROUP_COLORS.get(gname, PALETTE["text"]),
            linewidth=1.6,
        )
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("День относительно даты изменения")
    ax.set_ylabel("Среднее число заявок")
    ax.set_title("Динамика заявок: event-time")
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)
    save_figure(fig, FIG_DIR / "hex_activity_event_time_orders.pdf", preview_dpi=300)

    # 2) active share
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for gname, g in event_time.groupby("group"):
        ax.plot(
            g["relative_day"],
            g["active_share"],
            label=GROUP_LABELS.get(gname, gname),
            color=GROUP_COLORS.get(gname, PALETTE["text"]),
            linewidth=1.6,
        )
        ax.fill_between(
            g["relative_day"],
            g["active_share_ci_low"],
            g["active_share_ci_high"],
            color=GROUP_COLORS.get(gname, PALETTE["text"]),
            alpha=0.12,
            linewidth=0,
        )
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("День относительно даты изменения")
    ax.set_ylabel("Доля активных гексагонов")
    ax.set_title("Доля дней с заявками: event-time")
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)
    save_figure(
        fig, FIG_DIR / "hex_activity_event_time_active_share.pdf", preview_dpi=300
    )

    # 3) pre vs post scatter
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    type_colors = {
        "closed": "#7B8794",
        "region_and_workmode": PALETTE["region_and_workmode"],
        "region_only": PALETTE["region_only"],
        "workmode_only": PALETTE["workmode_only"],
        "opened": "#A0AEC0",
        "other": "#CBD5E0",
        "never_active": "#E2E8F0",
    }
    for ctype, g in classification.groupby("change_type"):
        ax.scatter(
            np.log1p(g["pre_orders"]),
            np.log1p(g["post_orders"]),
            s=12,
            alpha=0.45,
            label=ctype,
            color=type_colors.get(ctype, PALETTE["text_muted"]),
            edgecolors="none",
        )
    lim = max(
        np.log1p(classification["pre_orders"]).max(),
        np.log1p(classification["post_orders"]).max(),
    )
    ax.plot([0, lim], [0, lim], color=PALETTE["zero"], linewidth=0.9, linestyle="--")
    ax.set_xlabel("log(1 + заявки до изменения)")
    ax.set_ylabel("log(1 + заявки после изменения)")
    ax.set_title("Заявки до и после изменения (окно 28 дней)")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    style_axes(ax)
    save_figure(fig, FIG_DIR / "hex_pre_vs_post_orders.pdf", preview_dpi=300)

    # 4) last request relative distribution
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = classification["last_request_relative_day"].dropna()
    ax.hist(vals, bins=40, color=PALETTE["treated"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("Относительный день последней заявки")
    ax.set_ylabel("Число гексагонов")
    ax.set_title("Распределение дня последней заявки")
    style_axes(ax)
    save_figure(
        fig, FIG_DIR / "hex_last_request_relative_distribution.pdf", preview_dpi=300
    )

    # 5) heatmap metadata vs empirical
    pivot = (
        classification.groupby(["change_type", "empirical_status"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Число гексагонов"},
    )
    ax.set_xlabel("Эмпирический статус")
    ax.set_ylabel("Административный change_type")
    ax.set_title("Сопоставление административного и эмпирического статуса")
    save_figure(
        fig, FIG_DIR / "metadata_vs_empirical_closure_heatmap.pdf", preview_dpi=300
    )

    # 6) sensitivity
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for ratio, g in sensitivity.groupby("near_closed_ratio"):
        for min_pre, gg in g.groupby("min_pre_orders"):
            ax.plot(
                gg["window_days"],
                gg["n_empirical_closed"],
                marker="o",
                linewidth=1.4,
                label=f"pre≥{min_pre}, ratio={ratio:g}",
            )
    ax.set_xlabel("Окно, дней")
    ax.set_ylabel("Число empirical_closed")
    ax.set_title("Чувствительность классификации закрытия")
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    style_axes(ax)
    save_figure(fig, FIG_DIR / "closure_threshold_sensitivity.pdf", preview_dpi=300)


if __name__ == "__main__":
    main()
