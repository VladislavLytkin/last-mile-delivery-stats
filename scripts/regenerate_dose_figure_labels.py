"""Regenerate dose forest / event-study figure labels from saved CSV (no re-estimation)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from last_mile.plot_style import PALETTE, save_figure, style_axes, with_plot_style  # noqa: E402

FIG_DIR = ROOT / "figures" / "empirical"
OUT_FINAL = ROOT / "outputs" / "final"

DOSE_TERMS = ["post_R", "post_W_plus", "post_W_minus"]
DOSE_TERM_LABELS = {
    "post_R": "Смена региона (β_R)",
    "post_W_plus": "+1 раб. день (β₊)",
    "post_W_minus": "−1 раб. день (β₋)",
}
DOSE_TERM_COLORS = {
    "post_R": PALETTE["region_and_workmode"],
    "post_W_plus": PALETTE["workmode_only"],
    "post_W_minus": PALETTE["region_only"],
    "R": PALETTE["region_and_workmode"],
    "W_plus": PALETTE["workmode_only"],
    "W_minus": PALETTE["region_only"],
}
EVENT_PANEL_TITLES = {
    "R": r"$\beta_{R,k}$ (усл. регион)",
    "W_plus": r"$\beta_{+,k}$ (+1 день, усл.)",
    "W_minus": r"$\beta_{-,k}$ (−1 день, усл.)",
}


@with_plot_style
def plot_dose_forest(dose_did_summary: pd.DataFrame, outcomes: list[str], filename: str, ylabel: str, title: str) -> None:
    sub = dose_did_summary[
        dose_did_summary["outcome"].isin(outcomes)
        & (dose_did_summary["status"] == "ok")
        & (dose_did_summary["cohort_specification"] == "outcome_specific")
    ].copy()
    if sub.empty:
        print(f"[SKIP] forest plot {filename}")
        return

    outcome_order = [o for o in outcomes if o in set(sub["outcome"])]
    rows_plot = []
    y = 0.0
    for outcome in outcome_order:
        for term in DOSE_TERMS:
            row = sub[(sub["outcome"] == outcome) & (sub["term"] == term)]
            if row.empty:
                continue
            rows_plot.append((y, outcome, term, row.iloc[0]))
            y += 1
        y += 0.4

    fig_h = max(3.5, 0.55 * len(rows_plot) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    for y_pos, outcome, term, r in rows_plot:
        color = DOSE_TERM_COLORS[term]
        ax.errorbar(
            r["estimate"],
            y_pos,
            xerr=[[r["estimate"] - r["ci_lower"]], [r["ci_upper"] - r["estimate"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=5,
        )
    ax.axvline(0, color=PALETTE["zero"], linewidth=1.0)
    ax.set_yticks([r[0] for r in rows_plot])
    ax.set_yticklabels([f"{r[1]}: {DOSE_TERM_LABELS[r[2]]}" for r in rows_plot])
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    style_axes(ax, horizontal_grid=False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.65, alpha=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / filename, preview_png=True, preview_dpi=240)
    print("Saved", FIG_DIR / filename)


@with_plot_style
def plot_dose_event_study(dose_event_study: pd.DataFrame, outcome: str, filename: str) -> None:
    sub = dose_event_study[dose_event_study["outcome"] == outcome].copy()
    if sub.empty:
        print(f"[SKIP] event study {outcome}")
        return
    terms = ["R", "W_plus", "W_minus"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    est = sub[sub["estimated"] == True]  # noqa: E712
    lows = est["ci_lower"].dropna().tolist()
    highs = est["ci_upper"].dropna().tolist()
    y_lim = None
    if lows and highs:
        y_min, y_max = min(lows), max(highs)
        pad = 0.12 * (y_max - y_min if y_max > y_min else 0.05)
        y_lim = (y_min - pad, y_max + pad)

    for ax, term in zip(axes, terms):
        panel = sub[sub["term"] == term].sort_values("event_week")
        ax.axhline(0, color=PALETTE["zero"], linewidth=1.0)
        ax.axvline(0, linestyle="--", color=PALETTE["zero"], linewidth=1.0)
        plot_df = panel[panel["estimated"] == True].copy()  # noqa: E712
        if not plot_df.empty:
            x = plot_df["event_week"].to_numpy()
            y = plot_df["estimate"].to_numpy()
            yerr = np.vstack(
                [
                    y - plot_df["ci_lower"].to_numpy(),
                    plot_df["ci_upper"].to_numpy() - y,
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color=DOSE_TERM_COLORS[term],
                linewidth=1.5,
                markersize=4,
                capsize=3,
            )
        ax.scatter([-1], [0], marker="s", s=28, color=PALETTE["text_muted"], zorder=5)
        ax.set_title(EVENT_PANEL_TITLES[term])
        ax.set_xlabel("Относительная неделя")
        style_axes(ax)
        if y_lim is not None:
            ax.set_ylim(y_lim)
    unit = "дней" if outcome == "t_available" else "п.п."
    axes[0].set_ylabel(f"Коэффициент ({unit}), база −1")
    fig.suptitle(f"Дозовая event-study: {outcome}", y=1.03, fontsize=12)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / filename, preview_png=True, preview_dpi=240)
    print("Saved", FIG_DIR / filename)


def main() -> None:
    dose_did_summary = pd.read_csv(OUT_FINAL / "dose_did_summary.csv")
    dose_event_study = pd.read_csv(OUT_FINAL / "dose_event_study.csv")

    plot_dose_forest(
        dose_did_summary,
        ["sch_flg", "meet_flg", "success_within_20", "utlz_within_25"],
        "dose_coefficients_conversions.pdf",
        "Оценка, п.п. (условный компонент)",
        "Дозовая DiD: условные компоненты (конверсии)",
    )
    plot_dose_forest(
        dose_did_summary,
        ["t_available"],
        "dose_coefficients_speed.pdf",
        "Оценка, дни (условный компонент)",
        "Дозовая DiD: условные компоненты (t_available)",
    )
    for outcome in [
        "meet_flg",
        "t_available",
        "sch_flg",
        "success_within_20",
        "utlz_within_25",
    ]:
        filename = (
            "dose_event_study_success_flg.pdf"
            if outcome == "success_within_20"
            else f"dose_event_study_{outcome}.pdf"
        )
        plot_dose_event_study(dose_event_study, outcome, filename)


if __name__ == "__main__":
    main()
