"""Generate all publication figures used by ``thesis/main.tex`` from confirmed outputs.

This module never estimates a model or changes a sample. It only reads plotting
inputs produced by the empirical notebooks and applies the shared visual style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .plot_style import (
    PALETTE,
    TYPE_LABELS,
    TYPE_LINESTYLES,
    TYPE_MARKERS,
    TYPE_ORDER,
    contrasting_text_color,
    percent_formatter,
    plot_style,
    save_figure,
    style_axes,
)

OUTCOME_TITLES = {
    "t_available": "Event-study: время до первого доступного интервала",
    "sch_flg": "Event-study: назначение встречи",
    "meet_flg": "Event-study: состоявшаяся встреча",
    "success_within_20": "Event-study: успех в течение 20 дней",
    "utlz_within_25": "Event-study: утилизация в течение 25 дней",
}

EVENT_STEMS = {
    "t_available": "event_study_t_available_by_type",
    "sch_flg": "event_study_sch_by_type",
    "meet_flg": "event_study_meet_by_type",
    "success_within_20": "event_study_success_by_type",
    "utlz_within_25": "event_study_utlz_by_type",
}


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing plotting input: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _canonical_outcome(value: str) -> str:
    raw = str(value)
    if raw in {"utlz_flg", "utlz", "utilization"}:
        return "utlz_within_25"
    if raw in {"success_flg", "success"}:
        return "success_within_20"
    return raw


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> Path:
    return save_figure(fig, fig_dir / f"{stem}.pdf", preview_png=True, preview_dpi=240)


def plot_cohort_composition(root: Path, fig_dir: Path) -> Path:
    source = root / "outputs" / "cohort_diagnostics" / "composition_sch_n_hex.csv"
    frame = _read_csv(source).set_index("cohort")[TYPE_ORDER]
    shares = frame.div(frame.sum(axis=1), axis=0) * 100.0

    with plot_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        x = np.arange(len(shares))
        bottom = np.zeros(len(shares), dtype=float)
        for change_type in TYPE_ORDER:
            values = shares[change_type].to_numpy(dtype=float)
            color = PALETTE[change_type]
            bars = ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.64,
                label=TYPE_LABELS[change_type],
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            for index, (bar, value) in enumerate(zip(bars, values)):
                if value < 4.5:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[index] + value / 2,
                    f"{value:.1f}%".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=contrasting_text_color(color),
                )
            bottom += values

        ax.set_title("Состав CORE-когорт по типам изменения", pad=44, fontsize=11)
        ax.set_ylabel("Доля гексагонов, %")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.set_major_formatter(percent_formatter(100, 0))
        ax.set_xticks(x, shares.index.astype(str), rotation=0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            columnspacing=1.2,
            handlelength=1.4,
        )
        style_axes(ax)
        return _save(fig, fig_dir, "cohort_composition_core_shares")


def plot_pretrend(root: Path, fig_dir: Path, outcome: str) -> Path:
    candidates = [root / "outputs" / "empirical" / f"_pretrends_{outcome}.csv"]
    if outcome == "success_within_20":
        candidates.append(root / "outputs" / "empirical" / "_pretrends_success_flg.csv")
    source = next((path for path in candidates if path.exists()), candidates[0])
    frame = _read_csv(source)
    frame["outcome"] = frame["outcome"].map(_canonical_outcome)
    frame = frame.loc[frame["outcome"] == outcome].copy()
    if frame.empty and outcome == "success_within_20":
        # Legacy pretrend dumps may still label the series as success_flg.
        legacy = _read_csv(source)
        frame = legacy.loc[legacy["outcome"].astype(str).isin({"success_flg", "success"})].copy()
        frame["outcome"] = "success_within_20"
    if frame.empty:
        raise ValueError(f"No pretrend plotting rows for {outcome} in {source}")
    cohorts = sorted(frame["cohort"].astype(str).unique())
    title_metric = (
        "назначение встречи"
        if outcome == "sch_flg"
        else "успех в течение 20 дней"
    )

    with plot_style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 5.8),
            sharex=False,
            sharey=True,
            constrained_layout=True,
        )
        handles = []
        labels = []
        for ax, cohort in zip(axes.ravel(), cohorts):
            part = frame.loc[frame["cohort"].astype(str) == cohort].sort_values("event_week")
            min_x = float(part["event_week"].min())
            max_x = float(part["event_week"].max())
            ax.axvspan(-0.5, max_x + 0.5, color=PALETTE["post"], zorder=0)
            treated_line, = ax.plot(
                part["event_week"],
                part["treated"],
                color=PALETTE["treated"],
                marker="o",
                markersize=3.8,
                linewidth=1.35,
                label="Treated",
            )
            control_line, = ax.plot(
                part["event_week"],
                part["control"],
                color=PALETTE["control"],
                linestyle="--",
                marker="s",
                markerfacecolor="white",
                markeredgecolor=PALETTE["control"],
                markersize=3.8,
                linewidth=1.25,
                label="Control",
            )
            ax.axvline(-0.5, color=PALETTE["zero"], linestyle=(0, (3, 2)), linewidth=0.85)
            ax.set_xlim(min_x - 0.2, max_x + 0.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.set_title(cohort)
            ax.yaxis.set_major_formatter(percent_formatter(1.0, 0))
            style_axes(ax)
            if not handles:
                handles = [treated_line, control_line]
                labels = ["Treated", "Control"]

        for ax in axes.ravel()[len(cohorts) :]:
            ax.set_visible(False)
        fig.suptitle(f"Предоценочная динамика: {title_metric}", fontsize=11, y=1.05)
        fig.supxlabel("Неделя относительно изменения")
        fig.supylabel("Среднее значение")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)
        stem = f"pretrends_{'sch' if outcome == 'sch_flg' else 'success'}_by_cohort"
        return _save(fig, fig_dir, stem)


def plot_event_study(root: Path, fig_dir: Path, outcome: str) -> Path:
    source = root / "outputs" / "final" / "event_study_coefficients.csv"
    frame = _read_csv(source)
    frame["outcome"] = frame["outcome"].map(_canonical_outcome)
    frame = frame.loc[frame["outcome"] == outcome].copy()
    if frame.empty:
        raise ValueError(f"No event-study coefficients for {outcome} in {source}")
    scale = 1.0 if outcome == "t_available" else 100.0
    ylabel = "Изменение, дней" if outcome == "t_available" else "Коэффициент, п.п."
    y_low = float((frame["ci_low"] * scale).min())
    y_high = float((frame["ci_high"] * scale).max())
    y_pad = 0.08 * max(y_high - y_low, 0.1)

    with plot_style():
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 7.4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for ax, change_type in zip(axes, TYPE_ORDER):
            part = frame.loc[frame["change_type"] == change_type].sort_values("event_week")
            x = part["event_week"].to_numpy(dtype=float)
            y = part["coef"].to_numpy(dtype=float) * scale
            low = part["ci_low"].to_numpy(dtype=float) * scale
            high = part["ci_high"].to_numpy(dtype=float) * scale
            color = PALETTE[change_type]
            ax.axvspan(-0.5, float(x.max()) + 0.5, color=PALETTE["post"], zorder=0)
            if outcome == "utlz_within_25":
                ax.axvspan(2.5, float(x.max()) + 0.5, color="#E5E9ED", alpha=0.65, hatch="///", zorder=1)
            ax.axhline(0, color=PALETTE["zero"], linewidth=0.9, zorder=2)
            ax.axvline(-0.5, color=PALETTE["zero"], linestyle=(0, (3, 2)), linewidth=0.85, zorder=2)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - low, high - y]),
                fmt=TYPE_MARKERS[change_type],
                linestyle=TYPE_LINESTYLES[change_type],
                color=color,
                ecolor=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=3.8,
                linewidth=1.1,
                elinewidth=0.9,
                capsize=2.2,
                zorder=3,
            )
            ax.set_title(TYPE_LABELS[change_type], loc="left", pad=4)
            ax.set_ylim(y_low - y_pad, y_high + y_pad)
            ax.set_xticks(np.arange(-8, 9, 2))
            style_axes(ax)
        fig.suptitle(OUTCOME_TITLES[outcome], fontsize=11)
        fig.supxlabel("Неделя относительно изменения")
        fig.supylabel(ylabel)
        return _save(fig, fig_dir, EVENT_STEMS[outcome])


def plot_threshold_tradeoff(root: Path, fig_dir: Path) -> Path:
    source = root / "outputs" / "utilization_censoring" / "threshold_summary.csv"
    frame = _read_csv(source).sort_values("T")
    reference = frame.loc[frame["T"] == 25].iloc[0]

    with plot_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(
            frame["T"],
            frame["share_recorded_utilizations_within_T"],
            color=PALETTE["region_and_workmode"],
            marker="o",
            linewidth=1.45,
            markersize=4.0,
            label="Зафиксированные утилизации к H",
        )
        ax.plot(
            frame["T"],
            frame["eligible_share"],
            color=PALETTE["workmode_only"],
            linestyle="--",
            marker="s",
            markerfacecolor="white",
            linewidth=1.4,
            markersize=4.0,
            label="Заявки с follow-up не менее H дней",
        )
        ax.axvline(25, color=PALETTE["zero"], linestyle=(0, (3, 2)), linewidth=0.85)
        annotation = (
            f"H=25: {100 * reference['share_recorded_utilizations_within_T']:.1f}% / "
            f"{100 * reference['eligible_share']:.1f}%"
        ).replace(".", ",")
        ax.annotate(
            annotation,
            xy=(25, float(reference["eligible_share"])),
            xytext=(31, 0.73),
            fontsize=9.5,
            color=PALETTE["text_muted"],
            arrowprops={"arrowstyle": "->", "color": PALETTE["text_muted"], "lw": 0.75},
        )
        ax.set_title("Компромисс покрытия и зрелости выборки", pad=42, fontsize=11)
        ax.set_xlabel("Горизонт H, дней")
        ax.set_ylabel("Доля, %")
        ax.set_ylim(0.65, 1.0)
        ax.yaxis.set_major_formatter(percent_formatter(1.0, 0))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
        style_axes(ax)
        return _save(fig, fig_dir, "threshold_tradeoff_publication")


def plot_threshold_sensitivity(root: Path, fig_dir: Path) -> Path:
    source = root / "outputs" / "utilization_censoring" / "did_threshold_sensitivity.csv"
    frame = _read_csv(source)
    frame = frame.loc[(frame["status"] == "ok") & frame["avg_post_effect_pp"].notna()].copy()
    styles = {
        "natural": (PALETTE["region_and_workmode"], "o", "-", "Natural maturity"),
        "common_support": (PALETTE["workmode_only"], "s", "--", "Common support"),
    }

    with plot_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        for mode, (color, marker, linestyle, label) in styles.items():
            part = frame.loc[frame["sample_mode"] == mode].sort_values("T")
            y = part["avg_post_effect_pp"].to_numpy(dtype=float)
            low = part["ci_low_pp"].to_numpy(dtype=float)
            high = part["ci_high_pp"].to_numpy(dtype=float)
            ax.errorbar(
                part["T"],
                y,
                yerr=np.vstack([y - low, high - y]),
                fmt=marker,
                linestyle=linestyle,
                color=color,
                ecolor=color,
                markerfacecolor="white",
                markersize=4.2,
                linewidth=1.35,
                elinewidth=0.9,
                capsize=2.2,
                label=label,
            )
        ax.axhline(0, color=PALETTE["zero"], linewidth=0.9)
        ax.axvline(25, color=PALETTE["zero"], linestyle=(0, (3, 2)), linewidth=0.85)
        ax.text(25.5, ax.get_ylim()[1] * 0.88, "H=25", fontsize=9.5, color=PALETTE["text_muted"])
        ax.set_title("Чувствительность DiD-оценки к горизонту H", pad=42, fontsize=11)
        ax.set_xlabel("Горизонт H, дней")
        ax.set_ylabel("Средний post-эффект, п.п.")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
        style_axes(ax)
        return _save(fig, fig_dir, "did_threshold_sensitivity_natural_vs_common")


def generate_all(project_root: str | Path) -> list[Path]:
    """Generate the ten publication PDFs included by the thesis."""
    root = Path(project_root).resolve()
    fig_dir = root / "figures" / "empirical"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_cohort_composition(root, fig_dir),
        plot_pretrend(root, fig_dir, "sch_flg"),
        plot_pretrend(root, fig_dir, "success_within_20"),
    ]
    generated.extend(plot_event_study(root, fig_dir, outcome) for outcome in EVENT_STEMS)
    generated.extend(
        [
            plot_threshold_tradeoff(root, fig_dir),
            plot_threshold_sensitivity(root, fig_dir),
        ]
    )
    return generated
