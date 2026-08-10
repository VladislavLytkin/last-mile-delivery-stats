# -*- coding: utf-8 -*-
"""
Run work_mode open/close diagnostics (applications vs structural work_mode).

Writes:
  outputs/hex_open_close/*.csv
  figures/hex_open_close/*.{pdf,png}

Does not modify main DiD notebooks, main.tex, or production change_type rules.
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
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hex_open_close_diagnostics_lib import (  # noqa: E402
    EVENT_PAD_DAYS,
    HORIZONS,
    PRIMARY_HORIZON,
    activity_change_timing,
    agreement_rate,
    build_daily_apps,
    build_event_panel,
    build_summary_table,
    change_type_vs_structural,
    classify_empirical,
    classify_permanent_and_temporary,
    compute_window_metrics,
    event_time_aggregates,
    heatmap_matrix,
    lifetime_activity_diagnostics,
    load_raw,
    low_demand_false_closure_risk,
    prepare_hex_universe,
    project_paths,
    structural_vs_empirical_table,
    suspicious_hexes,
    threshold_sensitivity_summary,
    work_mode_semantics,
)
from last_mile.filter import (  # noqa: E402
    CLOSED,
    CORE_CHANGE_TYPES,
    INACTIVE_REGION,
    OPENED,
    STUDY_END,
    STUDY_START,
)
from last_mile.plot_style import PALETTE, save_figure, style_axes, with_plot_style  # noqa: E402

GROUP_COLORS = {
    "explicit_closed": "#B85C38",
    "explicit_opened": "#4C78A8",
    "remained_active": "#234E70",
    "remained_closed": "#7B8794",
}


def _structural_counts(df: pd.DataFrame) -> pd.DataFrame:
    vc = (
        df["structural_wm"]
        .value_counts(dropna=False)
        .rename("n")
        .rename_axis("structural_wm")
        .reset_index()
    )
    vc["share"] = vc["n"] / vc["n"].sum()
    return vc


@with_plot_style
def make_figures(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    timing: pd.DataFrame,
    event_agg: pd.DataFrame,
    fig_dir: Path,
) -> None:
    # A/B event-time mean apps and active share
    for ycol, ylab, fname in [
        ("mean_n_apps", "Среднее число заявок на hex-day", "applications_event_time"),
        ("active_share", "Доля hex с n_apps > 0", "active_share_event_time"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for gname, g in event_agg.groupby("structural_wm"):
            ax.plot(
                g["rel_day"],
                g[ycol],
                label=gname,
                color=GROUP_COLORS.get(gname, PALETTE["text"]),
                linewidth=1.6,
            )
        ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
        ax.set_xlabel("День относительно treatment_date")
        ax.set_ylabel(ylab)
        ax.set_title(ylab + " по structural work_mode")
        ax.legend(loc="best", frameon=False)
        style_axes(ax)
        # Split closed/opened overlays already on same plot; also save group-specific means
        save_figure(fig, fig_dir / f"{fname}.pdf", preview_dpi=300)

    # Separate closed / opened mean apps for required filenames
    for group, fname in [
        ("explicit_closed", "applications_event_time_closed"),
        ("explicit_opened", "applications_event_time_opened"),
    ]:
        g = event_agg[event_agg["structural_wm"] == group]
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        if not g.empty:
            ax.plot(g["rel_day"], g["mean_n_apps"], color=GROUP_COLORS[group], linewidth=1.7)
        ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
        ax.set_xlabel("День относительно treatment_date")
        ax.set_ylabel("Среднее число заявок на hex-day")
        ax.set_title(f"Поток заявок: {group}")
        style_axes(ax)
        save_figure(fig, fig_dir / f"{fname}.pdf", preview_dpi=300)

    # C/D heatmaps
    for group, mode, fname in [
        ("explicit_closed", "closed", "closure_heatmap"),
        ("explicit_opened", "opened", "opening_heatmap"),
    ]:
        mat, _ = heatmap_matrix(panel, structural_group=group, mode=mode)
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        if mat.empty:
            ax.text(0.5, 0.5, "Нет наблюдений", ha="center", va="center")
            ax.set_axis_off()
        else:
            # indicator or log1p
            plot_mat = np.log1p(mat.to_numpy(dtype=float))
            sns.heatmap(
                plot_mat,
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "log1p(n_apps)"},
                xticklabels=10,
                yticklabels=False,
            )
            # mark t=0 column
            cols = list(mat.columns)
            if 0 in cols:
                x0 = cols.index(0) + 0.5
                ax.axvline(x0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlabel("rel_day")
            ax.set_ylabel("hex (sorted)")
            ax.set_title(f"Heatmap активности: {group}")
        save_figure(fig, fig_dir / f"{fname}.pdf", preview_dpi=300)

    # E/F timing histograms
    closed_t = timing.loc[
        timing["structural_wm"] == "explicit_closed", "closure_timing_rel"
    ].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if len(closed_t):
        ax.hist(closed_t, bins=40, color=GROUP_COLORS["explicit_closed"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("last active day − treatment_date (дни)")
    ax.set_ylabel("Число гексагонов")
    ax.set_title("Тайминг исчезновения заявок (explicit_closed)")
    style_axes(ax)
    save_figure(fig, fig_dir / "closure_timing_hist.pdf", preview_dpi=300)

    opened_t = timing.loc[
        timing["structural_wm"] == "explicit_opened", "opening_timing_rel"
    ].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if len(opened_t):
        ax.hist(opened_t, bins=40, color=GROUP_COLORS["explicit_opened"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("first active day post − treatment_date (дни)")
    ax.set_ylabel("Число гексагонов")
    ax.set_title("Тайминг появления заявок (explicit_opened)")
    style_axes(ax)
    save_figure(fig, fig_dir / "opening_timing_hist.pdf", preview_dpi=300)


def main() -> dict:
    paths = project_paths(ROOT)
    out_dir, fig_dir = paths["out"], paths["fig"]

    print("Loading applications / hexagons via last_mile.io ...")
    applications, hexagons = load_raw()
    print(f"  apps={len(applications):,}, hex_rows={len(hexagons):,}")

    print("Section 1: work_mode semantics (full hexagon file) ...")
    sem = work_mode_semantics(hexagons)
    sem["value_counts"].to_csv(out_dir / "work_mode_value_counts.csv", index=False)
    sem["summaries"].to_csv(out_dir / "work_mode_summaries.csv", index=False)
    sem["examples"].to_csv(out_dir / "work_mode_examples.csv", index=False)
    sem["wm_vs_region_cross"].to_csv(out_dir / "work_mode_vs_region_cross.csv", index=False)
    print(sem["summaries"].to_string(index=False))
    print("\nwork_mode vs region_id==-100 (top rows):")
    print(sem["wm_vs_region_cross"].head(12).to_string(index=False))

    print("\nPreparing app-linked hex universe (NO min_orders>=5 filter) ...")
    print(
        "NOTE: build_analysis_panel / prepare_analysis_hex_universe apply "
        "min_orders_per_hex=5 AFTER cohort attach. This diagnostic intentionally "
        "starts WITHOUT that threshold; a parallel count with >=5 is reported below."
    )
    prep = prepare_hex_universe(
        applications, hexagons, min_orders_per_hex=None, apps_only=True
    )
    hex_meta = prep["hex_meta"]
    treated = prep["treated_meta"]
    # Focus diagnostics on non-excluded cohorts for fair post windows
    treated_diag = treated[~treated["excluded_cohort"]].copy()
    print(
        f"  hex_meta={len(hex_meta):,}, treated={len(treated):,}, "
        f"treated_diag(excl. late cohort)={len(treated_diag):,}"
    )

    struct_all = _structural_counts(treated_diag)
    struct_all.to_csv(out_dir / "structural_wm_counts.csv", index=False)
    print("\nStructural work_mode (treated, app-linked, no min_orders):")
    print(struct_all.to_string(index=False))

    # Parallel DiD-like threshold
    prep5 = prepare_hex_universe(
        applications, hexagons, min_orders_per_hex=5, apps_only=True
    )
    treated5 = prep5["treated_meta"]
    treated5 = treated5[~treated5["excluded_cohort"]]
    struct5 = _structural_counts(treated5)
    struct5.to_csv(out_dir / "structural_wm_counts_min5.csv", index=False)
    print("\nStructural work_mode after DiD min_orders>=5:")
    print(struct5.to_string(index=False))

    print("\nBuilding daily apps + event panel [-42,+42] ...")
    daily = build_daily_apps(prep["apps_study"])
    panel = build_event_panel(
        treated_diag,
        daily,
        app_date_min=max(prep["app_date_min"], STUDY_START),
        app_date_max=min(prep["app_date_max"], STUDY_END),
        pad_days=EVENT_PAD_DAYS,
    )
    print(f"  panel rows={len(panel):,}, zero_days={(panel['n_apps']==0).sum():,}")

    print("Window metrics + empirical labels 7/14/21/28 ...")
    metrics = compute_window_metrics(panel, horizons=HORIZONS)
    for h in HORIZONS:
        metrics = classify_empirical(metrics, horizon=h)
    metrics.to_csv(out_dir / "hex_open_close_metrics.csv", index=False)

    sens = threshold_sensitivity_summary(metrics, HORIZONS)
    sens.to_csv(out_dir / "closure_threshold_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    print("Permanent / temporary shutdown ...")
    perm_tmp = classify_permanent_and_temporary(panel, metrics, horizon=PRIMARY_HORIZON)
    perm_tmp.to_csv(out_dir / "zero_run_diagnostics.csv", index=False)
    print(
        f"  temporary_shutdown={int(perm_tmp['temporary_shutdown'].sum())}, "
        f"permanent={int(perm_tmp['empirical_permanent_closure'].sum())}, "
        f"right_censored_cannot_claim="
        f"{int((perm_tmp['permanent_status']=='right_censored_cannot_claim').sum())}"
    )
    # Distribution of post zero-run lengths
    zr = (
        perm_tmp["longest_post_zero_run"]
        .value_counts()
        .rename_axis("longest_post_zero_run")
        .rename("n_hex")
        .reset_index()
        .sort_values("longest_post_zero_run")
    )
    zr.to_csv(out_dir / "post_zero_run_length_distribution.csv", index=False)

    print("Structural vs empirical crosstabs ...")
    cross_frames = []
    for h in HORIZONS:
        ct = structural_vs_empirical_table(metrics, horizon=h)
        ct["horizon"] = h
        cross_frames.append(ct)
        agr = agreement_rate(metrics, h)
        print(f"  h={h}: {agr}")
    cross = pd.concat(cross_frames, ignore_index=True)
    cross.to_csv(out_dir / "structural_vs_empirical.csv", index=False)

    ct_struct = change_type_vs_structural(metrics)
    ct_struct.to_csv(out_dir / "change_type_vs_structural_wm.csv", index=False)

    # Specific questions 1-4 examples
    m14 = metrics.copy()
    q1 = m14[m14["structural_wm"] == "explicit_closed"]
    q2 = m14[m14["structural_wm"] == "explicit_opened"]
    q3 = m14[
        (m14["structural_wm"] == "remained_active")
        & (m14["empirical_14"] == "empirical_closed")
    ]
    q4 = m14[(m14["work_mode_new"] == 0) & (m14["post_apps_14"] > 0)]
    print(
        f"\nQ1 explicit_closed confirmed emp_closed_14: "
        f"{(q1['empirical_14']=='empirical_closed').sum()}/{len(q1)}"
    )
    print(
        f"Q2 explicit_opened confirmed emp_opened_14: "
        f"{(q2['empirical_14']=='empirical_opened').sum()}/{len(q2)}"
    )
    print(f"Q3 hidden empirical closures among remained_active: {len(q3)}")
    print(f"Q4 work_mode_new==0 but post_apps_14>0: {len(q4)}")

    timing = activity_change_timing(metrics)
    timing.to_csv(out_dir / "activity_change_timing.csv", index=False)

    sus = suspicious_hexes(metrics, PRIMARY_HORIZON)
    sus.to_csv(out_dir / "suspicious_hexes.csv", index=False)
    print(f"suspicious rows: {len(sus)}")

    print("Lifetime activity diagnostics (full study span) ...")
    # lifetime on treated_diag only to limit size — still ~treated x days
    life = lifetime_activity_diagnostics(
        prep["apps_study"], treated_diag[["hex", "treatment_date", "structural_wm", "change_type", "is_treated", "n_orders_study"]]
    )
    life.to_csv(out_dir / "lifetime_activity_diagnostics.csv", index=False)
    print(life["lifetime_pattern"].value_counts().to_string())

    print("Low-demand false closure risk ...")
    low = low_demand_false_closure_risk(metrics, perm_tmp, HORIZONS)
    low.to_csv(out_dir / "low_demand_false_closure_risk.csv", index=False)
    print(low.to_string(index=False))

    print("Event-time aggregates + figures ...")
    event_agg = event_time_aggregates(panel)
    event_agg.to_csv(out_dir / "event_time_aggregates.csv", index=False)
    make_figures(panel, metrics, timing, event_agg, fig_dir)

    summary = build_summary_table(
        struct_all,
        sens,
        perm_tmp,
        n_treated=len(treated_diag),
        min_orders_note="none (diagnostic); DiD uses >=5 after attach",
    )
    # enrich with change_type reference
    summary["n_admin_closed_change_type"] = int((metrics["change_type"] == CLOSED).sum())
    summary["n_admin_opened_change_type"] = int((metrics["change_type"] == OPENED).sum())
    summary["n_suspicious_rows"] = int(len(sus))
    summary["inactive_region_sentinel"] = INACTIVE_REGION
    summary["core_change_types"] = ",".join(CORE_CHANGE_TYPES)
    summary.to_csv(out_dir / "hex_open_close_summary.csv", index=False)

    # Document change_type rule location
    rule_note = pd.DataFrame(
        [
            {
                "file": "last_mile/filter.py",
                "function": "classify_hexagons",
                "opened_rule": "region_id_old == -100 AND region_id_new != -100",
                "closed_rule": "region_id_old != -100 AND region_id_new == -100",
                "work_mode_role": "only detects whether work_mode_old != work_mode_new for workmode_only / region_and_workmode",
                "apps_zero_equals_closed": False,
                "empirical_module": "last_mile/hex_activity.py (separate post-hoc diagnostics; does not assign change_type)",
                "min_orders_filter": "last_mile/filter.py::build_analysis_panel min_orders_per_hex=5",
            }
        ]
    )
    rule_note.to_csv(out_dir / "change_type_rule_reference.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.T.to_string(header=False))
    print(f"\nOutputs: {out_dir}")
    print(f"Figures: {fig_dir}")
    return {
        "summary": summary,
        "sens": sens,
        "metrics": metrics,
        "perm_tmp": perm_tmp,
        "struct": struct_all,
        "suspicious": sus,
        "sem": sem,
        "paths": paths,
    }


if __name__ == "__main__":
    main()
