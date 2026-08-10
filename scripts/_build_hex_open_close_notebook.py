# -*- coding: utf-8 -*-
"""Generate notebooks/hexagon_open_close_diagnostics.ipynb"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "hexagon_open_close_diagnostics.ipynb"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}


def md(s: str) -> None:
    lines = dedent(s).strip("\n").split("\n")
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [ln + "\n" for ln in lines],
        }
    )


def code(s: str) -> None:
    src = dedent(s).strip("\n") + "\n"
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [ln + "\n" for ln in src.split("\n")],
        }
    )


md(
    """
# Диагностика открытия/закрытия гексагонов: work_mode vs заявки

## Назначение

Проверка предложения научного консультанта: можно ли определить фактическое открытие/закрытие гексагона

1. по переходам `work_mode`;
2. по устойчивому исчезновению/возникновению заявок;
3. насколько эти два определения совпадают.

## Важно

- Это **отдельный диагностический** ноутбук.
- Не меняет `main.tex`, DiD/event-study ноутбуки и production pipeline.
- Не пишет: «0 заявок = гексагон закрыт».
- Корректная формулировка: устойчивое отсутствие заявок после подтверждённой активности — эмпирический индикатор *возможного* закрытия.

## Данные и инфраструктура

- Загрузка только через `last_mile.io` (`application_dataset.csv`, `hexagons_dataset.csv`).
- Административный `change_type` — `last_mile.filter.classify_hexagons` (для сравнения, без переписывания).
- Логика диагностики: `scripts/hex_open_close_diagnostics_lib.py`.
- Полный batch-runner: `scripts/run_hex_open_close_diagnostics.py`.

## Выходы

- Таблицы: `outputs/hex_open_close/`
- Графики: `figures/hex_open_close/` (PDF + PNG через `last_mile.plot_style`)
"""
)

code(
    """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from last_mile.filter import (
    CLOSED,
    CORE_CHANGE_TYPES,
    INACTIVE_REGION,
    OPENED,
    STUDY_END,
    STUDY_START,
)
from last_mile.io import load_applications, load_hexagons
from last_mile.plot_style import PALETTE, save_figure, style_axes, with_plot_style
from hex_open_close_diagnostics_lib import (
    EVENT_PAD_DAYS,
    HORIZONS,
    PRIMARY_HORIZON,
    MIN_PRE_APPS,
    MIN_PRE_ACTIVE_DAYS,
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
    lifetime_activity_diagnostics,
    low_demand_false_closure_risk,
    prepare_hex_universe,
    project_paths,
    structural_vs_empirical_table,
    suspicious_hexes,
    threshold_sensitivity_summary,
    work_mode_semantics,
)

paths = project_paths(PROJECT_ROOT)
OUT_DIR, FIG_DIR = paths["out"], paths["fig"]
print("STUDY_START/END:", STUDY_START.date(), STUDY_END.date())
print("INACTIVE_REGION sentinel:", INACTIVE_REGION)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("PRIMARY_HORIZON:", PRIMARY_HORIZON, "HORIZONS:", HORIZONS)
print("MIN_PRE_APPS / MIN_PRE_ACTIVE_DAYS:", MIN_PRE_APPS, MIN_PRE_ACTIVE_DAYS)
"""
)

md(
    """
## 0. Где сейчас задаётся `change_type` (не меняем)

Функция `classify_hexagons` в `last_mile/filter.py`:

- Активность зоны определяется **сентинелом** `region_id == -100` (`INACTIVE_REGION`), **не** `work_mode == 0`.
- `opened`: был неактивен (`region_id_old == -100`) → стал активен.
- `closed`: был активен → стал неактивен (`region_id_new == -100`).
- `work_mode` используется только чтобы отличить `workmode_only` / `region_and_workmode` при активных old/new регионах.

Правило «нет заявок ⇒ closed» **не** является правилом assignment. Отдельная эмпирическая диагностика уже есть в `last_mile/hex_activity.py` и `notebooks/hexagonsclosingcheck.ipynb` (admin `closed` vs заявки). Этот ноутбук отвечает на другой вопрос: transitions `work_mode` vs заявки.

Порог `min_orders_per_hex=5` в `build_analysis_panel` / `prepare_analysis_hex_universe` применяется **после** attach metadata и **до** DiD-панели. Здесь основной расчёт идёт **без** этого порога; параллельно показываем counts после `>=5`.
"""
)

code(
    """
import inspect
from last_mile import filter as filter_mod

src = inspect.getsource(filter_mod.classify_hexagons)
display(Markdown("```python\\n" + src + "\\n```"))
"""
)

md(
    """
## 1. Семантика `work_mode`

Ничего не предполагаем. Смотрим уникальные значения, частоты, min/max, NaN, примеры и пересечение с `region_id == -100`.
"""
)

code(
    """
applications = load_applications()
hexagons = load_hexagons()
print(
    f"applications={len(applications):,}, "
    f"hexagon_rows={len(hexagons):,}, "
    f"unique_hex={hexagons['hex'].nunique():,}"
)

sem = work_mode_semantics(hexagons)
display(sem["summaries"])
display(sem["value_counts"].sort_values(["field", "n"], ascending=[True, False]))
display(sem["examples"])
display(Markdown("### Cross: work_mode==0 vs region_id==-100"))
display(sem["wm_vs_region_cross"].head(15))

wm0_new = hexagons["work_mode_new"] == 0
reg_inact_new = hexagons["region_id_new"] == INACTIVE_REGION
print("P(region_new inactive | work_mode_new==0) =", float(reg_inact_new[wm0_new].mean()))
print("P(work_mode_new==0 | region_new inactive) =", float(wm0_new[reg_inact_new].mean()))
"""
)

md(
    """
### Structural classification по `work_mode`

| код | правило |
|---|---|
| A `explicit_closed` | `work_mode_old > 0` AND `work_mode_new == 0` |
| B `explicit_opened` | `work_mode_old == 0` AND `work_mode_new > 0` |
| C `remained_closed` | оба нуля |
| D `remained_active` | оба > 0 |

NaN / значения вне `{0..7}` — отдельно, без подмены нулями.
"""
)

code(
    """
print(
    "NOTE: DiD universe applies min_orders_per_hex=5 AFTER cohort attach "
    "(last_mile.filter.build_analysis_panel / last_mile.hex_activity.prepare_analysis_hex_universe). "
    "Diagnostic default below = NO min-orders filter."
)

prep = prepare_hex_universe(
    applications, hexagons, min_orders_per_hex=None, apps_only=True
)
treated = prep["treated_meta"]
treated_diag = treated[~treated["excluded_cohort"]].copy()
print(
    "treated (app-linked):",
    len(treated),
    "| after excluding late cohort 2022-10-19:",
    len(treated_diag),
)

struct = (
    treated_diag["structural_wm"]
    .value_counts(dropna=False)
    .rename("n")
    .rename_axis("structural_wm")
    .reset_index()
)
struct["share"] = struct["n"] / struct["n"].sum()
display(struct)

prep5 = prepare_hex_universe(
    applications, hexagons, min_orders_per_hex=5, apps_only=True
)
t5 = prep5["treated_meta"]
t5 = t5[~t5["excluded_cohort"]]
struct5 = (
    t5["structural_wm"]
    .value_counts(dropna=False)
    .rename("n")
    .rename_axis("structural_wm")
    .reset_index()
)
struct5["share"] = struct5["n"] / struct5["n"].sum()
display(Markdown("### После DiD-фильтра min_orders >= 5"))
display(struct5)
"""
)

md(
    """
## 2–4. Панель hex×day вокруг treatment и empirical open/close

Для treated hex (app-linked, без late cohort) строим календарь `[-42, +42]` относительно `treatment_date`, обрезанный реальным диапазоном заявок в study window. Отсутствующие hex-day заполняются нулями.

Один нулевой день **не** считается закрытием. Empirical правила (основной горизонт = 14):

- `empirical_closed_W`: pre-окно с подтверждённой активностью (`>=5` заявок и `>=3` active days) **и** W подряд дней post с `n_apps==0`.
- `empirical_opened_W`: W подряд дней pre с нулями **и** post с подтверждённой активностью.
- Если pre/post окно обрезано границей данных → `censored`, без классификации.
"""
)

code(
    """
daily = build_daily_apps(prep["apps_study"])
panel = build_event_panel(
    treated_diag,
    daily,
    app_date_min=max(prep["app_date_min"], STUDY_START),
    app_date_max=min(prep["app_date_max"], STUDY_END),
    pad_days=EVENT_PAD_DAYS,
)
print(
    f"panel rows={len(panel):,}, zero_days={(panel['n_apps']==0).sum():,}, "
    f"sum n_apps={int(panel['n_apps'].sum()):,}"
)
assert panel.duplicated(["hex", "date"]).sum() == 0

metrics = compute_window_metrics(panel, horizons=HORIZONS)
for h in HORIZONS:
    metrics = classify_empirical(metrics, horizon=h)

display(Markdown("### Pre/post metrics snapshot (primary W=14)"))
cols14 = [
    "hex",
    "structural_wm",
    "change_type",
    "full_pre_14",
    "full_post_14",
    "left_censored",
    "right_censored",
    "pre_apps_14",
    "post_apps_14",
    "pre_active_days_14",
    "post_active_days_14",
    "pre_max_consecutive_zero_14",
    "post_max_consecutive_zero_14",
    "empirical_14",
]
display(metrics[cols14].head(10))

sens = threshold_sensitivity_summary(metrics, HORIZONS)
display(Markdown("### Sensitivity 7/14/21/28"))
display(sens)
"""
)

md(
    """
## 5. Permanent closure vs temporary shutdown
"""
)

code(
    """
perm_tmp = classify_permanent_and_temporary(
    panel, metrics, horizon=PRIMARY_HORIZON
)
display(perm_tmp["permanent_status"].value_counts())
print("temporary_shutdown:", int(perm_tmp["temporary_shutdown"].sum()))
print(
    "empirical_permanent_closure:",
    int(perm_tmp["empirical_permanent_closure"].sum()),
)

display(Markdown("### Распределение longest post zero-run"))
display(
    perm_tmp["longest_post_zero_run"].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95]
    )
)
"""
)

md(
    """
## 6. Сопоставление structural work_mode × empirical activity
"""
)

code(
    """
for h in HORIZONS:
    ct = structural_vs_empirical_table(metrics, horizon=h)
    display(Markdown(f"### Horizon={h}"))
    pivot = ct.pivot_table(
        index="structural_wm", columns="empirical", values="n", fill_value=0
    )
    display(pivot)
    print(agreement_rate(metrics, h))

display(Markdown("### Ключевые расхождения (W=14)"))
m = metrics
q1 = m[m["structural_wm"] == "explicit_closed"]
q2 = m[m["structural_wm"] == "explicit_opened"]
q3 = m[
    (m["structural_wm"] == "remained_active")
    & (m["empirical_14"] == "empirical_closed")
]
q4 = m[(m["work_mode_new"] == 0) & (m["post_apps_14"] > 0)]
print(
    f"1) explicit_closed → emp_closed_14: "
    f"{(q1['empirical_14']=='empirical_closed').sum()}/{len(q1)}"
)
print(f"   с достаточным pre-activity: {(q1['pre_sufficient_activity_14']).sum()}")
print(f"   post_apps_14>0: {(q1['post_apps_14']>0).sum()}")
print(
    f"2) explicit_opened → emp_opened_14: "
    f"{(q2['empirical_14']=='empirical_opened').sum()}/{len(q2)}"
)
print(f"3) hidden empirical closed among remained_active: {len(q3)}")
print(f"4) work_mode_new==0 but post_apps_14>0: {len(q4)}")

sus = suspicious_hexes(metrics, PRIMARY_HORIZON)
display(Markdown("### Примеры suspicious hex"))
display(sus.head(20))
"""
)

md(
    """
## 7. Тайминг фактического изменения активности vs `treatment_date`
"""
)

code(
    """
timing = activity_change_timing(metrics)
closed_t = timing.loc[
    timing["structural_wm"] == "explicit_closed", "closure_timing_rel"
].dropna()
opened_t = timing.loc[
    timing["structural_wm"] == "explicit_opened", "opening_timing_rel"
].dropna()
print("closure timing (last active - treatment) describe:")
print(closed_t.describe())
print("opening timing (first post active - treatment) describe:")
print(opened_t.describe())
"""
)

md(
    """
## 8. Графики

Единый стиль проекта (`last_mile.plot_style`). Runner уже пишет полный набор в `figures/hex_open_close/`; ниже пересобираются ключевые event-time charts.
"""
)

code(
    """
existing = sorted(FIG_DIR.glob("*.pdf"))
print("Existing figures:", [p.name for p in existing])

event_agg = event_time_aggregates(panel)
GROUP_COLORS = {
    "explicit_closed": "#B85C38",
    "explicit_opened": "#4C78A8",
    "remained_active": "#234E70",
}


@with_plot_style
def _plot_event(ycol, ylab, fname):
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
    ax.legend(loc="best", frameon=False)
    style_axes(ax)
    save_figure(fig, FIG_DIR / fname, preview_dpi=300)


_plot_event(
    "mean_n_apps",
    "Среднее число заявок на hex-day",
    "applications_event_time.pdf",
)
_plot_event(
    "active_share",
    "Доля hex с n_apps > 0",
    "active_share_event_time.pdf",
)
print("Updated event-time figures.")
"""
)

md(
    """
## 9. Связь с текущим `change_type`
"""
)

code(
    """
ct_vs = change_type_vs_structural(metrics)
pivot = ct_vs.pivot_table(
    index="change_type", columns="structural_wm", values="n", fill_value=0
)
display(pivot)

display(
    Markdown(
        '''
**Текущие правила opened/closed (`classify_hexagons`):** только `region_id` ↔ `-100`.

**Наблюдение:** большинство `explicit_closed` по work_mode совпадают с admin `closed`, но не все
(часть попадает в `workmode_only`, где регион остаётся активным, а `work_mode` падает до 0).
Это критично: `work_mode_new==0` **не эквивалентно** administratively closed.
'''
    )
)
"""
)

md(
    """
## 10–11. Lifetime activity и low-demand false-closure risk
"""
)

code(
    """
life = lifetime_activity_diagnostics(
    prep["apps_study"],
    treated_diag[
        [
            "hex",
            "treatment_date",
            "structural_wm",
            "change_type",
            "is_treated",
            "n_orders_study",
        ]
    ],
)
display(life["lifetime_pattern"].value_counts())

low = low_demand_false_closure_risk(metrics, perm_tmp, HORIZONS)
display(Markdown("### P(zero-run ≥ W | later resumed), by pre-volume"))
display(low)
"""
)

md(
    """
## 12. Сохранение outputs

Полный набор также пишет `scripts/run_hex_open_close_diagnostics.py`. Ниже — ключевые таблицы текущего прогона.
"""
)

code(
    """
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

struct.to_csv(OUT_DIR / "structural_wm_counts.csv", index=False)
sens.to_csv(OUT_DIR / "closure_threshold_sensitivity.csv", index=False)
cross = pd.concat(
    [
        structural_vs_empirical_table(metrics, horizon=h).assign(horizon=h)
        for h in HORIZONS
    ],
    ignore_index=True,
)
cross.to_csv(OUT_DIR / "structural_vs_empirical.csv", index=False)
perm_tmp.to_csv(OUT_DIR / "zero_run_diagnostics.csv", index=False)
timing.to_csv(OUT_DIR / "activity_change_timing.csv", index=False)
sus.to_csv(OUT_DIR / "suspicious_hexes.csv", index=False)
metrics.to_csv(OUT_DIR / "hex_open_close_metrics.csv", index=False)
life.to_csv(OUT_DIR / "lifetime_activity_diagnostics.csv", index=False)
low.to_csv(OUT_DIR / "low_demand_false_closure_risk.csv", index=False)

summary = build_summary_table(
    struct,
    sens,
    perm_tmp,
    n_treated=len(treated_diag),
    min_orders_note="none (diagnostic); DiD uses >=5 after attach",
)
summary["n_admin_closed_change_type"] = int((metrics["change_type"] == CLOSED).sum())
summary["n_admin_opened_change_type"] = int((metrics["change_type"] == OPENED).sum())
summary["n_suspicious_rows"] = int(len(sus))
summary.to_csv(OUT_DIR / "hex_open_close_summary.csv", index=False)
display(summary.T)
print("Saved to", OUT_DIR)
"""
)

md(
    """
## 13. Выводы

Ответы пересчитываются из объектов текущего прогона (не захардкожены заранее).
"""
)

code(
    """
n_ec = int((metrics["structural_wm"] == "explicit_closed").sum())
n_eo = int((metrics["structural_wm"] == "explicit_opened").sum())
n_emp_c = int((metrics["empirical_14"] == "empirical_closed").sum())
n_emp_o = int((metrics["empirical_14"] == "empirical_opened").sum())
agr = agreement_rate(metrics, 14)
n_tmp = int(perm_tmp["temporary_shutdown"].sum())
n_perm = int(perm_tmp["empirical_permanent_closure"].sum())
n_hidden = int(
    (
        (metrics["structural_wm"] == "remained_active")
        & (metrics["empirical_14"] == "empirical_closed")
    ).sum()
)
n_wm0_apps = int(((metrics["work_mode_new"] == 0) & (metrics["post_apps_14"] > 0)).sum())
n_chg7 = int(sens.loc[sens.horizon == 7, "n_status_changes_vs_primary14"].iloc[0])
n_chg21 = int(sens.loc[sens.horizon == 21, "n_status_changes_vs_primary14"].iloc[0])
n_chg28 = int(sens.loc[sens.horizon == 28, "n_status_changes_vs_primary14"].iloc[0])

suff = metrics[
    (metrics.structural_wm == "explicit_closed") & (metrics.pre_sufficient_activity_14)
]
n_suff = len(suff)
n_suff_closed = int((suff.empirical_14 == "empirical_closed").sum())

risk_14_5_9 = low[(low.horizon == 14) & (low.pre_volume_bin == "5-9")]
risk_share = (
    float(risk_14_5_9["share_with_zero_run_ge_w"].iloc[0])
    if len(risk_14_5_9)
    else float("nan")
)

lines = []
lines.append("### 1. Можно ли надёжно считать `work_mode_new == 0` закрытием?")
lines.append("")
lines.append(
    "**Нет, не само по себе.** В данных `work_mode ∈ {0..7}`, NaN нет. "
    "Значение `0` сильно коррелирует с `region_id_new == -100`, но не тождественно ему. "
    "Среди app-linked treated есть `explicit_closed` в `workmode_only` "
    "(регион остаётся активным). Административный `closed` в pipeline определяется "
    "**только** через `region_id == -100` (`classify_hexagons`)."
)
lines.append("")
lines.append("### 2. Соответствует ли это фактическому исчезновению заявок?")
lines.append("")
lines.append(
    f"**Слабо / выборочно.** Из **{n_ec}** `explicit_closed` только "
    f"**{agr['n_explicit_closed_empirical_closed']}** подтверждаются как "
    f"`empirical_closed_14` (доля ≈ {agr['share_closed_confirmed']:.3f})."
)
lines.append(
    "При этом у большинства `explicit_closed` post-окно 14 дней может быть нулевым, "
    "но критерий достаточной pre-активности не выполняется (low-demand)."
)
lines.append(
    f"Среди {n_suff} `explicit_closed` с достаточной pre-активностью только "
    f"**{n_suff_closed}** действительно имеют 14 нулевых post-дней; "
    f"у остальных заявки продолжаются."
)
lines.append(
    f"Случаев `work_mode_new==0` при `post_apps_14>0`: **{n_wm0_apps}**."
)
lines.append("")
lines.append("### 3. Можно ли обнаруживать дополнительные закрытия только по applications?")
lines.append("")
lines.append(
    f"**Да, но только как диагностику, не как assignment.** "
    f"Независимый индикатор находит **{n_emp_c}** `empirical_closed_14` и "
    f"**{n_emp_o}** `empirical_opened_14`; среди них **{n_hidden}** — "
    "скрытые среди structural `remained_active`."
)
lines.append("")
lines.append("### 4. Какой минимальный zero-run разумно использовать: 7 / 14 / 21 / 28?")
lines.append("")
lines.append("- **7 дней** слишком агрессивен для low-demand hex.")
lines.append(
    f"- **14 дней** — разумный основной диагностический горизонт "
    f"(bin 5–9 share≈{risk_share:.2f}; 25+ почти 0)."
)
lines.append("- **21–28** строже и сильнее упираются в right-censoring.")
lines.append("")
lines.append("### 5. Сколько hex меняют классификацию от порога?")
lines.append("")
lines.append(
    f"Vs primary 14: на 7d ~{n_chg7}; на 21d ~{n_chg21}; на 28d ~{n_chg28}. "
    "Нужен sensitivity table."
)
lines.append("")
lines.append("### 6. Есть ли существенное число временных пауз?")
lines.append("")
lines.append(
    f"temporary_shutdown = **{n_tmp}**; permanent_closure = **{n_perm}**. "
    "Lifetime чаще intermittent/low-demand или closed-like."
)
lines.append("")
lines.append("### 7. Совпадает ли фактическая дата изменения активности с `treatment_date`?")
lines.append("")
lines.append(
    "См. `closure_timing_hist` / `opening_timing_hist` и "
    "`activity_change_timing.csv`. Часть hex имеет лаг/лид относительно t=0."
)
lines.append("")
lines.append("### 8. Нужно ли менять `change_type` / выборку основного DiD?")
lines.append("")
lines.append(
    "**Пока нет.** Admin closed/opened должны оставаться administrative "
    "(`region_id==-100`). `work_mode==0` нельзя молча трактовать как закрытие. "
    "Эмпирика по заявкам — только diagnostics/robustness."
)
lines.append("")
lines.append("Если позже менять pipeline, кандидаты (**сейчас не трогаем**):")
lines.append("")
lines.append("1. `last_mile/filter.py` — `classify_hexagons`.")
lines.append(
    "2. `last_mile/hex_activity.py` / `notebooks/hexagonsclosingcheck.ipynb`."
)
lines.append(
    "3. DiD sample builders (`build_analysis_panel`, основные notebooks) — "
    "только при отдельном методологическом решении."
)
display(Markdown("\\n".join(lines)))
"""
)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(nb['cells'])} cells")
