# Исследовательские ноутбуки

Каталог содержит 15 отслеживаемых Jupyter-ноутбуков. Канонические численные
результаты хранятся в `outputs/final/`, а порядок полного запуска задаётся
списком `NOTEBOOKS` в `scripts/run_full_recalculation.py`.

Статическая проверка всех отслеживаемых файлов:

```bash
poetry run python scripts/audit_release_notebooks.py
```

## Индекс и порядок запуска

| Порядок | Ноутбук | Назначение | Статус | Основные результаты | Канонический |
| ---: | --- | --- | --- | --- | :---: |
| вне полного запуска | `data_structure_eda.ipynb` | Разведка структуры данных, дат и панели | вспомогательный | компактная описательная диагностика | нет |
| 1 | `conversionfunnel_revised.ipynb` | Описательная воронка и условные конверсии | основной | `outputs/final/conversion_funnel_*.csv` | да |
| 2 | `hexagonsclosingcheck.ipynb` | Разведка активности и закрытия гексагонов | диагностический | промежуточные сводки активности | частично |
| вне полного запуска | `hexagon_open_close_diagnostics.ipynb` | Структурная диагностика открытия/закрытия и timing активности | диагностический | `outputs/hex_open_close/` | нет |
| 3 | `cohort_diagnostics.ipynb` | Состав когорт и поддержка событийного времени | диагностический | `outputs/cohort_diagnostics/` | да, диагностика |
| 4 | `preestimationDID.ipynb` | Предоценочные тренды и поддержка дизайна | диагностический | предтренды в `outputs/empirical/` | да, диагностика |
| 5 | `utilization_censoring_threshold_diagnostics.ipynb` | Выбор 25-дневного горизонта утилизации | диагностический | `outputs/utilization_censoring/` | да, диагностика |
| 6 | `final_empirical_recalculation.ipynb` | Основные событийные модели PanelOLS, дозовая спецификация и ATT(g,t) | основной | основная часть `outputs/final/*.csv` | да |
| вне полного запуска | `dose_support_diagnostics.ipynb` | Support `R_h × Δw_h` для дозовой модели без переоценки регрессий | диагностический | `outputs/final/dose_support_*.csv` | нет |
| 7 | `conditional_funnel_did.ipynb` | DiD для условных переходов воронки | основной | `outputs/final/conditional_funnel_summary.csv` | да |
| 8 | `speed_conversion_association.ipynb` | Непричинная связь времени ожидания с конверсиями | основной | `outputs/final/speed_conversion_*.csv` | да |
| 9 | `honest_did_sensitivity.ipynb` | Чувствительность оценок к нарушениям параллельных трендов | диагностический | `outputs/honest_did/` | да, чувствительность |
| 10 | `harmonized_empirical_figures.ipynb` | Единая генерация графиков из проверенных CSV | вспомогательный | `figures/empirical/` | да, фигуры |
| вне полного запуска | `speed_did_extension.ipynb` | Отдельная диагностика времени до доступного интервала | вспомогательный | `outputs/empirical/speed_did_summary_revised.csv` | нет |
| вне полного запуска | `success_timing_diagnostics.ipynb` | Multi-horizon диагностика timing успешной встречи | диагностический | `outputs/success_timing_diagnostics/` | нет |

После `hexagonsclosingcheck.ipynb` оркестратор дополнительно запускает
`scripts/run_hex_closure_diagnostics.py` и
`scripts/run_closure_did_sensitivity.py`. После ноутбуков строятся фигуры и
выполняется строгая проверка результатов.

Полный запуск:

```bash
poetry run python scripts/run_full_recalculation.py
```

Для него необходимы исходные CSV в `data/raw/`, `Rscript` в `PATH` и R-пакеты
`HonestDiD`, `jsonlite`, `readr`.
