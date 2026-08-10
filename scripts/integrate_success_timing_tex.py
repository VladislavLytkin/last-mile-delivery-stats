# -*- coding: utf-8 -*-
"""Insert success-timing subsection into thesis/main.tex using saved CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "thesis" / "main.tex"
OUT = ROOT / "outputs" / "success_timing_diagnostics"

TYPE_RU = {
    "region_and_workmode": "Смена рег.+реж.",
    "region_only": "Только регион",
    "workmode_only": "Только режим",
}
TYPE_ORDER = ["region_and_workmode", "region_only", "workmode_only"]


def fmt_pp(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_pp(lo)};\\,{fmt_pp(hi)}]"


def build_subsection() -> str:
    nat = pd.read_csv(OUT / "success_horizon_did.csv")
    cs = pd.read_csv(OUT / "success_horizon_did_common_support.csv")
    mt = pd.read_csv(OUT / "success_horizon_multiple_testing.csv")
    mt_nat = mt.loc[mt["spec"] == "natural_maturity"].copy()
    n_raw = int((mt_nat["raw_pvalue"] < 0.05).sum())
    n_holm = int((mt_nat["holm_pvalue"] < 0.05).sum())
    n_bh = int((mt_nat["bh_qvalue"] < 0.05).sum())

    rows = []
    for ctype in TYPE_ORDER:
        for H in [3, 7, 14, 25]:
            r = cs.loc[(cs["change_type"] == ctype) & (cs["horizon_days"] == H)].iloc[0]
            n = nat.loc[(nat["change_type"] == ctype) & (nat["horizon_days"] == H)].iloc[0]
            # Compare natural vs common for compactness column
            n_est = float(n["DID_post_estimate_pp"])
            c_est = float(r["post_estimate_pp"])
            if abs(n_est - c_est) < 0.75:
                rob = "близко к natural"
            elif abs(c_est) > abs(n_est) + 1:
                rob = "усилен / изменён"
            else:
                rob = "ослаблен / изменён"
            effect = (
                f"${fmt_pp(float(r['post_estimate_pp']))}\\ "
                f"{fmt_ci(float(r['ci_low_pp']), float(r['ci_high_pp']))}$"
            )
            rows.append(
                f"{TYPE_RU[ctype]} & ${int(H)}$ & {effect} & "
                f"${float(r['pretrend_pvalue']):.3f}$ & {rob} \\\\"
            )

    table_body = "\n".join(rows)

    # Per-type narrative from final logic
    return rf"""
\subsection{{Дополнительная диагностика времени до успешной встречи}}
\label{{subsec:success_timing_diag}}

Основная оценка по \texttt{{success\_within\_20}} характеризует изменение
вероятности достижения успешной встречи в фиксированном горизонте, но не
различает, произошло ли событие через 3, 7, 14 или 25 дней. Поэтому
дополнительно исследуется временная структура успеха как
\textit{{exploratory multi-horizon diagnostic}}, без замены основного
20-дневного estimand.

Пусть
\[
T_i^{{\mathrm{{success}}}}
=
\texttt{{first\_success\_dttm}}_i
-
\texttt{{request\_timestamp}}_i.
\]
Условное распределение \(T_i^{{\mathrm{{success}}}}\) среди заявок с
\texttt{{success\_flg}}=1 используется только описательно: conditioning on
success является conditioning on a potentially treatment-affected outcome и
не допускает причинной интерпретации.

Описательно основная масса успешных встреч возникает в первые дни после
заявки, а распределение имеет выраженный правый хвост. Это мотивирует
переход к бинарным исходам
\[
Y_i^{{\mathrm{{success}}}}(H)
=
\mathbb{{I}}\!\left(
0\le T_i^{{\mathrm{{success}}}}\le H
\right),
\quad
H\in\{{3,7,14,25\}},
\]
где заявка включается в оценку горизонта \(H\) только при наличии не менее
\(H\) дней последующего наблюдения. Диагностика длинного хвоста около
100--110 дней показывает, что его происхождение однозначно не установлено по
доступным полям; соответствующий кластер не используется для причинных
выводов.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{cumulative_success_curves.pdf}}
\caption{{Накопленная вероятность достижения успешной встречи в зависимости
от времени после заявки. Правая панель показывает описательную разницу
post--pre для treated CORE; эта разница не является DiD-оценкой.}}
\label{{fig:success_cum_curves}}
\end{{figure}}

На рисунке~\ref{{fig:success_cum_curves}} сырая post--pre разница по
накопленной вероятности положительна, однако не имеет причинной
интерпретации: она не очищена от календарных изменений и различий состава
выборки.

Для причинной диагностики оценивается та же TWFE event-study спецификация,
что и в основной оценке (FE гексагона и календарного дня, cluster SE по
гексагону, baseline \(w=-1\), окно примерно \([-8;+8]\)), отдельно для
\(Y_i^{{\mathrm{{success}}}}(H)\). Поскольку при разных \(H\) естественная
зрелость меняет состав sample и post support, основной robustness-график
строится на \textit{{common-support}} выборке заявок, зрелых как минимум
для \(H=25\), с общей поддержкой post-недель.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{multi_horizon_did_common_support.pdf}}
\caption{{Common-support multi-horizon event-study DiD-оценки для вероятности
достижения успешной встречи в пределах \(H\) дней. Точки --- взвешенные
средние post-коэффициентов, отрезки --- 95\%-е доверительные интервалы.
Символ <<*>> отмечает спецификации с joint pretrend \(p<0.05\) и
\textit{{не}} означает статистическую значимость эффекта treatment.}}
\label{{fig:success_multi_horizon_cs}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Common-support multi-horizon DiD для \(Y^{{\mathrm{{success}}}}(H)\)}}
\label{{tab:success_timing_common}}
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l c >{{\centering\arraybackslash}}X c >{{\raggedright\arraybackslash}}X@{{}}}}
\toprule
Тип изменения & \(H\) & Post effect, п.п.\ [95\% CI] & Pretrend \(p\) & vs natural maturity \\
\midrule
{table_body}
\bottomrule
\end{{tabularx}}
\begin{{minipage}}{{0.98\textwidth}}
\scriptsize
\textit{{Примечание.}} Оценки на выборке, зрелой для \(H=25\); одна и та же
поддержка post-недель для всех горизонтов. SE кластеризованы по гексагону;
средний post-эффект --- взвешенное среднее post-коэффициентов через полную
ковариационную матрицу. Individual horizon-specific \(p\)-values носят
exploratory характер, так как одновременно рассматриваются 3 типа изменений
и 4 горизонта.
\end{{minipage}}
\end{{table}}

Дополнительная временная декомпозиция \textbf{{не выявляет единого устойчивого
изменения времени достижения успешной встречи}} для всех типов treatment.
Для simultaneous смены региона и режима на коротких горизонтах наблюдаются
локальные отрицательные оценки (особенно \(H=7\)), которые ослабевают к
\(H=25\) и на common-support сохраняют тот же качественный профиль; для
\(H=3\) joint pretrend отвергается (\(p<0.05\)), поэтому короткий горизонт
не интерпретируется причинно. Для только смены режима на natural-maturity
выборка даёт ранний положительный сигнал на \(H=3\), но на common-support
95\%-й интервал уже включает ноль, а на длинных горизонтах оценка
приближается к нулю --- это совместимо с возможным ранним ускорением, которое
не проходит robustness. Для только смены региона natural-maturity даёт
отрицательную оценку на \(H=25\), а common-support усиливает отрицательные
оценки на коротких горизонтах; ввиду чувствительности к определению sample и
множественным проверкам сигнал трактуется как неустойчивый exploratory
finding, а не как подтверждённое longer-run снижение.

Семейство из 12 natural-maturity post-эффектов даёт {n_raw} номинальных
результатов с raw \(p<0.05\), однако после поправок Holm и Benjamini--Hochberg
ни один эффект не остаётся значимым (Holm: {n_holm}; BH FDR \(q<0.05\):
{n_bh}). Поэтому даже локальные краткосрочные сигналы интерпретируются как
exploratory evidence о возможном перераспределении успешных встреч во
времени, а не как новый единый причинный эффект.

""".replace(
        "{table_body}", table_body
    ).replace(
        "{n_raw}", str(n_raw)
    ).replace(
        "{n_holm}", str(n_holm)
    ).replace(
        "{n_bh}", str(n_bh)
    )


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    subsection = build_subsection()

    # 1) Abstract sentence (only if absent)
    abs_sent = (
        "Дополнительная multi-horizon диагностика успешной встречи показывает, "
        "что отдельные краткосрочные изменения её вероятности не образуют "
        "устойчивого общего эффекта на длинных горизонтах."
    )
    if "multi-horizon диагностика успешной встречи" not in tex:
        old_abs_end = (
            "устойчивости, а связь скорости с конверсиями интерпретируется как\n"
            "статистическая ассоциация, но не как причинный эффект.\n"
            "\\end{abstract}"
        )
        new_abs_end = (
            "устойчивости, а связь скорости с конверсиями интерпретируется как\n"
            "статистическая ассоциация, но не как причинный эффект. "
            + abs_sent
            + "\n\\end{abstract}"
        )
        if old_abs_end not in tex:
            raise SystemExit("abstract anchor not found")
        tex = tex.replace(old_abs_end, new_abs_end)

    # 2) Methods bullet
    methods_item = (
        "    \\item дополнительная multi-horizon event-study диагностика времени "
        "до успешной встречи с построением бинарных исходов достижения "
        "успешной встречи в пределах нескольких горизонтов зрелости;\n"
    )
    if "multi-horizon event-study диагностика времени" not in tex:
        anchor = (
            "    \\item анализ чувствительности ATT\\((g,t)\\)-оценок к возможным "
            "нарушениям предпосылки параллельных трендов методом HonestDiD "
            "Рамбачана и Рота.\n"
            "\\end{itemize}"
        )
        repl = (
            "    \\item анализ чувствительности ATT\\((g,t)\\)-оценок к возможным "
            "нарушениям предпосылки параллельных трендов методом HonestDiD "
            "Рамбачана и Рота;\n"
            + methods_item
            + "\\end{itemize}"
        )
        if anchor not in tex:
            raise SystemExit("methods anchor not found")
        tex = tex.replace(anchor, repl)

    # 3) Conditioning note after T^success definition
    if "conditioning on a potentially treatment-affected outcome" not in tex:
        needle = (
            "Данная метрика рассчитывается только для заявок с \\(success\\_flg_i = 1\\) "
            "и показывает число календарных дней от создания заявки до успешного "
            "завершения встречи. Она не используется как основная метрика скорости, "
            "поскольку включает только успешно завершённые заявки."
        )
        if needle not in tex:
            raise SystemExit("T^success paragraph not found")
        tex = tex.replace(
            needle,
            needle
            + " Условное распределение времени среди successful applications "
            "имеет только описательный статус, поскольку conditioning on success "
            "является conditioning on a potentially treatment-affected outcome.",
        )

    # 4) Multi-horizon definition after success_within_20 maturity paragraph
    if "Y_i^{\\mathrm{success}}(H)" not in tex and "Y_i^{\\mathrm{success}}(H)" not in tex:
        pass
    horizon_block = r"""
Дополнительно к основному 20-дневному estimand для диагностики timing
вводятся горизонт-специфические исходы
\[
Y_i^{\mathrm{success}}(H)
=
\mathbb{I}\!\left(
0\le first\_success\_dttm_i-request\_timestamp_i\le H
\right),
\quad
H\in\{3,7,14,25\},
\]
причём заявка включается в оценку горизонта \(H\) только при наличии не менее
\(H\) дней последующего наблюдения. Это вероятность успеха \emph{within}
\(H\) дней, а не исходный lifetime-флаг \texttt{success\_flg}. Сравнение
нескольких \(H\) позволяет отличить изменение timing от изменения eventual
conversion.
"""
    if "H\\in\\{3,7,14,25\\}" not in tex and r"H\in\{3,7,14,25\}" not in tex:
        anchor = (
            "не смешивается с 20-дневным исходом.\n"
        )
        if anchor not in tex:
            raise SystemExit("maturity success anchor not found")
        tex = tex.replace(
            anchor,
            "не смешивается с 20-дневным исходом.\n" + horizon_block,
        )

    # 5) Insert subsection before dose
    if "subsec:success_timing_diag" not in tex:
        dose = "\\subsection{Дополнительная дозовая спецификация изменения режима работы}"
        if dose not in tex:
            raise SystemExit("dose subsection not found")
        tex = tex.replace(dose, subsection + "\n" + dose)

    # 6) Limitations paragraph
    lim = r"""
Отдельное ограничение относится к multi-horizon анализу успешной встречи. Он
подвержен трём дополнительным ограничениям. Во-первых, horizon-specific
maturity при разных \(H\) меняет доступный follow-up и потенциально состав
post support; common-support спецификация используется как robustness к этому
ограничению. Во-вторых, одновременно исследуются несколько горизонтов и типов
изменений, поэтому отдельные nominal \(p<0.05\) следует рассматривать как
exploratory. В-третьих, условное распределение времени только среди
successful applications нельзя интерпретировать причинно, поскольку treatment
потенциально влияет на вероятность попадания в successful sample.
"""
    if "multi-horizon анализу успешной встречи" not in tex:
        marker = "Наконец, модель не учитывает все возможные ненаблюдаемые факторы"
        if marker not in tex:
            raise SystemExit("limitations marker not found")
        tex = tex.replace(marker, lim + "\n" + marker)

    # 7) Conclusion paragraph
    concl = (
        "Дополнительная декомпозиция success по горизонту показала, что отсутствие "
        "единого эффекта на агрегированную вероятность успешной встречи не исключает "
        "различий в краткосрочной динамике. Для отдельных типов изменений наблюдаются "
        "локальные ранние положительные или отрицательные оценки, которые ослабевают "
        "на более длинных горизонтах. При этом результаты неоднородны, часть "
        "спецификаций чувствительна к предпосылкам предтрендов, поддержке выборки и "
        "множественным проверкам. Поэтому они интерпретируются как дополнительная "
        "диагностика возможного перераспределения успешных встреч во времени, "
        "а не как новый единый причинный эффект.\n"
    )
    if "декомпозиция success по горизонту" not in tex:
        # find near end of conclusion
        marker = (
            "Практически решения по каждому типу изменения следует проверять отдельными пилотами "
            "с заранее зафиксированными outcome и горизонтом наблюдения."
        )
        if marker not in tex:
            raise SystemExit("conclusion marker not found")
        tex = tex.replace(marker, marker + "\n" + concl)

    # 8) Reproducibility
    if "success_timing_diagnostics.ipynb" not in tex:
        rep = (
            "Диагностический multi-horizon анализ времени до успешной встречи "
            "воспроизводится ноутбуком \\texttt{notebooks/success\\_timing\\_diagnostics.ipynb} "
            "и скриптом \\texttt{scripts/run\\_success\\_timing\\_robustness.py}; "
            "машинно-читаемые результаты находятся в "
            "\\path{outputs/success\\_timing\\_diagnostics/}.\n"
        )
        m = (
            "Манифест фиксирует\nверсии пакетов, commit hash и SHA-256 входных данных.\n"
        )
        if m not in tex:
            raise SystemExit("repro marker not found")
        tex = tex.replace(m, m + rep)

    # 9) Hypothesis outcomes note for group 2
    if "Multi-horizon timing diagnostic" not in tex:
        g2 = (
            "    \\item Для группы~2 в 12 основных заранее определённых категориальных\n"
            "    спецификациях 95\\%-е доверительные интервалы среднего post-эффекта\n"
            "    включают ноль. На стандартных уровнях значимости не получено\n"
            "    оснований отвергнуть нулевую гипотезу; универсальный\n"
            "    однонаправленный эффект изменений не подтверждён. Это не является\n"
            "    доказательством того, что фактический эффект строго равен нулю.\n"
            "    Выводы формулируются отдельно по CORE-типам, outcomes и фиксированным\n"
            "    горизонтам, а дозовая модель используется как дополнительная\n"
            "    robustness-диагностика.\n"
        )
        g2_new = (
            "    \\item Для группы~2 в 12 основных заранее определённых категориальных\n"
            "    спецификациях 95\\%-е доверительные интервалы среднего post-эффекта\n"
            "    включают ноль. На стандартных уровнях значимости не получено\n"
            "    оснований отвергнуть нулевую гипотезу; универсальный\n"
            "    однонаправленный эффект изменений не подтверждён. Это не является\n"
            "    доказательством того, что фактический эффект строго равен нулю.\n"
            "    Выводы формулируются отдельно по CORE-типам, outcomes и фиксированным\n"
            "    горизонтам, а дозовая модель используется как дополнительная\n"
            "    robustness-диагностика. Multi-horizon timing diagnostic: единого\n"
            "    устойчивого эффекта не выявлено; обнаружены отдельные краткосрочные\n"
            "    неоднородные сигналы по типам изменений.\n"
        )
        if g2 not in tex:
            raise SystemExit("group2 outcomes anchor not found")
        tex = tex.replace(g2, g2_new)

    TEX.write_text(tex, encoding="utf-8")
    print("Updated", TEX)


if __name__ == "__main__":
    main()
