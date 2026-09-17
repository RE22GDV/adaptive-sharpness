"""Ukrainian-labelled figures, built from the published JSON and nothing else.

The English figures in `docs/figures/` are written by `render_results.py`
alongside the tables.  These are the same measurements captioned in Ukrainian
for `docs/STUDY_UA_REPORT.md`, kept as a separate tool so that a change to one
language's captions cannot silently alter the other's.

Two rules this file follows, both of them the result of a defect:

*   **Every panel states how many recordings it covers.**  Rank correlation
    against the spot reference exists on two recordings; adjacent-step
    discrimination exists on eight.  A table that mixed the two was published
    and read as covering eight, and the conclusion drawn from it was wrong.
    The axis label carries the count so the panel cannot be quoted without it.

*   **Nothing here opens a frame.**  The input is `reports/*.json`, which holds
    measurements only.  The recordings are photographs of the operator's room
    and never leave the machine, so every figure this produces is safe to
    publish.

    python3 tools/figures_ua.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "figure.dpi": 140, "savefig.dpi": 180,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .17, "axes.axisbelow": True, "svg.fonttype": "none",
})
BLUE, TEAL, ORANGE, RED, GRAY = "#2563a6", "#07867d", "#d47722", "#b84751", "#778295"

NAMES = {
    "six_shipped": "Шість метрик", "only_laplacian": "Лапласіан",
    "only_tenengrad": "Тененград", "only_brenner": "Бреннер",
    "only_wavelet": "Вейвлетна", "only_fourier": "Фур'є",
    "only_edge_width": "Ширина меж",
}
ORDER = list(NAMES)

#: File stems, in figure order.  Keeping them here rather than deriving them
#: from the titles means a renamed caption does not rename a published image.
STEMS = (
    "corpus", "before_after", "singles_adj", "adj_heatmap", "plateaus",
    "saturation", "fair_comparison", "factorial", "normalisation", "fixes",
    "agreement", "leave_one_out", "analysis_width", "reference_correlations",
    "reference_peaks", "reliability_deltas", "runtime", "pair_ordering",
    "factorial_effects",
)


def run(reports: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    D = {p.stem: json.loads(p.read_text(encoding="utf-8"))
         for p in reports.glob("*.json")}
    missing = {"ablation_singles", "ablation_factorial", "ablation_factorial_altref",
               "ablation_normalisation", "ablation_fixes", "ablation_agreement",
               "ablation_metrics", "protocol_study", "before_after",
               "fair_comparison", "reference_sensitivity"} - set(D)
    if missing:
        raise SystemExit(f"missing reports: {sorted(missing)}")

    RUNS = list(D["ablation_singles"]["runs"])
    RN = {
        k: ("Умови " + str(i + 1) if k.startswith("cond")
            else "Точкове джерело " + ("1" if "111545" in k else "2")
            if k.startswith("point")
            else "Мала фактурність" if "plain" in k else "Багата фактура")
        for i, k in enumerate(RUNS)
    }
    manifest: list[dict[str, object]] = []

    def save(fig, n, title, source, note):
        stem = f"{n:02d}_{STEMS[n - 1]}"
        fig.suptitle(title, fontsize=16, fontweight="bold", x=.02, ha="left")
        fig.text(.02, .012, note, fontsize=9, color="#485568")
        fig.tight_layout(rect=(0, .18 if n == 14 else .065, 1, .91), pad=1.6)
        for ext in ("png", "svg"):
            fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append(
            {"number": n, "stem": stem, "title": title, "source": source, "note": note}
        )

    def hbar(ax, labels, vals, xlabel, color=BLUE, fmt=".3f", xlim=None):
        y = np.arange(len(vals))
        bars = ax.barh(y, vals, color=color, height=.65)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(
                    format(v, fmt), (v, rect.get_y() + rect.get_height() / 2),
                    xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9,
                )
        ax.set_xlim(*xlim) if xlim else ax.margins(x=.18)
        ax.grid(axis="y", visible=False)

    def agg(name, key, order):
        return [D[name]["aggregate"][v][key] for v in order]

    S, F = D["ablation_singles"], D["ablation_factorial"]
    A, P = D["ablation_factorial_altref"], D["protocol_study"]
    B, C = D["before_after"]["runs"], D["fair_comparison"]

    # 1 - corpus
    fig, ax = plt.subplots(figsize=(10, 6))
    tot = np.array([P["runs"][r]["frames"] for r in RUNS])
    sel = np.array([S["selection"][r]["selected"] for r in RUNS])
    y = np.arange(len(RUNS))
    ax.barh(y, tot, color="#dce5ee", label="Усі кадри")
    ax.barh(y, sel, color=BLUE, label="Відібрані для порівняння")
    ax.set_yticks(y, [RN[r] for r in RUNS])
    ax.invert_yaxis()
    ax.set_xlabel("Кількість кадрів")
    ax.legend(loc="lower right")
    for i, (v, t) in enumerate(zip(sel, tot)):
        ax.text(t + 15, i, f"{v} / {t}", va="center", fontsize=9)
    ax.set_xlim(0, max(tot) * 1.2)
    save(fig, 1, "Склад корпусу та відбір кадрів",
         "protocol_study.json; ablation_singles.json",
         "8 записів; кадри одного запису не є незалежними повтореннями.")

    # 2 - before and after
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for i, r in enumerate(B):
        ax = axs[0, i]
        steps = r["steps"]
        ax.plot(steps, r["before"]["profile"], "o-", color=RED,
                label="Історична конфігурація")
        ax.plot(steps, r["after"]["profile"], "o-", color=TEAL,
                label="Поточна конфігурація")
        ax.set(xlabel="Номер кроку", ylabel="Медіанна оцінка",
               title=RN[r["name"]], ylim=(-.05, 1.05))
        ax.legend(fontsize=8)
        ax = axs[1, i]
        ax.plot(steps, r["spot_radius"], "o-", color=GRAY)
        ax.set(xlabel="Номер кроку", ylabel="Радіус плями, пікселів")
        ax.invert_yaxis()
    save(fig, 2, "Як змінився зв'язок оцінки з опорним вимірюванням",
         "before_after.json",
         "2 записи; історичну конфігурацію відтворено сучасним кодом. "
         "Сіра крива - радіус плями (вісь перевернуто).")

    # 3 - singles
    fig, ax = plt.subplots(figsize=(11, 6))
    vals = agg("ablation_singles", "adj", ORDER)
    hbar(ax, [NAMES[x] for x in ORDER], vals,
         "Розрізнення сусідніх кроків (adj): 8 записів",
         color=[TEAL] + [BLUE] * 6, xlim=(0, 1.02))
    for i, v in enumerate(ORDER):
        per = [S["runs"][r][v]["adj"] for r in RUNS]
        ax.scatter(per, [i] * len(per), s=16, color="#1c2c3f", zorder=5, alpha=.65)
    save(fig, 3, "Ансамбль та одиничні метрики в однаковому конвеєрі",
         "ablation_singles.json",
         "Стовпчики - середнє за 8 записами; точки - окремі записи, "
         "не довірчі інтервали.")

    # 4 - per-recording heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    mat = np.array([[S["runs"][r][v]["adj"] for v in ORDER] for r in RUNS])
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=.4, vmax=1.0)
    ax.grid(False)
    ax.set_xticks(range(len(ORDER)), [NAMES[x] for x in ORDER],
                  rotation=25, ha="right")
    ax.set_yticks(range(len(RUNS)), [RN[r] for r in RUNS])
    for i, j in itertools.product(range(len(RUNS)), range(len(ORDER))):
        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if mat[i, j] > .78 else "#17222f")
    fig.colorbar(im, ax=ax, label="adj", shrink=.85)
    save(fig, 4, "Розрізнення кроків у кожному записі", "ablation_singles.json",
         "Ансамбль кращий за вейвлетну метрику у 6/8 записів; "
         "за найкращу одиничну в кожному записі - у 5/8.")

    # 5 - the two plateau statistics, side by side and never mixed
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, title in zip(
        axs, ["peak_plateau_steps", "peak_plateau"],
        ["Усі 8 записів (peak_plateau_steps)",
         "Лише 2 записи з еталоном (peak_plateau)"],
    ):
        hbar(ax, [NAMES[x] for x in ORDER], agg("ablation_singles", key, ORDER),
             "Середня ширина плато, кроків", color=[TEAL] + [BLUE] * 6,
             fmt=".2f", xlim=(0, 8.2))
        ax.set_title(title)
    save(fig, 5, "Ширина плато: дві різні вибірки", "ablation_singles.json",
         "Це окремі статистики. Вузьке плато означає однозначніший максимум "
         "сигналу, але не доводить правильність фокуса.")

    # 6 - saturation
    fig, ax = plt.subplots(figsize=(10, 6))
    hbar(ax, [NAMES[x] for x in ORDER],
         np.array(agg("ablation_singles", "sat", ORDER)) * 100,
         "Кадри біля власних крайніх значень сигналу, %: 8 записів",
         color=[TEAL] + [BLUE] * 6, fmt=".2f", xlim=(0, 29))
    save(fig, 6, "Насичення сигналу в поточному конвеєрі", "ablation_singles.json",
         "Поріг - 10^-6 від власного діапазону сигналу; це не насичення пікселів.")

    # 7 - raw signals against whole pipelines
    keys = ["six:shipped", "six:plain_mean", "six:fixed_weights", "six:with_kernel",
            "single:wavelet", "single:tenengrad", "single:brenner", "single:laplacian"]
    labels = ["Шість: адаптивні ваги", "Шість: рівні ваги", "Шість: апріорні ваги",
              "Шість: з ядром згоди", "Одна: вейвлетна", "Одна: Тененград",
              "Одна: Бреннер", "Одна: Лапласіан"]
    fig, ax = plt.subplots(figsize=(11, 7))
    hbar(ax, ["Сира вейвлетна"] + labels,
         [C["signals"]["wavelet"]["spearman_gt"]]
         + [C["pipelines"][k]["spearman_gt"] for k in keys],
         "Кореляція Спірмена з еталоном: 2 записи",
         color=[ORANGE, TEAL] + [BLUE] * 7, xlim=(0, 1.12))
    save(fig, 7, "Сирий сигнал і повні конвеєри: узгодження з еталоном",
         "fair_comparison.json",
         "Сира міра теж працює онлайн. Розрив 0.0222 - не шум еталона: "
         "зміна рецепта зсуває парні різниці лише на 0.002-0.005.")

    # 8 - factorial
    fk = ["---", "--F", "-A-", "-AF", "R--", "R-F", "RA-", "RAF"]
    fl = ["Без трьох компонентів", "Лише фільтр", "Лише ядро згоди",
          "Згода + фільтр", "Лише надійність", "Надійність + фільтр",
          "Надійність + згода", "Усі три"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    hbar(axs[0], fl, agg("ablation_factorial", "spearman_gt", fk),
         "Спірмен: 2 записи", xlim=(0, 1.12))
    hbar(axs[1], fl, agg("ablation_factorial", "adj", fk),
         "adj: 8 записів", xlim=(0, 1.02))
    save(fig, 8, "Повний факторний експеримент для трьох компонентів",
         "ablation_factorial.json",
         "Компоненти: модель надійності, ядро згоди, часовий фільтр. "
         "Два критерії дають різний порядок - див. рисунок 16.")

    # 9 - normalisation
    nk = ["rolling+linear", "rolling+logistic", "frozen+linear", "frozen+logistic",
          "short+rolling+linear", "short+frozen+logistic"]
    nl = ["Ковзна + лінійна", "Ковзна + логістична", "Фіксована + лінійна",
          "Фіксована + логістична", "Коротка ковзна + лінійна",
          "Коротка фіксована + логістична"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    hbar(axs[0], nl, agg("ablation_normalisation", "spearman_gt", nk),
         "Спірмен: 2 записи", xlim=(-.4, 1.12))
    hbar(axs[1], nl, np.array(agg("ablation_normalisation", "sat", nk)) * 100,
         "Насичення: 8 записів, %", fmt=".2f", xlim=(0, 20))
    save(fig, 9, "Вплив способу нормалізації", "ablation_normalisation.json",
         "Ця група використовує проміжну конфігурацію з увімкненим ядром згоди.")

    # 10 - each repair on its own
    fixk = ["baseline", "range", "horizon", "noise", "edges", "ready", "window",
            "freeze", "logistic", "freeze+logistic", "all"]
    fixl = ["Історична основа", "Поріг діапазону", "Довгий горизонт",
            "Квантиль шуму", "Поріг меж", "Умова готовності", "Довше вікно",
            "Фіксація шкали", "Логістична шкала", "Фіксація + логістична",
            "Усі виправлення"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 7))
    for ax, key, label, scale, xlim in zip(
        axs, ["spearman_gt", "sentinel", "noise_zero"],
        ["Спірмен: 2 записи", "Низька інформативність, %: 8 записів",
         "Нульова оцінка шуму, %: 8 записів"],
        [1, 100, 100], [(-.45, 1.12), (0, 48), (0, 116)],
    ):
        hbar(ax, fixl, np.array(agg("ablation_fixes", key, fixk)) * scale, label,
             fmt=".3f" if scale == 1 else ".1f", xlim=xlim)
    save(fig, 10, "Внески виправлень та діагностичні показники",
         "ablation_fixes.json",
         "Це окремі варіанти від базової конфігурації, а не послідовне "
         "додавання всіх змін.")

    # 11 - agreement kernel width
    ak = ["off", "scale1", "scale1.5", "scale2.5", "scale4", "scale8"]
    al = ["Вимкнено", "Масштаб 1", "Масштаб 1.5", "Масштаб 2.5", "Масштаб 4",
          "Масштаб 8"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    hbar(axs[0], al, agg("ablation_agreement", "spearman_gt", ak),
         "Спірмен: 2 записи", xlim=(0, 1.12))
    hbar(axs[1], al, agg("ablation_agreement", "adj", ak),
         "adj: 8 записів", xlim=(0, 1.02))
    save(fig, 11, "Перевірка ширини ядра згоди", "ablation_agreement.json",
         "Розширення ядра лише наближає результат до 'вимкнено'. Жоден запис "
         "не містить відмови, від якої ядро мало б захищати.")

    # 12 - leave one out
    mk = ["without_" + v for v in
          ["laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width"]]
    ml = ["Без Лапласіана", "Без Тененграда", "Без Бреннера", "Без вейвлетної",
          "Без Фур'є", "Без ширини меж"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    base = D["ablation_metrics"]["aggregate"]["six"]
    for ax, key, label in zip(
        axs, ["spearman_gt", "adj"],
        ["Зміна Спірмена: 2 записи", "Зміна adj: 8 записів"],
    ):
        vals = [D["ablation_metrics"]["aggregate"][v][key] - base[key] for v in mk]
        hbar(ax, ml, vals, label,
             color=[TEAL if v >= 0 else RED for v in vals], fmt="+.4f")
        ax.axvline(0, color=GRAY)
    save(fig, 12, "Почергове вилучення однієї метрики", "ablation_metrics.json",
         "База - шість метрик з увімкненим ядром згоди, не поточна стандартна "
         "конфігурація. Жодна міра не є незамінною.")

    # 13 - analysis width
    wk, ws = ["w240", "six", "w480", "w640"], [240, 320, 480, 640]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, key, label in zip(
        axs, ["spearman_gt", "adj", "ms"],
        ["Спірмен: 2 записи", "adj: 8 записів", "Середнє медіан часу, мс"],
    ):
        vals = agg("ablation_metrics", key, wk)
        ax.plot(ws, vals, "o-", color=BLUE)
        ax.set(xlabel="Ширина аналізу, пікселів", ylabel=label)
        ax.set_xticks(ws)
        for x, v in zip(ws, vals):
            ax.annotate(f"{v:.3f}" if key != "ms" else f"{v:.2f}", (x, v),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9)
        ax.margins(y=.3)
    save(fig, 13, "Роздільність аналізу: якість та обчислювальні витрати",
         "ablation_metrics.json",
         "Проміжна конфігурація з ядром згоди; ширший кадр не дає "
         "автоматичного поліпшення.")

    # 14 - reference recipe cross-correlation
    R = D["reference_sensitivity"]["runs"]
    rv = list(next(iter(R.values()))["variants"])
    rvl = [x.replace("r", "").replace("_w", "% / ").replace("_median", " / мед.")
            .replace("_p25", " / фон 25%") for x in rv]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (run, d) in zip(axs, R.items()):
        mat = np.array([[d["cross_correlation"][x][y] for y in rv] for x in rv])
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu", aspect="equal")
        ax.grid(False)
        ax.set_title(RN[run])
        ax.set_xticks(range(len(rv)), rvl, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(range(len(rv)), rvl, fontsize=7)
        for i, j in itertools.product(range(len(rv)), range(len(rv))):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.8,
                    color="white" if abs(mat[i, j]) > .65 else "#243345")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=.75,
                 label="Кореляція варіантів", location="bottom", pad=.06)
    save(fig, 14, "Чутливість опорного вимірювання до параметрів",
         "reference_sensitivity.json",
         "Частка суми інтенсивностей / півширина вікна, пікселів / оцінка фону. "
         "Вікна 120 пікселів анти-корельовані з меншими.")

    # 15 - where each recipe puts the best step
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (run, d) in zip(axs, R.items()):
        for i, v in enumerate(rv):
            q = d["variants"][v]
            a, b = q["best_step_first"], q["best_step_last"]
            ax.plot([a, b], [i, i], lw=6, color=BLUE)
            ax.scatter([(a + b) / 2], [i], color=BLUE, s=35)
        ax.set_yticks(np.arange(len(rv)), rvl, fontsize=9)
        ax.invert_yaxis()
        ax.set(xlabel="Крок мінімального радіуса плями", title=RN[run])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, 15, "Положення найкращого кроку за різними рецептами еталона",
         "reference_sensitivity.json",
         "Зміна рецепта може змінити опорний максимум; це не довірчий інтервал "
         "похибки методу.")

    # 16 - what adaptivity contributes, by criterion and by recording
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    deltas = [F["runs"][r]["R-F"]["adj"] - F["runs"][r]["--F"]["adj"] for r in RUNS]
    hbar(axs[0], [RN[r] for r in RUNS], deltas,
         "Адаптивні мінус апріорні ваги: Δadj, 8 записів",
         color=[TEAL if x >= 0 else RED for x in deltas], fmt="+.4f")
    axs[0].axvline(0, color=GRAY)
    axs[0].set_title(f"Виграш у {sum(d > 0 for d in deltas)}/8, "
                     f"середнє {np.mean(deltas):+.4f}", fontsize=11)
    ll, vv = [], []
    for ref, j in [("Основний", F), ("Альтернативний", A)]:
        for r in RUNS:
            if "spearman_gt" in j["runs"][r]["R-F"]:
                ll.append(f"{ref} / {RN[r].replace('Точкове джерело ', 'запис ')}")
                vv.append(j["runs"][r]["R-F"]["spearman_gt"]
                          - j["runs"][r]["plain_mean"]["spearman_gt"])
    hbar(axs[1], ll, vv, "Адаптивні мінус рівні ваги: ΔСпірмена, 2 записи",
         color=TEAL, fmt="+.4f", xlim=(0, .022))
    save(fig, 16, "Внесок адаптивності залежить від критерію й запису",
         "ablation_factorial.json; ablation_factorial_altref.json",
         "Знак протилежний за двома критеріями. Праворуч - 2 записи x 2 рецепти, "
         "а не 4 незалежні експерименти.")

    # 17 - runtime
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(ORDER))
    for off, key, label, col in [(-.23, "ms", "Медіана", BLUE),
                                 (0, "ms_p95", "95-й процентиль", ORANGE),
                                 (.23, "ms_p99", "99-й процентиль", RED)]:
        ax.barh(y + off, agg("ablation_singles", key, ORDER), height=.22,
                label=label, color=col)
    ax.set_yticks(y, [NAMES[x] for x in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Час обробки кадру, мс (Raspberry Pi 5)")
    ax.legend(loc="lower right")
    save(fig, 17, "Час обробки та його верхні процентилі", "ablation_singles.json",
         "Середнє відповідного показника за 8 записами; це не процентилі "
         "об'єднаної вибірки. Затримки захоплення й привода не включені.")

    # 18 - correct, wrong and tied pairs
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom, y = np.zeros(len(ORDER)), np.arange(len(ORDER))
    comparable = np.array(agg("ablation_singles", "pairs_comparable", ORDER))
    for key, label, col in [("pairs_correct", "Правильний порядок", TEAL),
                            ("pairs_tied", "Нічия", ORANGE),
                            ("pairs_wrong", "Помилковий порядок", RED)]:
        vals = np.array(agg("ablation_singles", key, ORDER)) / comparable * 100
        ax.barh(y, vals, left=bottom, label=label, color=col)
        bottom += vals
    ax.set_yticks(y, [NAMES[x] for x in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Частка порівнюваних пар кроків, %: 2 записи")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, fontsize=9)
    save(fig, 18, "Правильні, помилкові та нерозрізнені пари кроків",
         "ablation_singles.json",
         "Інверсії з урахуванням нічиїх = (помилки + 0.5 x нічиї) / усі пари.")

    # 19 - descriptive factorial contrasts
    terms = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    tl = ["Надійність", "Ядро згоди", "Фільтр", "Надійність x згода",
          "Надійність x фільтр", "Згода x фільтр", "Потрійна взаємодія"]
    effects: dict[str, list[float]] = {}
    for key in ("spearman_gt", "adj"):
        effects[key] = [
            float(sum(
                np.prod([1 if v[i] != "-" else -1 for i in term])
                * F["aggregate"][v][key] for v in fk
            ) / 4)
            for term in terms
        ]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key, label in zip(
        axs, ["spearman_gt", "adj"],
        ["Контраст Спірмена: 2 записи", "Контраст adj: 8 записів"],
    ):
        vals = effects[key]
        hbar(ax, tl, vals, label,
             color=[TEAL if x >= 0 else RED for x in vals], fmt="+.4f")
        ax.axvline(0, color=GRAY)
    save(fig, 19, "Описові факторні ефекти та взаємодії", "ablation_factorial.json",
         "Контраст = сума результатів зі знаками / 4 для плану 2^3. "
         "Це опис ефектів, без перевірки значущості.")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    derived = {
        "total_frames": int(sum(tot)),
        "selected_frames": int(sum(sel)),
        "best_single_wins": sum(
            S["runs"][r]["six_shipped"]["adj"]
            > max(S["runs"][r][v]["adj"] for v in ORDER[1:]) for r in RUNS
        ),
        "wavelet_wins": sum(
            S["runs"][r]["six_shipped"]["adj"] > S["runs"][r]["only_wavelet"]["adj"]
            for r in RUNS
        ),
        "adaptive_beats_prior_adj_wins": int(sum(d > 0 for d in deltas)),
        "adaptive_vs_prior_adj_mean": float(np.mean(deltas)),
        "adaptive_vs_mean_adj_mean": float(np.mean([
            F["runs"][r]["R-F"]["adj"] - F["runs"][r]["plain_mean"]["adj"]
            for r in RUNS
        ])),
        "factorial_contrasts": effects,
        "adaptive_vs_mean_rho": dict(zip(ll, vv)),
    }
    (out / "derived.json").write_text(
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # ASCII only: a console that cannot encode this line must not be able to
    # destroy a completed run, which is how an hour of compute was lost once.
    print(f"wrote {len(manifest)} figures to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "ua")
    args = parser.parse_args(argv)
    return run(args.reports, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
