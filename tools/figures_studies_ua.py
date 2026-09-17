"""Ukrainian figures for the Д07-Д20 studies, built from reports/study_*.json.

Same two rules as `tools/figures_ua.py`: nothing here opens a frame, and every
panel says how many recordings it covers.  Rank correlation against the spot
reference exists on two recordings and adjacent-step discrimination on eight,
and a table of this project was once published with the two mixed.

    python3 tools/figures_studies_ua.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "figure.dpi": 140, "savefig.dpi": 180,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .17, "axes.axisbelow": True, "svg.fonttype": "none",
    "svg.hashsalt": "adaptive-sharpness-studies-ua",
})
_SVG_METADATA = {"Date": None}

BLUE, TEAL, ORANGE, RED, GRAY = "#2563a6", "#07867d", "#d47722", "#b84751", "#778295"

SHORT = {
    "cond_20260916_110502": "Умови 1", "cond_20260916_110600": "Умови 2",
    "cond_20260916_110656": "Умови 3", "cond_20260916_110746": "Умови 4",
    "point_source_20260916_111545": "Точкове 1",
    "point_source_20260916_111724": "Точкове 2",
    "sweep_plain_20260916_110955": "Мала фактурність",
    "sweep_texture_20260916_110108": "Багата фактура",
}
STAGE_UA = {
    "1_raw": "1. Сира міра",
    "2_log": "2. Логарифм",
    "3_fixed_linear": "3. Фікс. лінійна",
    "4_fixed_logistic": "4. Фікс. логістична",
    "5a_streaming_single": "5a. Потокова, одна міра",
    "5b_streaming_ensemble": "5b. Потокова, шість",
    "6_temporal_filter": "6. + часовий фільтр",
}
METRIC_UA = {
    "laplacian": "Лапласіан", "tenengrad": "Тененград", "brenner": "Бреннер",
    "wavelet": "Вейвлетна", "fourier": "Фур'є", "edge_width": "Ширина меж",
}


class Builder:
    def __init__(self, reports: Path, out: Path) -> None:
        self.reports = reports
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.manifest: list[dict[str, Any]] = []

    def load(self, study: str) -> dict[str, Any] | None:
        path = self.reports / f"study_{study}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, fig, study: str, title: str, note: str) -> None:
        fig.suptitle(title, fontsize=15, fontweight="bold", x=.02, ha="left")
        fig.text(.02, .012, note, fontsize=9, color="#485568")
        fig.tight_layout(rect=(0, .07, 1, .92), pad=1.5)
        for ext in ("png", "svg"):
            fig.savefig(
                self.out / f"{study}.{ext}", bbox_inches="tight",
                facecolor="white",
                metadata=_SVG_METADATA if ext == "svg" else None,
            )
        plt.close(fig)
        self.manifest.append({"study": study, "title": title, "note": note})


def hbar(ax, labels, values, xlabel, color=BLUE, fmt="{:.3f}"):
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=color, height=.66)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    for rect, v in zip(bars, values):
        if np.isfinite(v):
            ax.annotate(fmt.format(v),
                        (v, rect.get_y() + rect.get_height() / 2),
                        xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9)
    ax.margins(x=.2)
    ax.grid(axis="y", visible=False)


def figure_d07(b: Builder) -> None:
    d = b.load("d07")
    if not d:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    blocks = d["blocks"]
    widths = []
    for block in blocks:
        vals = [
            r["blocks"]["adaptive-prior"][str(block)]["width"]
            for r in d["runs"].values()
            if "adaptive-prior" in r["blocks"]
            and str(block) in r["blocks"]["adaptive-prior"]
        ]
        widths.append(np.mean(vals) if vals else np.nan)
    axs[0].plot(blocks, widths, "o-", color=BLUE)
    axs[0].set(xlabel="Довжина блока, кадрів",
               ylabel="Ширина 95% інтервалу (Δadj)",
               title="Наївний ресемплінг кадрів занижує інтервал")
    axs[0].set_xscale("log")
    for x, y in zip(blocks, widths):
        axs[0].annotate(f"{y:.3f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)

    names, means, lows, highs = [], [], [], []
    for pair, v in d["corpus"].items():
        names.append(
            pair.replace("adaptive-prior", "адаптивні − апріорні")
                .replace("adaptive-equal", "адаптивні − рівні")
        )
        means.append(v["mean"])
        lows.append(v["low"])
        highs.append(v["high"])
    y = np.arange(len(names))
    axs[1].errorbar(
        means, y,
        xerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="o", color=TEAL, capsize=6, markersize=8, linewidth=2,
    )
    axs[1].axvline(0, color=RED, linestyle="--", linewidth=1.4)
    axs[1].set_yticks(y, names)
    axs[1].set(xlabel="Різниця adj (8 записів)",
               title="Обидва інтервали перетинають нуль")
    axs[1].grid(axis="y", visible=False)
    axs[1].margins(y=.5)
    for i, (m, lo, hi) in enumerate(zip(means, lows, highs)):
        axs[1].annotate(f"{m:+.4f}  [{lo:+.3f}, {hi:+.3f}]", (m, i),
                        xytext=(0, 18), textcoords="offset points",
                        ha="center", fontsize=9)
    b.save(fig, "d07",
           "Д07. Невизначеність парних різниць",
           "Блоковий bootstrap у межах запису; кадри автокорельовані "
           "(лаг декореляції 3–74 кадри), тому ресемплінг окремих кадрів "
           "дає інтервал утричі вужчий за чесний.")


def figure_d09(b: Builder) -> None:
    d = b.load("d09")
    if not d:
        return
    a = d["aggregate"]
    stages = sorted(a)
    labels = [STAGE_UA.get(s, s) for s in stages]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    hbar(axs[0], labels, [a[s]["adj"] for s in stages],
         "adj: 8 записів", color=[TEAL] * len(stages))
    hbar(axs[1], labels, [a[s].get("spearman_gt", np.nan) for s in stages],
         "Спірмен: 2 записи", color=[BLUE] * len(stages), fmt="{:.4f}")
    hbar(axs[2], labels, [a[s]["saturated"] * 100 for s in stages],
         "Насичення, %: 8 записів", color=[ORANGE] * len(stages), fmt="{:.2f}")
    b.save(fig, "d09",
           "Д09. Нормалізація поетапно",
           "Стадії 1–4 — вейвлетна міра сама; 5a — вона ж через увесь "
           "потоковий конвеєр; 5b — шість метрик. Без 5a падіння на кроці 5 "
           "не можна віднести ні до потокової шкали, ні до набору метрик.")


def figure_d10(b: Builder) -> None:
    d = b.load("d10")
    if not d:
        return
    a = d["aggregate"]
    order = ["start_10pct", "start_25pct", "start_50pct", "reversed"]
    order = [o for o in order if o in a]
    labels = {"start_10pct": "Старт із 10%", "start_25pct": "Старт із 25%",
              "start_50pct": "Старт із 50%", "reversed": "Зворотний порядок"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    names = [labels[o] for o in order]
    hbar(axs[0], names, [a[o]["max_abs_difference"] for o in order],
         "Макс. |Δ оцінки| того самого кадру", color=[RED] * len(order))
    hbar(axs[1], names, [a[o]["rank_agreement"] for o in order],
         "Узгодження рангів із повним проходом", color=[BLUE] * len(order),
         fmt="{:.4f}")
    hbar(axs[2], names, [a[o]["adj_change"] for o in order],
         "Зміна adj", color=[TEAL if a[o]["adj_change"] >= 0 else RED
                             for o in order], fmt="{:+.4f}")
    axs[2].axvline(0, color=GRAY)
    b.save(fig, "d10",
           "Д10. Залежність оцінки від передісторії потоку",
           "Порівнюються ті самі кадри, тому різниця — це історія й нічого "
           "більше. Середнє за 8 записами.")


def figure_d11(b: Builder) -> None:
    d = b.load("d11")
    if not d:
        return
    a = d["aggregate"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    keys = [k for k in a if k != "off"]
    adj = np.array([a[k]["adj"] for k in keys])
    spear = np.array([a[k].get("spearman_gt", np.nan) for k in keys])
    axs[0].scatter(adj, spear, s=46, color=BLUE, label="Варіанти сітки", zorder=3)
    axs[0].scatter([a["off"]["adj"]], [a["off"].get("spearman_gt", np.nan)],
                   s=150, color=RED, marker="X", zorder=4, label="Фіксацію вимкнено")
    shipped = "min240_pat120_stab0.05"
    if shipped in a:
        axs[0].scatter([a[shipped]["adj"]], [a[shipped]["spearman_gt"]],
                       s=150, color=TEAL, marker="*", zorder=5,
                       label="Поточні параметри")
    axs[0].set(xlabel="adj: 8 записів", ylabel="Спірмен: 2 записи",
               title="Механізм важить набагато більше за його пороги")
    axs[0].legend(fontsize=9, loc="lower left")

    best = sorted(keys, key=lambda k: -a[k]["adj"])[:8]
    hbar(axs[1], best + ["ВИМКНЕНО"],
         [a[k].get("spearman_gt", np.nan) for k in best]
         + [a["off"].get("spearman_gt", np.nan)],
         "Спірмен: 2 записи",
         color=[BLUE] * len(best) + [RED], fmt="{:.4f}")
    axs[1].set_title("Вісім найкращих наборів і вимкнена фіксація", fontsize=11)
    b.save(fig, "d11",
           "Д11. Параметри автофіксації шкали",
           "Вимкнена фіксація дає Спірмена 0.50; будь-який працездатний набір "
           "порогів — близько 0.96. Сталі підбиралися на цих самих записах, "
           "але результат від них майже не залежить.")


def figure_d12(b: Builder) -> None:
    d = b.load("d12")
    if not d:
        return
    runs = d["runs"]
    names = list(runs)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    metrics = list(next(iter(runs.values()))["metrics"])
    bottom = np.zeros(len(names))
    y = np.arange(len(names))
    palette = [BLUE, TEAL, ORANGE, RED, "#7d5ba6", GRAY]
    for position, metric in enumerate(metrics):
        share = np.array([
            runs[n]["metrics"][metric]["weight_mean"] for n in names
        ])
        axs[0].barh(y, share, left=bottom, height=.68,
                    color=palette[position % len(palette)],
                    label=METRIC_UA.get(metric, metric))
        bottom += share
    axs[0].set_yticks(y, [SHORT.get(n, n) for n in names])
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Середня вага метрики")
    axs[0].legend(fontsize=8, ncol=3, loc="upper center",
                  bbox_to_anchor=(.5, -.1))
    axs[0].grid(axis="y", visible=False)
    axs[0].set_title("Ваги майже не рухаються між записами", fontsize=11)

    pinned = np.array([
        runs[n]["drivers_at_minimum_fraction"]["edge_density"] * 100 for n in names
    ])
    hbar(axs[1], [SHORT.get(n, n) for n in names], pinned,
         "Кадрів із щільністю меж на мінімумі, %",
         color=[RED if p > 70 else BLUE for p in pinned], fmt="{:.0f}")
    axs[1].set_title("Вхід моделі надійності насичений", fontsize=11)
    b.save(fig, "d12",
           "Д12. Що насправді роблять адаптивні ваги",
           "Ваги зчитано з кожного кадру. Два записи з найвищим насиченням "
           "меж — ті самі, де адаптивність програє найбільше, але звʼязок по "
           "восьми записах слабкий (ρ=−0.41).")


def figure_d13(b: Builder) -> None:
    d = b.load("d13")
    if not d:
        return
    runs = d["runs"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for name, run in runs.items():
        if not run["has_reference"]:
            continue
        pts = [(p["coverage"], p.get("spearman_gt")) for p in run["curve"]
               if p.get("spearman_gt") is not None]
        if pts:
            xs, ys = zip(*sorted(pts))
            axs[0].plot(xs, ys, "o-", label=SHORT.get(name, name))
    axs[0].set(xlabel="Частка збережених кадрів",
               ylabel="Спірмен з еталоном",
               title="Відбір за впевненістю псує узгодження")
    axs[0].legend(fontsize=9)
    axs[0].invert_xaxis()

    within = [(n, r["within_step"]) for n, r in runs.items()
              if isinstance(r.get("within_step"), dict)
              and "high_confidence" in r["within_step"]]
    if within:
        labels = [SHORT.get(n, n) for n, _ in within]
        y = np.arange(len(within))
        for offset, key, colour, label in (
            (-.25, "low_confidence", RED, "Низька впевненість"),
            (0.0, "mid_confidence", ORANGE, "Середня"),
            (.25, "high_confidence", TEAL, "Висока"),
        ):
            axs[1].barh(y + offset,
                        [w.get(key, np.nan) for _, w in within],
                        height=.24, color=colour, label=label)
        axs[1].set_yticks(y, labels)
        axs[1].invert_yaxis()
        axs[1].set_xlabel("Середнє |оцінка − медіана свого кроку|")
        axs[1].legend(fontsize=9)
        axs[1].grid(axis="y", visible=False)
        axs[1].set_title("У межах кроку впевненість не відокремлює точні кадри",
                         fontsize=11)
    b.save(fig, "d13",
           "Д13. Чи несе показник впевненості інформацію",
           "Ліва панель має два артефакти: відкидання кадрів прибирає кроки й "
           "звужує діапазон еталона. Права їх не має — у межах одного кроку "
           "справжній фокус сталий.")


def figure_d14(b: Builder) -> None:
    d = b.load("d14")
    if not d:
        return
    a = d["aggregate"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    keys = [k for k in a if k != "off"]
    settle = np.array([a[k].get("settling_frames", np.nan) for k in keys])
    adj = np.array([a[k]["adj"] for k in keys])
    coupled = np.array(["coupled" in k for k in keys])
    axs[0].scatter(settle[coupled], adj[coupled], s=60, color=TEAL,
                   label="Звʼязано з впевненістю", zorder=3)
    axs[0].scatter(settle[~coupled], adj[~coupled], s=60, color=BLUE,
                   label="Без звʼязку", zorder=3)
    if "off" in a:
        axs[0].scatter([0], [a["off"]["adj"]], s=140, color=RED, marker="X",
                       zorder=4, label="Фільтр вимкнено")
    for k, x, y in zip(keys, settle, adj):
        if np.isfinite(x):
            axs[0].annotate(k.replace("alpha", "α=").replace("_coupled", "")
                            .replace("_plain", ""), (x, y), xytext=(5, 4),
                            textcoords="offset points", fontsize=7.5)
    axs[0].set(xlabel="Кадрів до усталення після зміни кроку",
               ylabel="adj: 8 записів",
               title="Згладжування купується затримкою")
    axs[0].legend(fontsize=9)

    order = sorted(a, key=lambda k: -a[k]["adj"])[:8]
    hbar(axs[1], [k.replace("alpha", "α=") for k in order],
         [a[k]["adj"] for k in order], "adj: 8 записів",
         color=[TEAL] * len(order))
    b.save(fig, "d14",
           "Д14. Часовий фільтр: згладжування проти затримки",
           "Спірмен майже не залежить від α (0.956–0.965) — фільтр змінює "
           "розрізнення й затримку, а не узгодження з еталоном.")


def figure_d15(b: Builder) -> None:
    d = b.load("d15")
    if not d:
        return
    a = d["aggregate"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sizes = np.array([a[k]["size"] for k in a])
    adj = np.array([a[k]["adj"] for k in a])
    ms = np.array([a[k]["ms"] for k in a])
    axs[0].scatter(sizes + np.random.default_rng(1).normal(0, .06, sizes.size),
                   adj, s=30, color=BLUE, alpha=.65)
    for size in sorted(set(sizes)):
        best = adj[sizes == size].max()
        axs[0].scatter([size], [best], s=120, color=TEAL, marker="*", zorder=4)
        axs[0].annotate(f"{best:.4f}", (size, best), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9)
    axs[0].set(xlabel="Кількість метрик у наборі", ylabel="adj: 8 записів",
               title="Найкращий набір — пʼять метрик, не шість")
    axs[0].set_xticks(sorted(set(sizes)))

    axs[1].scatter(ms, adj, s=30, color=BLUE, alpha=.6)
    for key in sorted(a, key=lambda k: -a[k]["adj"])[:4]:
        axs[1].annotate(key, (a[key]["ms"], a[key]["adj"]), xytext=(6, 3),
                        textcoords="offset points", fontsize=8)
    if "6:six" in a:
        axs[1].scatter([a["6:six"]["ms"]], [a["6:six"]["adj"]], s=140,
                       color=RED, marker="X", zorder=5, label="Поточні шість")
        axs[1].legend(fontsize=9)
    axs[1].set(xlabel="Медіана часу, мс", ylabel="adj: 8 записів",
               title="Ціна й якість кожного з 63 наборів")
    b.save(fig, "d15",
           "Д15. Усі 63 підмножини шести метрик",
           "Вибір найкращого набору зроблено на тих самих восьми записах, "
           "тож перевага +0.013 лежить у межах інтервалу Д07 і не є "
           "встановленою.")


def figure_d16(b: Builder) -> None:
    d = b.load("d16")
    if not d:
        return
    a = d["aggregate"]
    widths = [160, 240, 320, 480, 640]
    regions = [("full", "Весь кадр"), ("centre50", "Центр 50%"),
               ("centre25", "Центр 25%")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, field, label in zip(
        axs, ["adj", "spearman_gt", "ms"],
        ["adj: 8 записів", "Спірмен: 2 записи", "Медіана часу, мс"],
    ):
        for key, name in regions:
            values = [a.get(f"w{w}_{key}", {}).get(field, np.nan) for w in widths]
            ax.plot(widths, values, "o-", label=name)
        ax.set(xlabel="Ширина аналізу, пікселів", ylabel=label)
        ax.set_xticks(widths)
        ax.legend(fontsize=8)
    b.save(fig, "d16",
           "Д16. Роздільність аналізу та область інтересу",
           "Поточні 320 пікселів по всьому кадру — найкращі за обома "
           "критеріями; ширші кадри різко втрачають Спірмена.")


def figure_d17(b: Builder) -> None:
    d = b.load("d17")
    if not d:
        return
    a = d["aggregate"]
    families = {"noise": "Шум", "brightness": "Яскравість", "clip": "Пересвіт",
                "shift": "Зсув", "blur": "Розмиття"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for family, label in families.items():
        keys = sorted(
            [k for k in a if k.startswith(family + "_")],
            key=lambda k: float(k.rsplit("_", 1)[1]),
        )
        if not keys:
            continue
        amounts = [float(k.rsplit("_", 1)[1]) for k in keys]
        axs[0].plot(amounts, [a[k]["adj"] for k in keys], "o-", label=label)
        axs[1].plot(amounts,
                    [a[k].get("rank_agreement_with_clean", np.nan) for k in keys],
                    "o-", label=label)
    base = a.get("none", {}).get("adj", np.nan)
    axs[0].axhline(base, color=GRAY, linestyle="--", label="Без деградації")
    axs[0].set(xlabel="Величина деградації", ylabel="adj: 8 записів",
               title="Розмиття має знижувати оцінку — перевірка розумності")
    axs[0].legend(fontsize=9)
    axs[1].set(xlabel="Величина деградації",
               ylabel="Узгодження рангів із чистим прогоном",
               title="Наскільки змінюється сам порядок кадрів")
    axs[1].legend(fontsize=9)
    b.save(fig, "d17",
           "Д17. Штучні деградації реальних кадрів",
           "Це модельні перетворення з фіксованим seed, а не нові фізичні "
           "вимірювання: додати гаусів шум до вже знешумленого 8-бітного "
           "перегляду — не те саме, що підняти ISO.")


def figure_d18(b: Builder) -> None:
    d = b.load("d18")
    if not d:
        return
    a, held = d["aggregate"], d.get("held_out", {})
    factors = d["factors"]
    keys = [f"kappa{f:g}" for f in factors]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(factors, [a[k]["adj"] for k in keys], "o-", color=TEAL,
                label="adj: 8 записів")
    twin = axs[0].twinx()
    twin.plot(factors, [a[k].get("spearman_gt", np.nan) for k in keys], "s--",
              color=BLUE, label="Спірмен: 2 записи")
    twin.grid(False)
    axs[0].set(xlabel="Множник матриці κ (1 = поточна)", ylabel="adj")
    twin.set_ylabel("Спірмен")
    axs[0].axvline(1.0, color=GRAY, linestyle=":")
    axs[0].legend(fontsize=9, loc="lower left")
    twin.legend(fontsize=9, loc="lower right")
    axs[0].set_title("Результат майже не залежить від коефіцієнтів", fontsize=11)

    if held:
        labels = ["Підібрано на\nзаписах умов", "Поточне κ=1"]
        fit = [held.get("fit_adj", np.nan), held.get("shipped_fit_adj", np.nan)]
        out = [held.get("held_out_adj", np.nan),
               held.get("shipped_held_out_adj", np.nan)]
        y = np.arange(2)
        axs[1].barh(y - .2, fit, height=.36, color=ORANGE, label="Набір підбору")
        axs[1].barh(y + .2, out, height=.36, color=TEAL, label="Відкладені записи")
        axs[1].set_yticks(y, labels)
        axs[1].invert_yaxis()
        axs[1].set_xlabel("adj")
        axs[1].legend(fontsize=9)
        axs[1].grid(axis="y", visible=False)
        axs[1].set_xlim(0.78, 0.87)
        axs[1].set_title("Підбір на частині корпусу не переноситься", fontsize=11)
    b.save(fig, "d18",
           "Д18. Чутливість коефіцієнтів надійності",
           f"Підбір дав {held.get('chosen_on_fit','?')}, що на відкладених "
           f"записах гірше за поточне κ=1. Найкраще на відкладених — "
           f"{held.get('best_on_held_out','?')}.")


def figure_d19(b: Builder) -> None:
    d = b.load("d19")
    if not d:
        return
    a = d["aggregate"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for strategy, colour, label in (("hill_climb", BLUE, "Сходження вгору"),
                                    ("ternary", TEAL, "Тернарний пошук")):
        keys = sorted([k for k in a if k.startswith(strategy)],
                      key=lambda k: int(k.rsplit("budget", 1)[1]))
        budgets = [int(k.rsplit("budget", 1)[1]) for k in keys]
        axs[0].plot(budgets, [a[k]["found_signal_peak_fraction"] for k in keys],
                    "o-", color=colour, label=label)
        within = [a[k].get("within_one_step_of_reference", np.nan) for k in keys]
        axs[1].plot(budgets, within, "o-", color=colour, label=label)
    axs[0].set(xlabel="Бюджет оцінювань", ylabel="Частка запусків",
               title="Знайдено максимум самого сигналу")
    axs[0].legend(fontsize=9)
    axs[1].set(xlabel="Бюджет оцінювань", ylabel="Частка запусків",
               title="У межах одного кроку від еталона (2 записи)")
    axs[1].legend(fontsize=9)
    b.save(fig, "d19",
           "Д19. Пошук по вже записаному проходу",
           "Це програвання збереженого профілю: приводу, люфту й часу "
           "усталення тут немає. Показує форму сигналу, а не швидкість "
           "автофокуса.")


def figure_d20(b: Builder) -> None:
    d = b.load("d20")
    if not d:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ag = d["criteria_agreement"]
    keys = list(ag)
    matrix = np.array([[ag[a][b_] for b_ in keys] for a in keys])
    im = axs[0].imshow(matrix, vmin=-1, vmax=1, cmap="RdBu")
    axs[0].grid(False)
    axs[0].set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    axs[0].set_yticks(range(len(keys)), keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            axs[0].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(matrix[i, j]) > .6 else "#22303f")
    fig.colorbar(im, ax=axs[0], shrink=.8, label="Узгодження рейтингів")
    axs[0].set_title(f"{d['variants_ranked']} конфігурацій", fontsize=11)

    tol = d["plateau_tolerance"]
    xs = [float(t) for t in tol]
    ys = [tol[t]["mean_plateau_steps"] for t in tol]
    axs[1].plot(xs, ys, "o-", color=ORANGE)
    for x, y in zip(xs, ys):
        axs[1].annotate(f"{y:.2f}", (x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
    axs[1].set(xlabel="Допуск визначення плато (частка діапазону)",
               ylabel="Середня ширина плато, кроків",
               title="Плато втричі ширшає разом із допуском")
    b.save(fig, "d20",
           "Д20. Чи узгоджуються критерії оцінювання",
           "adj і Спірмен узгоджуються на 0.85 — розбіжність у §5.3 була "
           "властивістю одного порівняння, а не критеріїв. Похибка максимуму "
           "не ранжує взагалі.")


def figure_d08(b: Builder) -> None:
    d = b.load("d08")
    if not d or "aggregate" not in d:
        return
    a = d["aggregate"]
    if "first_commit" not in a or "current" not in a:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    hbar(axs[0], ["Перший коміт", "Поточна версія"],
         [a["first_commit"]["adj"], a["current"]["adj"]],
         "adj: 8 записів", color=[RED, TEAL])
    hbar(axs[1], ["Перший коміт", "Поточна версія"],
         [a["first_commit"].get("spearman_gt", np.nan),
          a["current"].get("spearman_gt", np.nan)],
         "Спірмен: 2 записи", color=[RED, TEAL], fmt="{:.4f}")
    b.save(fig, "d08",
           "Д08. Перший коміт, виконаний насправді",
           "Не реконструкція параметрів у сучасному коді, а сам оригінальний "
           "код на тих самих кадрах.")


BUILDERS = (
    figure_d07, figure_d08, figure_d09, figure_d10, figure_d11, figure_d12,
    figure_d13, figure_d14, figure_d15, figure_d16, figure_d17, figure_d18,
    figure_d19, figure_d20,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "figures" / "ua" / "studies"
    )
    args = parser.parse_args(argv)

    builder = Builder(args.reports, args.out)
    for build in BUILDERS:
        try:
            build(builder)
        except Exception as error:  # a missing study must not stop the rest
            print(f"skipped {build.__name__}: {type(error).__name__}: {error}")
    (args.out / "manifest.json").write_text(
        json.dumps(builder.manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(builder.manifest)} figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
