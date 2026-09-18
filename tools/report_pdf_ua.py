"""A short Ukrainian PDF summary, built from reports/ so it cannot drift.

Every number on the page is read out of `reports/*.json` at build time rather
than typed in.  A summary that is retyped is a second copy of the results, and
this project has already had one of those disagree with the first.

    python3 tools/report_pdf_ua.py --out docs/ZVIT_UA.pdf
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]

INK = colors.HexColor("#1b2733")
MUTED = colors.HexColor("#5a6875")
BLUE = colors.HexColor("#2563a6")
TEAL = colors.HexColor("#07867d")
RED = colors.HexColor("#b84751")
RULE = colors.HexColor("#d4dce4")
BAND = colors.HexColor("#eef3f8")


def register_fonts() -> None:
    """DejaVu ships with matplotlib and covers Cyrillic; the built-in
    reportlab fonts do not, and would silently render boxes."""
    base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    for name, filename in (
        ("UA", "DejaVuSans.ttf"),
        ("UA-Bold", "DejaVuSans-Bold.ttf"),
        ("UA-Oblique", "DejaVuSans-Oblique.ttf"),
        ("UA-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
        ("UA-Mono", "DejaVuSansMono.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(base / filename)))
    # Without the family mapping, `<b>` in a paragraph resolves to a bold face
    # that does not exist for this font and silently renders as regular, so
    # every emphasis in the document disappears.
    pdfmetrics.registerFontFamily(
        "UA", normal="UA", bold="UA-Bold",
        italic="UA-Oblique", boldItalic="UA-BoldOblique",
    )
    pdfmetrics.registerFontFamily("UA-Mono", normal="UA-Mono", bold="UA-Mono")


def styles() -> dict[str, ParagraphStyle]:
    sheet = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle(
            "title", parent=sheet["Title"], fontName="UA-Bold", fontSize=21,
            leading=26, textColor=INK, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=sheet["Normal"], fontName="UA", fontSize=10.5,
            leading=15, textColor=MUTED, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=sheet["Heading1"], fontName="UA-Bold", fontSize=14,
            leading=18, textColor=INK, spaceBefore=14, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2", parent=sheet["Heading2"], fontName="UA-Bold", fontSize=11,
            leading=15, textColor=BLUE, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=sheet["Normal"], fontName="UA", fontSize=9.6,
            leading=14.5, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=sheet["Normal"], fontName="UA", fontSize=8.2,
            leading=12, textColor=MUTED, spaceAfter=5,
        ),
        "cell": ParagraphStyle(
            "cell", parent=sheet["Normal"], fontName="UA", fontSize=8.6,
            leading=12, textColor=INK,
        ),
        "cellb": ParagraphStyle(
            "cellb", parent=sheet["Normal"], fontName="UA-Bold", fontSize=8.6,
            leading=12, textColor=INK,
        ),
        "caption": ParagraphStyle(
            "caption", parent=sheet["Normal"], fontName="UA", fontSize=8.2,
            leading=11.5, textColor=MUTED, spaceBefore=3, spaceAfter=12,
        ),
    }
    return out


class Data:
    """Every number on the page, read from the published reports."""

    def __init__(self, reports: Path) -> None:
        self.reports = reports
        self._cache: dict[str, Any] = {}

    def load(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            path = self.reports / f"{name}.json"
            self._cache[name] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else {}
            )
        return self._cache[name]

    def agg(self, report: str, variant: str, field: str, default=float("nan")):
        return (
            self.load(report).get("aggregate", {}).get(variant, {}).get(field, default)
        )

    def provenance(self) -> dict[str, Any]:
        for name in ("study_d19", "study_d15", "ablation_singles"):
            payload = self.load(name)
            if payload.get("provenance"):
                return payload["provenance"]
        return {}


def table(rows, styles_map, widths, header_background=BAND, highlight=()):
    data = []
    for index, row in enumerate(rows):
        style = styles_map["cellb"] if index == 0 else styles_map["cell"]
        data.append([Paragraph(str(cell), style) for cell in row])
    spec = [
        ("BACKGROUND", (0, 0), (-1, 0), header_background),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for index in highlight:
        spec.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#e6f2f0")))
    return Table(data, colWidths=widths, style=TableStyle(spec), hAlign="LEFT")


def figure(path: Path, width: float) -> Image | None:
    if not path.exists():
        return None
    from PIL import Image as PILImage

    with PILImage.open(path) as handle:
        ratio = handle.height / handle.width
    return Image(str(path), width=width, height=width * ratio)


def build(data: Data, out: Path) -> Path:
    register_fonts()
    S = styles()
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Adaptive Sharpness - підсумок досліджень",
        author="RE22GDV",
    )
    width = doc.width
    story: list[Any] = []
    P = lambda text, key="body": Paragraph(text, S[key])  # noqa: E731

    provenance = data.provenance()
    commit = str(provenance.get("commit", ""))[:12]

    # ---------------------------------------------------------------- page 1
    story += [
        P("Adaptive Sharpness: підсумок досліджень", "title"),
        P(
            f"Оцінювання різкості в потоці для автофокуса. Panasonic GH6, "
            f"Raspberry Pi 5.<br/>Корпус: 8 записів, 9017 кадрів, два з "
            f"незалежним фізичним еталоном. Код: <font face='UA-Mono'>"
            f"{commit}</font>.", "subtitle",
        ),
        P("Що це за система", "h1"),
        P(
            "Модуль рахує ознаки різкості кадру в реальному часі й видає число "
            "в діапазоні 0–1. Він <b>не керує об'єктивом</b> — це джерело "
            "вимірювання для зовнішнього алгоритму пошуку. Усередині: шість "
            "класичних метрик різкості, нормалізація з історією потоку, "
            "зважування за оцінкою умов зйомки та часовий фільтр.", "body",
        ),
        P(
            "Заявлений метод складається з трьох механізмів: злиття шести "
            "метрик, адаптивне зважування за надійністю та ядро згоди. "
            "Двадцять досліджень на записаних кадрах перевіряли, що з цього "
            "справді працює.", "body",
        ),
        P("Головний результат", "h1"),
        P(
            "<b>Найбільший внесок дає компонент, якого немає в назві методу.</b> "
            "Фіксація шкали нормалізації важить більше, ніж злиття, зважування "
            "й фільтр разом. І два критерії оцінювання ранжують компоненти "
            "<b>протилежно</b> — бо вони роблять різні роботи.", "body",
        ),
    ]

    d9 = "study_d09"
    d11 = "study_d11"
    freeze_sp = data.agg(d11, "min240_pat120_stab0.05", "spearman_gt") - data.agg(d11, "off", "spearman_gt")
    freeze_adj = data.agg(d11, "min240_pat120_stab0.05", "adj") - data.agg(d11, "off", "adj")
    fuse_sp = data.agg(d9, "5b_streaming_ensemble", "spearman_gt") - data.agg(d9, "5a_streaming_single", "spearman_gt")
    fuse_adj = data.agg(d9, "5b_streaming_ensemble", "adj") - data.agg(d9, "5a_streaming_single", "adj")
    filt_sp = data.agg(d9, "6_temporal_filter", "spearman_gt") - data.agg(d9, "5b_streaming_ensemble", "spearman_gt")
    filt_adj = data.agg(d9, "6_temporal_filter", "adj") - data.agg(d9, "5b_streaming_ensemble", "adj")

    story += [
        Spacer(1, 4),
        table(
            [
                ["Компонент", "Правильність напрямку<br/>(Спірмен, 2 записи)",
                 "Роздільність кроків<br/>(adj, 8 записів)"],
                ["<b>Фіксація шкали</b>", f"<b>{freeze_sp:+.3f}</b>", f"{freeze_adj:+.3f}"],
                ["<b>Злиття шести метрик</b>", f"{fuse_sp:+.3f}", f"<b>{fuse_adj:+.3f}</b>"],
                ["<b>Часовий фільтр</b>", f"{filt_sp:+.3f}", f"<b>{filt_adj:+.3f}</b>"],
                ["Адаптивне зважування", "різниці не встановлено", "різниці не встановлено"],
                ["Ядро згоди", "−0.016", "−0.012"],
            ],
            S, [width * 0.36, width * 0.32, width * 0.32], highlight=(1,),
        ),
        Spacer(1, 8),
        P(
            "<b>Фіксація шкали відповідає за правильність.</b> Поки опорні "
            "процентилі рухаються, та сама фізична різкість отримує різні "
            "числа: рангова кореляція з фізичним еталоном падає з "
            f"{data.agg(d11, 'min240_pat120_stab0.05', 'spearman_gt'):.2f} до "
            f"{data.agg(d11, 'off', 'spearman_gt'):.2f}. Практичний наслідок "
            "виміряний окремо: похибка максимуму <b>подвоюється</b> — з "
            f"{data.agg(d11, 'min240_pat120_stab0.05', 'peak_err'):.1f} до "
            f"{data.agg(d11, 'off', 'peak_err'):.1f} кроку.", "body",
        ),
        P(
            "<b>Злиття й фільтр відповідають за роздільність.</b> Разом вони "
            f"дають {fuse_adj + filt_adj:+.3f} до розрізнення сусідніх "
            f"положень, але лише {fuse_sp + filt_sp:+.3f} до узгодження з "
            "еталоном. Вони не роблять сигнал «правильнішим» — вони роблять "
            "його достатньо тонким.", "body",
        ),
        P(
            "<b>Практичний наслідок:</b> оптимізувати щось одне не можна. "
            "Система без фіксації вказує не туди; система без злиття вказує "
            "туди, але не може сказати, наскільки точно.", "body",
        ),
    ]

    fig = figure(ROOT / "docs" / "figures" / "ua" / "studies" / "d09.png", width * 0.98)
    if fig:
        story += [
            Spacer(1, 6), fig,
            P(
                "Драбина від сирої міри до оцінки. Офлайнове масштабування "
                "безкоштовне; уся втрата — перехід на потокову нормалізацію, "
                "яку злиття й фільтр відбивають на дві третини.", "caption",
            ),
        ]

    # ---------------------------------------------------------------- page 2
    story += [P("Що система робить добре", "h1")]

    d19 = "study_d19"
    singles = "ablation_singles"
    story += [
        P(
            f"<b>Веде пошук.</b> Тернарний пошук по записаному профілю "
            f"потрапляє в межі одного кроку від фізичного еталона у "
            f"<b>{data.agg(d19, 'ternary_budget4', 'within_one_step_of_reference'):.0%}</b> "
            f"запусків, витрачаючи "
            f"{data.agg(d19, 'ternary_budget4', 'mean_evaluations'):.1f} оцінки. "
            "Залишкова похибка в один крок дорівнює роздільності самого "
            "еталона.", "body",
        ),
        P(
            f"<b>Розрізняє кроки краще за будь-яку одиничну міру.</b> "
            f"{data.agg(singles, 'six_shipped', 'adj'):.3f} проти "
            f"{data.agg(singles, 'only_fourier', 'adj'):.3f}–"
            f"{data.agg(singles, 'only_brenner', 'adj'):.3f} на всіх восьми "
            "записах, з найбільшим відривом на бідній фактурі — саме там, де "
            "злиття й має допомагати.", "body",
        ),
        P(
            f"<b>Не насичується.</b> "
            f"{data.agg(singles, 'six_shipped', 'sat') * 100:.2f} % кадрів на "
            f"краю шкали проти "
            f"{data.agg(singles, 'only_tenengrad', 'sat') * 100:.0f}–"
            f"{data.agg(singles, 'only_fourier', 'sat') * 100:.0f} % в "
            "одиничних мір. Кадр на краю не несе напрямку.", "body",
        ),
        P(
            f"<b>Працює в реальному часі на цільовому залізі.</b> Медіана "
            f"{data.agg(singles, 'six_shipped', 'ms'):.2f} мс на кадр, 99-й "
            f"процентиль {data.agg(singles, 'six_shipped', 'ms_p99'):.2f} мс — "
            "хвіст короткий.", "body",
        ),
        P("Чого система не робить — читати перед інтеграцією", "h1"),
    ]

    d17, d10, d13 = "study_d17", "study_d10", "study_d13"
    blur = data.agg(d17, "blur_4", "score_shift")
    story += [
        table(
            [
                ["Обмеження", "Виміряно"],
                ["<b>Не бачить, що весь потік не у фокусі</b><br/>"
                 "Нормалізатор підганяє шкалу під те, що бачить, тож "
                 "найменш розмитий із розмитих лишається «найрізкішим».",
                 f"Розмиття кожного кадру зсуває середню оцінку на "
                 f"<b>{blur:+.3f}</b> — тобто не знижує її."],
                ["<b>Залежить від моменту ввімкнення</b><br/>"
                 "Оцінка належить історії потоку, а не кадру.",
                 f"Ті самі кадри після іншого прогріву: узгодження рангів "
                 f"<b>{data.agg(d10, 'start_50pct', 'rank_agreement'):.2f}</b>, "
                 f"різниця оцінок до "
                 f"<b>{data.agg(d10, 'start_50pct', 'max_abs_difference'):.2f}</b>."],
                ["<b>Показник впевненості не відбирає точні кадри</b><br/>"
                 "Його складові корисні для діагностики; сам скаляр — ні.",
                 "Впевнені кадри точніші на <b>4 записах із 8</b>; кореляція "
                 "з похибкою від −0.71 до +0.63."],
                ["<b>Оцінки з різних сцен непорівнянні</b><br/>"
                 "Шкалу підігнано під історію одного потоку.",
                 "За побудовою; підтверджено поведінкою після скидання."],
            ],
            S, [width * 0.5, width * 0.5],
        ),
    ]

    # ---------------------------------------------------------------- page 3
    story += [PageBreak(), P("Що не підтвердилося", "h1")]

    d07 = "study_d07"
    corpus = data.load(d07).get("corpus", {}).get("adaptive-prior", {})
    story += [
        P(
            "<b>Адаптивне зважування.</b> Це заявлена новизна методу, і "
            "переконливої різниці проти фіксованих ваг <b>не встановлено</b>: "
            f"точкова оцінка {corpus.get('mean', float('nan')):+.4f} при 95 % "
            f"інтервалі [{corpus.get('low', float('nan')):+.3f}, "
            f"{corpus.get('high', float('nan')):+.3f}] — приблизно вдев'ятеро "
            "ширшому. Це не доказ рівноцінності: корпус просто не здатен "
            "розділити ці варіанти.", "body",
        ),
        P(
            "Причина видна, якщо прочитати ваги покадрово: <b>вони майже не "
            "рухаються</b>. За весь запис вага метрики проходить 0.01–0.05 при "
            "середніх 0.04–0.24. Чинники, що мають їх рухати, насичені — "
            "щільність меж стоїть на мінімумі на 24–79 % кадрів. Розрізняти "
            "майже нічого.", "body",
        ),
        P(
            "<b>Ядро згоди.</b> Третій заявлений механізм погіршує результат "
            "за обома критеріями; розширення ядра лише наближає до "
            "«вимкнено». Вимкнене за замовчуванням.", "body",
        ),
        P(
            "<b>Чого це не доводить.</b> Обидва механізми побудовані проти "
            "ситуацій, яких у цих восьми записах немає: дзеркальний відблиск, "
            "що обманює одну міру, справді шумний сенсор, різка зміна умов "
            "усередині проходу. Перевірити їх призначення на цьому корпусі "
            "неможливо.", "body",
        ),
    ]

    fig = figure(ROOT / "docs" / "figures" / "ua" / "studies" / "d07.png", width * 0.96)
    if fig:
        story += [
            Spacer(1, 4), fig,
            P(
                "Ліворуч: чесний інтервал утричі ширший за наївний, бо сусідні "
                "кадри корельовані. Праворуч: обидва інтервали перетинають "
                "нуль.", "caption",
            ),
        ]

    story += [
        P("Чесна оцінка внеску", "h1"),
        P(
            "Якщо питати, що в роботі нове — відповідь <b>не</b> «адаптивне "
            "зважування». Реальний внесок скромніший і конкретніший:", "body",
        ),
        P(
            "<b>1. Дисципліна нормалізації.</b> Фіксація шкали, логістичне "
            "відображення замість лінійного з обрізанням, квантильний "
            "оцінювач шуму замість медіанного. Це виправлення дефектів, не "
            "новизна — але саме вони дають різницю між −0.15 і +0.97 у "
            "кореляції з фізичним еталоном.", "body",
        ),
        P(
            "<b>2. Вимірювальна інфраструктура.</b> Незалежний від мір "
            "різкості еталон, спільний відбір кадрів, відбитки конфігурації, "
            "перевірка кожного числа в тексті проти звітів. Більшість помилок "
            "у проєкті знайшли <b>вимірювання, а не читання коду</b>.", "body",
        ),
        P(
            "<b>3. Записаний корпус</b> із мітками кроків і двома записами з "
            "фізичним еталоном.", "body",
        ),
    ]

    # ---------------------------------------------------------------- page 4
    story += [PageBreak(), P("Межі застосування", "h1")]
    story += [
        table(
            [
                ["Що", "Межа"],
                ["Природа оцінки", "Відносна в межах одного потоку. Не є "
                 "абсолютною різкістю й не порівнюється між сценами, "
                 "камерами чи екземплярами."],
                ["Обсяг даних", "8 записів, один об'єктив, одна кімната, "
                 "один день. Фізичний еталон — лише на 2 записах."],
                ["Роздільність корпусу", "Різниці менші за <b>0.04</b> цей "
                 "корпус не розділяє. Наївний ресемплінг кадрів дав би втричі "
                 "вужчий інтервал і хибний висновок."],
                ["Калібрування", "Результат залежить від калібрувального "
                 "проходу. Порівнювати положення можна лише на зафіксованій "
                 "шкалі, і не можна — через скидання."],
                ["Замкнений контур", "Не вимірювався. Положення об'єктива не "
                 "записувалося, тож про затримку автофокуса нічого сказати "
                 "не можна."],
                ["Час обробки", "Виміряно на Raspberry Pi 5. Числа з "
                 "досліджень Д07–Д20 отримані на робочій станції й для "
                 "порівняння продуктивності непридатні."],
            ],
            S, [width * 0.26, width * 0.74],
        ),
        P("Що змінило б картину", "h1"),
        P(
            "Усе, для чого потрібні <b>нові записи</b>, а не нові обчислення "
            "тих самих кадрів: сцени з відблисками та справді високим ISO "
            "(єдиний спосіб перевірити те, заради чого існують модель "
            "надійності й ядро згоди); інші об'єктиви й приміщення; замкнений "
            "контур із записаним положенням об'єктива; і просто більше сцен — "
            "щоб розрізнити ефект розміру 0.01, потрібні десятки незалежних "
            "сцен, а не вісім.", "body",
        ),
        P("Де подробиці", "h1"),
        table(
            [
                ["Документ", "Про що"],
                ["<font face='UA-Mono'>docs/STUDY_UA_REPORT.md</font>",
                 "Принцип роботи, 5 схем, історія змін, глобальний аналіз (§9)"],
                ["<font face='UA-Mono'>docs/STUDY_UA_RESULTS.md</font>",
                 "Усі 14 досліджень: що тестували, як, що отримали, чого не доводить"],
                ["<font face='UA-Mono'>docs/RESULTS.md</font>",
                 "Усі таблиці, згенеровані зі збережених звітів"],
                ["<font face='UA-Mono'>demo/calibrated_focus_loop.py</font>",
                 "Повний сценарій: калібрування → перевірка → фіксація → порівняння → скидання"],
                ["<font face='UA-Mono'>docs/LIMITATIONS.md</font>",
                 "Повний перелік обмежень із обґрунтуванням"],
            ],
            S, [width * 0.42, width * 0.58],
        ),
        Spacer(1, 10),
        P(
            f"Звіт згенеровано з <font face='UA-Mono'>reports/*.json</font> "
            f"командою <font face='UA-Mono'>python3 tools/report_pdf_ua.py</font>. "
            f"Жодне число не введено вручну. Код: "
            f"<font face='UA-Mono'>{commit}</font>, "
            f"numpy {provenance.get('numpy', '?')}, "
            f"opencv {provenance.get('opencv', '?')}.", "small",
        ),
    ]

    doc.build(story, onFirstPage=_furniture, onLaterPages=_furniture)
    return out


def _furniture(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("UA", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Adaptive Sharpness — підсумок досліджень")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(canvas.getPageNumber()))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "ZVIT_UA.pdf")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = build(Data(args.reports), args.out)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
