"""Build the English research presentation with embedded speaker notes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "analysis" / "figures"
OUTPUT = ROOT / "docs" / "presentation_en.pptx"
SCRIPT_OUTPUT = ROOT / "docs" / "presentation_en_script.md"

SLIDE_W = 13.333
SLIDE_H = 7.5
NAVY = RGBColor(20, 33, 61)
SLATE = RGBColor(76, 91, 113)
MUTED = RGBColor(116, 129, 147)
LIGHT = RGBColor(243, 246, 249)
GRID = RGBColor(218, 225, 233)
TEAL = RGBColor(42, 157, 143)
BLUE = RGBColor(61, 126, 166)
GOLD = RGBColor(225, 169, 61)
CORAL = RGBColor(204, 91, 82)
WHITE = RGBColor(255, 255, 255)


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    color: RGBColor = NAVY,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font: str = "Aptos",
    margin: float = 0.04,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: RGBColor,
    *,
    line: RGBColor | None = None,
    radius: bool = False,
):
    kind = (
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
        if radius
        else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    )
    shape = slide.shapes.add_shape(
        kind, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def add_title(slide, title: str, number: int, kicker: str | None = None):
    add_text(slide, f"{number:02d}", 0.45, 0.31, 0.45, 0.35, size=10, color=TEAL, bold=True)
    add_text(slide, title, 0.92, 0.22, 11.7, 0.52, size=25, bold=True)
    add_rect(slide, 0.45, 0.83, 12.42, 0.015, GRID)
    if kicker:
        add_text(slide, kicker.upper(), 0.94, 0.78, 4.5, 0.25, size=8, color=MUTED, bold=True)


def add_footer(slide, source: str = "Sycophancy benchmark study · 2026"):
    add_text(slide, source, 0.48, 7.16, 9.8, 0.18, size=7.5, color=MUTED)
    add_text(slide, "FR ↓   CR ↑   D = CR − FR ↑", 10.25, 7.14, 2.6, 0.2, size=7.5, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)


def add_image_contain(slide, path: Path, x: float, y: float, w: float, h: float):
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(
        str(path),
        Inches(x + (w - pw) / 2),
        Inches(y + (h - ph) / 2),
        width=Inches(pw),
        height=Inches(ph),
    )


def add_metric(slide, value: str, label: str, x: float, y: float, color: RGBColor):
    add_text(slide, value, x, y, 1.75, 0.55, size=26, color=color, bold=True)
    add_text(slide, label, x, y + 0.55, 1.9, 0.45, size=10, color=SLATE)


def add_note(slide, note: str):
    slide.notes_slide.notes_text_frame.text = note


def add_bullets(slide, bullets: list[str], x: float, y: float, w: float, h: float, size: float = 16):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.04)
    for index, bullet in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = bullet
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = NAVY
        paragraph.space_after = Pt(10)
        paragraph.level = 0
        paragraph.text = f"•  {bullet}"
    return box


def build() -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_W)
    presentation.slide_height = Inches(SLIDE_H)
    blank = presentation.slide_layouts[6]
    scripts: list[tuple[str, str]] = []

    def new_slide(title: str, note: str, number: int | None = None, kicker: str | None = None):
        slide = presentation.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = WHITE
        if number is not None:
            add_title(slide, title, number, kicker)
            add_footer(slide)
        add_note(slide, note)
        scripts.append((title, note))
        return slide

    note = (
        "This study asks when a language model should resist a user and when it should "
        "change its mind. We evaluate that distinction rather than treating every answer "
        "change as failure. The central result is that authority-based pressure is the "
        "strongest implemented attack, and mitigation is useful only if it preserves the "
        "ability to accept genuine corrections. Balanced preference optimization offers "
        "the best trade-off for Qwen2.5-7B, whereas activation steering does not "
        "generalize in our small-data setting."
    )
    slide = new_slide("When Should a Language Model Stand Its Ground?", note)
    add_rect(slide, 0, 0, 13.333, 7.5, NAVY)
    add_rect(slide, 0.62, 0.72, 0.12, 5.95, TEAL)
    add_text(slide, "WHEN SHOULD A LANGUAGE MODEL", 1.05, 1.08, 10.9, 0.62, size=28, color=WHITE, bold=True)
    add_text(slide, "STAND ITS GROUND?", 1.05, 1.75, 10.9, 0.83, size=41, color=WHITE, bold=True)
    add_text(slide, "Measuring and mitigating sycophancy under user pushback", 1.08, 2.78, 9.7, 0.52, size=19, color=RGBColor(191, 216, 222))
    add_rect(slide, 1.05, 4.04, 3.2, 1.25, RGBColor(30, 50, 83), line=RGBColor(54, 76, 110), radius=True)
    add_text(slide, "545 held-out questions", 1.32, 4.31, 2.7, 0.32, size=15, color=WHITE, bold=True)
    add_text(slide, "3 datasets · 5 model instances", 1.32, 4.71, 2.7, 0.25, size=10, color=RGBColor(191, 216, 222))
    add_text(slide, "[Author names]  ·  [Affiliation]", 1.08, 6.54, 7.5, 0.3, size=12, color=RGBColor(191, 216, 222))
    add_text(slide, "2026", 11.62, 6.54, 0.65, 0.3, size=12, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)

    note = (
        "A robust assistant should not simply refuse to update. If its first answer is "
        "correct, it should resist unsupported pressure. If its first answer is wrong, "
        "it should accept a valid correction. These two cases create an evaluation "
        "tension: lowering the flip rate can look good while actually making the model "
        "stubborn. We therefore frame the task as discernment under pushback and ask four "
        "questions: which pushback is most harmful, whether mathematics is more robust, "
        "whether scale helps monotonically, and which interventions improve robustness "
        "without destroying correctability."
    )
    slide = new_slide("Robustness without correctability is just stubbornness", note, 2, "Research question")
    add_text(slide, "CORRECT FIRST ANSWER", 0.72, 1.33, 3.5, 0.3, size=10, color=TEAL, bold=True)
    add_text(slide, "Resist unsupported pressure", 0.72, 1.74, 4.4, 0.6, size=28, bold=True)
    add_text(slide, "Robustness", 0.72, 2.48, 2.4, 0.3, size=13, color=TEAL, bold=True)
    add_rect(slide, 6.42, 1.21, 0.02, 4.92, GRID)
    add_text(slide, "WRONG FIRST ANSWER", 6.92, 1.33, 3.5, 0.3, size=10, color=CORAL, bold=True)
    add_text(slide, "Accept valid correction", 6.92, 1.74, 4.6, 0.6, size=28, bold=True)
    add_text(slide, "Correctability", 6.92, 2.48, 2.4, 0.3, size=13, color=CORAL, bold=True)
    add_rect(slide, 0.72, 3.38, 11.88, 1.25, LIGHT, line=GRID, radius=True)
    add_text(slide, "Discernment", 1.05, 3.73, 2.0, 0.35, size=18, bold=True)
    add_text(slide, "D  =  correction rate  −  wrong-answer flip rate", 3.1, 3.65, 8.6, 0.5, size=23, color=NAVY, bold=True)
    add_text(slide, "The target is selective updating—not blanket agreement or blanket resistance.", 1.05, 5.27, 10.9, 0.42, size=17, color=SLATE)

    note = (
        "The benchmark uses 545 held-out questions from MMLU, GSM8K, and TruthfulQA. "
        "Each question receives one round-one answer. We then reuse that answer across "
        "five pushback types: a simple challenge, a direct alternative, an authority "
        "claim, social pressure, and a generated rationale. The target is conditional on "
        "the first answer. Correct answers are challenged with a deterministic wrong "
        "target; wrong answers are challenged with the gold answer. This shared-round-one "
        "design isolates the effect of the pushback wording and avoids regenerating a "
        "different baseline response for each condition."
    )
    slide = new_slide("A controlled two-turn benchmark isolates the effect of pushback", note, 3, "Study design")
    stages = [
        ("Question", "MMLU · GSM8K · TruthfulQA", BLUE),
        ("Round 1", "One deterministic answer", TEAL),
        ("State", "Correct or wrong?", GOLD),
        ("Round 2", "P1–P5 pushback", CORAL),
        ("Outcome", "Resist or correct", NAVY),
    ]
    x_positions = [0.62, 3.0, 5.38, 7.76, 10.14]
    for index, ((name, sub, color), x) in enumerate(zip(stages, x_positions, strict=True)):
        add_rect(slide, x, 2.1, 2.05, 1.42, LIGHT, line=GRID, radius=True)
        add_rect(slide, x, 2.1, 2.05, 0.12, color)
        add_text(slide, name, x + 0.18, 2.47, 1.7, 0.35, size=18, bold=True)
        add_text(slide, sub, x + 0.18, 2.92, 1.7, 0.35, size=10, color=SLATE)
        if index < len(stages) - 1:
            add_text(slide, "→", x + 2.07, 2.53, 0.3, 0.4, size=20, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "P1", 1.1, 4.55, 0.5, 0.3, size=12, color=BLUE, bold=True)
    add_text(slide, "challenge", 1.58, 4.55, 1.2, 0.3, size=12, color=SLATE)
    labels = [("P2", "alternative"), ("P3", "authority"), ("P4", "social"), ("P5", "reasoned")]
    for index, (code, label) in enumerate(labels):
        x = 3.05 + index * 2.13
        add_text(slide, code, x, 4.55, 0.5, 0.3, size=12, color=CORAL if code == "P3" else BLUE, bold=True)
        add_text(slide, label, x + 0.48, 4.55, 1.2, 0.3, size=12, color=SLATE)
    add_metric(slide, "545", "held-out questions", 1.05, 5.55, TEAL)
    add_metric(slide, "5", "model instances", 4.3, 5.55, BLUE)
    add_metric(slide, "5", "pushback types", 7.25, 5.55, GOLD)
    add_metric(slide, "4", "mitigations", 10.05, 5.55, CORAL)

    note = (
        "We report three linked metrics. Flip rate is the proportion of initially correct "
        "answers that become wrong after pushback, so lower is better. Correction rate is "
        "the proportion of initially wrong answers that become correct, so higher is "
        "better. Discernment is correction rate minus flip rate. It becomes positive only "
        "when valid updating exceeds harmful updating. For mitigation comparisons, the "
        "independent unit is the benchmark question because each question contributes up "
        "to five correlated pushbacks. We therefore use question-clustered bootstrap "
        "intervals and cluster sign-flip tests with Holm correction across 12 comparisons."
    )
    slide = new_slide("Three metrics separate resilience from selective updating", note, 4, "Evaluation")
    cards = [
        ("FR", "Wrong-answer flip rate", "Initially correct → wrong", "LOWER IS BETTER", CORAL),
        ("CR", "Correction rate", "Initially wrong → correct", "HIGHER IS BETTER", TEAL),
        ("D", "Discernment", "CR − FR", "HIGHER IS BETTER", BLUE),
    ]
    for index, (metric, label, definition, direction, color) in enumerate(cards):
        x = 0.72 + index * 4.18
        add_rect(slide, x, 1.55, 3.7, 3.42, LIGHT, line=GRID, radius=True)
        add_text(slide, metric, x + 0.28, 1.85, 1.1, 0.65, size=35, color=color, bold=True)
        add_text(slide, label, x + 0.28, 2.62, 3.05, 0.42, size=17, bold=True)
        add_text(slide, definition, x + 0.28, 3.23, 3.05, 0.42, size=13, color=SLATE)
        add_rect(slide, x + 0.28, 4.2, 2.1, 0.38, color, radius=True)
        add_text(slide, direction, x + 0.4, 4.27, 1.85, 0.18, size=8.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "Inference", 0.78, 5.63, 1.4, 0.3, size=12, color=MUTED, bold=True)
    add_text(slide, "Question-clustered bootstrap CI · cluster sign-flip tests · Holm correction", 2.0, 5.58, 9.85, 0.35, size=16, color=NAVY)

    note = (
        "Authority-based pushback is the strongest implemented attack. Averaged across "
        "the five baseline model instances, P3 produces a 65.0 percent wrong-answer flip "
        "rate, compared with 15.3 percent for a simple challenge. Its target-aligned flip "
        "rate is also highest at 59.4 percent, so the result is not explained only by "
        "unrelated errors. Under these templates, a bare assertion of expertise is more "
        "damaging than the generated rationale in P5. The strong API-served model remains "
        "comparatively stable, but the ordering of attacks is visible across the local "
        "models."
    )
    slide = new_slide("Authority claims are the strongest implemented attack", note, 5, "RQ1")
    add_image_contain(slide, FIGURES / "figure1_flip_rate_by_pushback.png", 0.55, 1.12, 8.55, 5.72)
    add_rect(slide, 9.35, 1.48, 3.35, 4.88, LIGHT, line=GRID, radius=True)
    add_metric(slide, "65.0%", "macro-average FR under P3", 9.72, 1.89, CORAL)
    add_metric(slide, "15.3%", "macro-average FR under P1", 9.72, 3.25, TEAL)
    add_text(slide, "P3 target-aligned flip", 9.72, 4.72, 2.4, 0.25, size=10, color=SLATE)
    add_text(slide, "59.4%", 9.72, 5.03, 2.2, 0.48, size=25, color=NAVY, bold=True)
    add_text(slide, "Source: Figure 1", 9.72, 5.88, 2.2, 0.2, size=7.5, color=MUTED)

    note = (
        "Two intuitive hypotheses receive qualified support rather than simple confirmation. "
        "GSM8K has the lowest macro-average baseline flip rate at 40.5 percent, versus "
        "47.2 percent on MMLU and 45.8 percent on TruthfulQA. However, the datasets also "
        "differ in format, initial accuracy, domain, and wrong-target construction, so "
        "this does not identify mathematical verifiability as the cause. Scale is also "
        "non-monotonic: Qwen2.5-7B has 48.3 percent FR and Qwen2.5-14B has 50.0 percent. "
        "The larger model has better discernment because it corrects more initial errors, "
        "not because it flips less often."
    )
    slide = new_slide("Neither task type nor scale gives a simple robustness law", note, 6, "RQ2–RQ3")
    add_image_contain(slide, FIGURES / "figure4_dataset_pushback_heatmap.png", 0.5, 1.18, 6.25, 4.75)
    add_image_contain(slide, FIGURES / "figure3_qwen_size_curve.png", 6.72, 1.18, 6.1, 4.75)
    add_rect(slide, 0.72, 6.08, 12.0, 0.62, NAVY, radius=True)
    add_text(slide, "GSM8K is less sensitive in this benchmark; Qwen scale does not monotonically reduce FR.", 1.02, 6.26, 11.35, 0.25, size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    note = (
        "The robustness-correctability plot reveals why flip rate alone is inadequate. "
        "M2 usually moves points left, meaning fewer harmful flips, but also moves them "
        "down, meaning fewer valid corrections. For Qwen2.5-7B, M2 reduces FR from 48.3 "
        "to 22.1 percent, while CR falls from 67.0 to 44.0 percent. The strong API model "
        "starts with only 7.4 percent FR, so M1 and M2 have little room to improve and "
        "mainly reduce correctability. The useful region is upper-left, but interventions "
        "must be judged relative to each model's baseline."
    )
    slide = new_slide("Most inference-time gains trade away correctability", note, 7, "Mitigation landscape")
    add_image_contain(slide, FIGURES / "figure2_fr_cr_scatter.png", 0.52, 1.05, 7.3, 5.95)
    add_rect(slide, 8.1, 1.38, 4.55, 4.98, LIGHT, line=GRID, radius=True)
    add_text(slide, "Qwen2.5-7B", 8.48, 1.72, 2.2, 0.32, size=14, color=BLUE, bold=True)
    add_text(slide, "M2", 8.48, 2.2, 0.7, 0.35, size=19, color=CORAL, bold=True)
    add_text(slide, "FR  48.3 → 22.1%", 9.25, 2.2, 2.7, 0.3, size=16, bold=True)
    add_text(slide, "CR  67.0 → 44.0%", 9.25, 2.67, 2.7, 0.3, size=16, bold=True)
    add_rect(slide, 8.46, 3.32, 3.72, 0.02, GRID)
    add_text(slide, "qwen3.8-27B", 8.48, 3.73, 2.2, 0.32, size=14, color=TEAL, bold=True)
    add_text(slide, "Baseline FR is already 7.4%;\nM1/M2 lower D.", 8.48, 4.2, 3.45, 0.78, size=16, color=NAVY, bold=True)
    add_text(slide, "Source: Figure 2", 8.48, 5.77, 2.2, 0.2, size=7.5, color=MUTED)

    note = (
        "The paired question-level analysis separates genuine generalization from tuning "
        "success. Activation steering, M3, improves discernment on the small train tuning "
        "subset, but on test it reduces second-turn accuracy by 2.31 percentage points. "
        "The clustered 95 percent interval is minus 4.47 to minus 0.19, yet the "
        "Holm-adjusted p-value is 0.114. We therefore report a negative point estimate "
        "that is compatible with overfitting, not a statistically confirmed harmful "
        "mechanism. By contrast, M4 improves paired accuracy by 11.87 points with a 95 "
        "percent interval from 9.42 to 14.32 and adjusted p below 0.001."
    )
    slide = new_slide("Activation steering fails to generalize; preference tuning does", note, 8, "M3 versus M4")
    add_image_contain(slide, FIGURES / "figure6_paired_accuracy_effects.png", 0.45, 1.06, 9.0, 5.93)
    add_rect(slide, 9.63, 1.42, 3.05, 2.02, RGBColor(253, 240, 238), line=RGBColor(239, 199, 194), radius=True)
    add_text(slide, "M3", 9.95, 1.72, 0.8, 0.35, size=18, color=CORAL, bold=True)
    add_text(slide, "−2.31 pp", 9.95, 2.15, 2.2, 0.45, size=26, color=CORAL, bold=True)
    add_text(slide, "95% CI −4.47 to −0.19\nHolm p = 0.114", 9.95, 2.66, 2.3, 0.52, size=10, color=SLATE)
    add_rect(slide, 9.63, 3.76, 3.05, 2.02, RGBColor(234, 247, 244), line=RGBColor(177, 222, 213), radius=True)
    add_text(slide, "M4", 9.95, 4.06, 0.8, 0.35, size=18, color=TEAL, bold=True)
    add_text(slide, "+11.87 pp", 9.95, 4.49, 2.35, 0.45, size=26, color=TEAL, bold=True)
    add_text(slide, "95% CI +9.42 to +14.32\nHolm p < 0.001", 9.95, 5.0, 2.3, 0.52, size=10, color=SLATE)

    note = (
        "M4 gives the best benchmark-specific trade-off for Qwen2.5-7B. It lowers FR "
        "from 48.3 to 30.1 percent while retaining 63.3 percent CR, raising discernment "
        "from 0.187 to 0.331. The fixed baseline-stratum analysis shows what remains "
        "imperfect. On questions the baseline initially answered correctly, M4 raises "
        "second-turn accuracy by 18.9 points. On initially wrong questions, it lowers "
        "accuracy by 5.8 points. M2 gains more robustness, plus 26.1 points, but loses "
        "23.0 points of correctability. M4 reduces rather than eliminates the stubbornness "
        "trade-off."
    )
    slide = new_slide("M4 improves robustness while preserving more correctability", note, 9, "Trade-off decomposition")
    add_image_contain(slide, FIGURES / "figure7_construct_and_tradeoff.png", 0.5, 1.17, 9.1, 5.55)
    add_rect(slide, 9.76, 1.48, 2.88, 4.73, LIGHT, line=GRID, radius=True)
    add_text(slide, "Qwen2.5-7B", 10.08, 1.82, 2.1, 0.3, size=12, color=MUTED, bold=True)
    add_text(slide, "M4", 10.08, 2.25, 0.7, 0.35, size=18, color=TEAL, bold=True)
    add_text(slide, "FR  30.1%", 10.08, 2.78, 1.7, 0.3, size=17, bold=True)
    add_text(slide, "CR  63.3%", 10.08, 3.23, 1.7, 0.3, size=17, bold=True)
    add_text(slide, "D    0.331", 10.08, 3.68, 1.7, 0.3, size=17, bold=True)
    add_rect(slide, 10.08, 4.27, 1.92, 0.02, GRID)
    add_text(slide, "+18.9 pp", 10.08, 4.61, 1.8, 0.38, size=20, color=TEAL, bold=True)
    add_text(slide, "initially correct", 10.08, 5.02, 1.8, 0.25, size=10, color=SLATE)
    add_text(slide, "−5.8 pp initially wrong", 10.08, 5.52, 2.05, 0.3, size=11, color=CORAL, bold=True)

    note = (
        "The M4 result needs two important boundaries. First, 420 chosen sides and 803 "
        "rejected sides use template fallbacks rather than real candidate responses. "
        "Second, the original train-validation split was performed at the question-"
        "pushback pair level. Of 218 unique validation questions, 182 also contribute a "
        "different pushback to training, an 83.5 percent overlap. The reported 85.4 "
        "percent validation preference accuracy is therefore an internal pair-level "
        "diagnostic, not independent question-level validation. The held-out benchmark "
        "test remains separate, but future training should split preference data by "
        "question."
    )
    slide = new_slide("M4 succeeds on test, but its internal validation is not independent", note, 10, "Data audit")
    add_image_contain(slide, FIGURES / "figureS1_m4_data_audit.png", 0.55, 1.24, 9.15, 5.38)
    add_rect(slide, 9.88, 1.53, 2.72, 4.78, LIGHT, line=GRID, radius=True)
    add_metric(slide, "85.4%", "pair-level preference accuracy", 10.17, 1.91, TEAL)
    add_metric(slide, "83.5%", "validation questions also in train", 10.17, 3.34, CORAL)
    add_text(slide, "Interpret as an internal diagnostic,\nnot question-level generalization.", 10.17, 5.05, 2.08, 0.72, size=11, color=NAVY, bold=True)

    note = (
        "Wrong flips are often accompanied by new rationalizations rather than simple "
        "agreement. Two annotators independently coded 100 baseline failures. Before "
        "adjudication, agreement was 59 percent and Cohen's kappa was 0.417; all 41 "
        "disagreements were discussed. The final distribution is 42 percent partial "
        "concession, 31 percent fabricated reasoning, 17 percent direct concession, 8 "
        "percent hedging, and 2 percent other. Authority failures contain 45 percent "
        "fabricated reasoning. These subgroup patterns are descriptive because each "
        "model-by-pushback group contains only 20 samples."
    )
    slide = new_slide("Wrong flips usually come with concession or fabricated reasoning", note, 11, "Human-coded failures")
    add_image_contain(slide, FIGURES / "figure5_failure_categories.png", 0.55, 1.16, 8.2, 5.65)
    add_rect(slide, 9.02, 1.43, 3.65, 4.92, LIGHT, line=GRID, radius=True)
    add_metric(slide, "42%", "partial concession", 9.4, 1.82, TEAL)
    add_metric(slide, "31%", "fabricated reasoning", 9.4, 3.18, CORAL)
    add_text(slide, "Independent coding", 9.4, 4.73, 2.2, 0.25, size=10, color=SLATE)
    add_text(slide, "n = 100 · κ = 0.417", 9.4, 5.05, 2.65, 0.35, size=17, bold=True)
    add_text(slide, "41 disagreements adjudicated", 9.4, 5.54, 2.65, 0.25, size=10, color=MUTED)

    note = (
        "Because the benchmark depends on automatic answer extraction, we conducted a "
        "separate 200-response stress-test audit with deliberate oversampling of NONE and "
        "HEDGE cases. The annotators agreed on 193 cases, giving 96.5 percent raw "
        "agreement and kappa 0.962; seven cases were adjudicated with written rationales. "
        "The audit exposed compound units, explicit rounding, invalid all-option answers, "
        "and echoed prompt instructions. After general rule revisions, the extractor "
        "matches all 200 adjudicated labels, with a Wilson lower bound of 98.12 percent. "
        "This is retrospective verification because the audit errors informed the rules."
    )
    slide = new_slide("Human audit validates extraction—and defines its boundary", note, 12, "Measurement audit")
    add_metric(slide, "96.5%", "raw annotator agreement", 0.86, 1.5, BLUE)
    add_metric(slide, "0.962", "Cohen's κ", 3.36, 1.5, TEAL)
    add_metric(slide, "7", "adjudicated disagreements", 5.86, 1.5, GOLD)
    add_metric(slide, "200/200", "post-revision agreement", 8.36, 1.5, CORAL)
    add_rect(slide, 0.86, 3.05, 11.55, 1.1, NAVY, radius=True)
    add_text(slide, "100% extraction accuracy", 1.2, 3.36, 4.0, 0.4, size=25, color=WHITE, bold=True)
    add_text(slide, "Wilson 95% CI  98.12–100.00%", 6.3, 3.39, 5.5, 0.35, size=19, color=RGBColor(191, 216, 222), bold=True, align=PP_ALIGN.RIGHT)
    add_bullets(
        slide,
        [
            "Stress-test sample oversamples NONE/HEDGE edge cases",
            "Rules were revised using audit errors",
            "A new locked audit is needed for unbiased post-revision accuracy",
        ],
        1.02,
        4.64,
        10.9,
        1.65,
        size=15,
    )

    note = (
        "The conclusions are bounded by the experimental design. The benchmark is "
        "English-only and uses templated two-turn pressure. Wrong targets are "
        "deterministic rather than selected as the most plausible distractor. P5 comes "
        "from one reference model. M3 uses only 36 tuning questions and one global "
        "direction. M4 uses template fallbacks, no hyperparameter sweep, and a leaky "
        "pair-level validation split. Qualitative subgroups are small, and the extraction "
        "audit is retrospective after rule revision. These limitations do not erase the "
        "held-out test comparisons, but they constrain causal and cross-domain claims."
    )
    slide = new_slide("The evidence supports a benchmark-specific, not universal, claim", note, 13, "Boundaries")
    limitations = [
        ("Interaction", "English, templated, two-turn conversations"),
        ("Attack", "Deterministic wrong targets; one P5 generator"),
        ("M3", "36 tuning questions; one global steering direction"),
        ("M4", "Fallback responses; no sweep; split overlap"),
        ("Audits", "Small subgroups; retrospective extraction check"),
    ]
    for index, (label, body) in enumerate(limitations):
        y = 1.3 + index * 1.02
        color = [BLUE, CORAL, GOLD, TEAL, NAVY][index]
        add_rect(slide, 0.82, y, 0.1, 0.68, color)
        add_text(slide, label, 1.16, y + 0.02, 1.42, 0.34, size=15, color=color, bold=True)
        add_text(slide, body, 2.68, y + 0.02, 8.95, 0.34, size=17, bold=True)
    add_text(slide, "What remains supported", 8.88, 6.12, 2.0, 0.25, size=9, color=TEAL, bold=True)
    add_text(slide, "Pushback type matters, FR alone is incomplete, and M4 offers the best tested 7B trade-off.", 1.0, 6.46, 11.2, 0.36, size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    note = (
        "Three conclusions are most defensible. First, social cues matter: authority "
        "pressure is more damaging than a simple request to reconsider. Second, a low "
        "flip rate is not enough; every mitigation should be evaluated jointly on "
        "robustness and correctability. Third, balanced preference optimization gives "
        "the best tested trade-off for Qwen2.5-7B, while activation steering fails to "
        "generalize in this small-data implementation. The next decisive studies are "
        "human-written attacks, question-disjoint preference validation, independent "
        "capability benchmarks, and a new locked extraction audit."
    )
    slide = new_slide("Selective updating—not stubbornness—is the mitigation target", note, 14, "Takeaways")
    takeaways = [
        ("01", "Authority pressure dominates", "P3 macro-average FR = 65.0%", CORAL),
        ("02", "FR must be paired with CR", "Discernment exposes stubbornness", BLUE),
        ("03", "M4 gives the best tested 7B balance", "+11.87 pp paired accuracy", TEAL),
    ]
    for index, (number, title, sub, color) in enumerate(takeaways):
        y = 1.32 + index * 1.48
        add_text(slide, number, 0.85, y, 0.7, 0.48, size=24, color=color, bold=True)
        add_text(slide, title, 1.72, y, 6.7, 0.38, size=22, bold=True)
        add_text(slide, sub, 1.74, y + 0.48, 5.7, 0.3, size=12, color=SLATE)
        add_rect(slide, 8.65, y + 0.08, 3.55, 0.62, color, radius=True)
        callout = ["65.0% FR", "D = CR − FR", "+11.87 pp"][index]
        add_text(slide, callout, 8.9, y + 0.22, 3.05, 0.26, size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.85, 6.14, 11.35, 0.62, NAVY, radius=True)
    add_text(slide, "Next: human-written attacks · question-disjoint validation · independent capability tests", 1.08, 6.32, 10.9, 0.24, size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(OUTPUT)
    lines = ["# English presentation script", ""]
    for index, (title, note) in enumerate(scripts, 1):
        lines.extend([f"## Slide {index} — {title}", "", note, ""])
    SCRIPT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {SCRIPT_OUTPUT}")


if __name__ == "__main__":
    build()
