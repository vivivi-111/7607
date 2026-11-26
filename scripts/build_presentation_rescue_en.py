"""Build the construct-valid English presentation with embedded speaker notes."""

from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_presentation_en import (
    BLUE,
    CORAL,
    GOLD,
    GRID,
    LIGHT,
    MUTED,
    NAVY,
    SLATE,
    TEAL,
    WHITE,
    add_bullets,
    add_image_contain,
    add_note,
    add_rect,
    add_text,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = ROOT / "paper/neurips2026/figures"
ANALYSIS_FIGURES = ROOT / "analysis/figures"
M4_DIR = ROOT / "analysis/m4_revalidation"
OUTPUT_DIR = ROOT / "docs/final_delivery_en"
OUTPUT = OUTPUT_DIR / "presentation_en.pptx"
SCRIPT_OUTPUT = OUTPUT_DIR / "presentation_en_script.md"

SLIDE_W = 13.333
SLIDE_H = 7.5


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    return next(row for row in rows if row["metric"] == name)


def add_header(slide, title: str, number: int, kicker: str) -> None:
    add_text(slide, f"{number:02d}", 0.48, 0.26, 0.42, 0.3, size=10, color=TEAL, bold=True)
    add_text(slide, title, 0.96, 0.17, 11.85, 0.48, size=24, bold=True)
    add_text(slide, kicker.upper(), 0.98, 0.66, 5.6, 0.18, size=7.5, color=MUTED, bold=True)
    add_rect(slide, 0.48, 0.91, 12.35, 0.012, GRID)


def add_footer(slide, number: int, source: str = "Selective updating study · 2026") -> None:
    add_text(slide, source, 0.5, 7.16, 9.8, 0.17, size=7.5, color=MUTED)
    add_text(slide, str(number), 12.15, 7.14, 0.65, 0.18, size=8, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)


def add_takeaway(slide, text: str, color: RGBColor = NAVY) -> None:
    add_rect(slide, 0.62, 6.42, 12.05, 0.48, color, radius=True)
    add_text(slide, text, 0.88, 6.53, 11.55, 0.24, size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)


def build() -> None:
    seed_aggregate = read_rows(M4_DIR / "seed_aggregate.csv")
    factorial_aggregate = read_rows(M4_DIR / "factorial_effects_seed_aggregate.csv")
    capability_aggregate = read_rows(M4_DIR / "capability_seed_aggregate.csv")
    truth = next(
        row
        for row in factorial_aggregate
        if row["effect"] == "truth_discrimination_in_adoption"
        and row["initial_correct"] == "all"
        and row["authority"] == "all"
        and row["rationale"] == "all"
    )
    capability_all = next(row for row in capability_aggregate if row["dataset"] == "all")
    fr = metric(seed_aggregate, "FR")
    cr = metric(seed_aggregate, "CR")

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    scripts: list[tuple[str, str]] = []

    def slide(title: str, number: int, kicker: str, note: str):
        current = prs.slides.add_slide(blank)
        current.background.fill.solid()
        current.background.fill.fore_color.rgb = WHITE
        add_header(current, title, number, kicker)
        add_footer(current, number)
        add_note(current, note)
        scripts.append((title, note))
        return current

    note = (
        "A helpful assistant must do two things that are easy to conflate: resist unsupported "
        "pressure when it is already correct, and accept valid feedback when it is wrong. "
        "The sharpest result is a reversal: the adaptive protocol reads independent self-review "
        "as stubborn because its correction rate falls from 67.1 to 44.3 percent, while the "
        "controlled protocol shows it is the most selective update, raising the truth contrast "
        "from 21.5 to 58.8 points. The talk then shows what that contrast does and does not measure."
    )
    current = prs.slides.add_slide(blank)
    current.background.fill.solid()
    current.background.fill.fore_color.rgb = NAVY
    add_rect(current, 0.68, 0.72, 0.11, 5.92, TEAL)
    add_text(current, "SELECTIVE UPDATING", 1.08, 1.02, 10.8, 0.4, size=15, color=TEAL, bold=True)
    add_text(current, "Separating sycophancy\nfrom correctability", 1.05, 1.58, 10.7, 1.55, size=39, color=WHITE, bold=True)
    add_text(current, "Old protocol: self-review looks stubborn.\nControlled protocol: it is the most selective.", 1.08, 3.35, 10.6, 0.85, size=18, color=RGBColor(194, 214, 224))
    add_text(current, "NeurIPS 2026 · Anonymous Authors", 1.08, 6.42, 7.2, 0.28, size=11, color=RGBColor(194, 214, 224))
    add_text(current, "538 questions × 16 conditions", 9.4, 6.38, 2.7, 0.3, size=12, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)
    add_note(current, note)
    scripts.append(("Selective Updating under User Pushback", note))

    note = (
        "Low wrong-answer flip rate is not sufficient. A model can obtain it by refusing every "
        "correction. The target behavior is selective updating: low adoption of false feedback "
        "and high adoption of true feedback. These outcomes must be measured on comparable questions."
    )
    current = slide("The safety target is selective updating", 2, "Problem", note)
    add_text(current, "Already correct", 0.85, 1.35, 3.8, 0.35, size=14, color=TEAL, bold=True)
    add_text(current, "RESIST", 0.82, 1.95, 4.2, 0.7, size=38, color=TEAL, bold=True)
    add_text(current, "unsupported feedback", 0.85, 2.68, 3.7, 0.34, size=16, color=SLATE)
    add_rect(current, 5.98, 1.22, 0.018, 4.45, GRID)
    add_text(current, "Initially wrong", 6.52, 1.35, 3.8, 0.35, size=14, color=CORAL, bold=True)
    add_text(current, "REVISE", 6.48, 1.95, 4.2, 0.7, size=38, color=CORAL, bold=True)
    add_text(current, "when feedback is valid", 6.52, 2.68, 4.1, 0.34, size=16, color=SLATE)
    add_text(current, "False-feedback adoption ↓", 1.08, 4.12, 4.5, 0.4, size=21, bold=True)
    add_text(current, "True-feedback adoption ↑", 6.72, 4.12, 4.5, 0.4, size=21, bold=True)
    add_takeaway(current, "A model that never changes its answer is robust only by definition—not by judgment.")

    note = (
        "In the original adaptive protocol, the truth of the feedback is determined by the "
        "model's first response. Correct first answers receive false targets, and incorrect "
        "first answers receive the gold answer. Correction rate can therefore reward simple "
        "agreement, and different models are evaluated on different question subsets."
    )
    current = slide("The adaptive benchmark couples truth to the first answer", 3, "Construct audit", note)
    add_text(current, "Model first answer", 0.82, 1.3, 2.3, 0.3, size=13, color=MUTED, bold=True)
    add_text(current, "CORRECT", 0.82, 1.86, 2.25, 0.48, size=24, color=TEAL, bold=True)
    add_text(current, "WRONG", 0.82, 4.05, 2.25, 0.48, size=24, color=CORAL, bold=True)
    add_text(current, "→", 3.05, 1.86, 0.6, 0.5, size=27, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_text(current, "→", 3.05, 4.05, 0.6, 0.5, size=27, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_rect(current, 3.72, 1.45, 3.5, 1.24, LIGHT, line=GRID, radius=True)
    add_rect(current, 3.72, 3.63, 3.5, 1.24, LIGHT, line=GRID, radius=True)
    add_text(current, "User asserts a wrong target", 4.02, 1.82, 2.9, 0.32, size=16, bold=True, align=PP_ALIGN.CENTER)
    add_text(current, "User asserts the gold answer", 4.02, 4.0, 2.9, 0.32, size=16, bold=True, align=PP_ALIGN.CENTER)
    add_text(current, "Feedback truth", 8.0, 1.3, 2.2, 0.3, size=13, color=MUTED, bold=True)
    add_text(current, "FALSE", 8.0, 1.86, 2.1, 0.48, size=24, color=CORAL, bold=True)
    add_text(current, "TRUE", 8.0, 4.05, 2.1, 0.48, size=24, color=TEAL, bold=True)
    add_text(current, "FR and CR use model-selected denominators", 8.0, 5.03, 3.9, 0.62, size=16, color=NAVY, bold=True)
    add_takeaway(current, "The operational stress test is useful, but it cannot identify sensitivity to feedback truth.")

    note = (
        "We independently cross four binary factors. Every eligible question appears in all "
        "sixteen cells, including cases where the initial answer and the feedback are two "
        "different wrong answers. Authority and rationale are added while the target remains fixed."
    )
    current = slide("A 2×2×2×2 design restores the missing contrast", 4, "Controlled protocol", note)
    factors = [
        ("Initial answer", "correct / wrong", TEAL),
        ("Feedback target", "correct / wrong", CORAL),
        ("Authority cue", "absent / present", BLUE),
        ("Rationale", "absent / present", GOLD),
    ]
    for index, (name, levels, color) in enumerate(factors):
        x = 0.72 + index * 3.12
        add_rect(current, x, 1.55, 2.65, 1.55, LIGHT, line=GRID, radius=True)
        add_rect(current, x, 1.55, 2.65, 0.12, color)
        add_text(current, name, x + 0.2, 1.96, 2.22, 0.3, size=16, bold=True, align=PP_ALIGN.CENTER)
        add_text(current, levels, x + 0.2, 2.48, 2.22, 0.3, size=12, color=SLATE, align=PP_ALIGN.CENTER)
        if index < 3:
            add_text(current, "×", x + 2.72, 2.05, 0.34, 0.34, size=18, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_text(current, "538", 1.1, 4.02, 1.4, 0.5, size=30, color=TEAL, bold=True)
    add_text(current, "paired questions", 1.1, 4.55, 2.0, 0.3, size=12, color=SLATE)
    add_text(current, "8,608", 4.25, 4.02, 1.8, 0.5, size=30, color=BLUE, bold=True)
    add_text(current, "responses per model", 4.25, 4.55, 2.2, 0.3, size=12, color=SLATE)
    add_text(current, "Δtruth", 7.52, 4.02, 1.8, 0.5, size=30, color=CORAL, bold=True)
    add_text(current, "adopt true − false feedback", 7.52, 4.55, 2.7, 0.3, size=12, color=SLATE)
    add_text(current, "Question-paired", 10.25, 4.08, 2.1, 0.35, size=17, bold=True)
    add_text(current, "bootstrap + sign flips", 10.25, 4.52, 2.1, 0.3, size=11, color=SLATE)
    add_takeaway(current, "Feedback truth, authority, rationale, and initial state now vary independently.")

    note = (
        "All four open text models adopt correct feedback more often than incorrect feedback, "
        "but the size of that separation varies sharply. Qwen2.5-14B reaches about fifty points, "
        "while the 1.5B, 7B, and Mistral models are much lower. Authority does not have a universal effect."
    )
    current = slide("Truth sensitivity exists—but differs by model", 5, "Controlled baselines", note)
    add_image_contain(current, PAPER_FIGURES / "figure_controlled_baselines.png", 0.45, 1.12, 12.45, 4.95)
    add_takeaway(current, "Authority reduces truth discrimination for Qwen2.5-7B/14B, not universally across models.")

    note = (
        "The observational benchmark still diagnoses real failures. Under its implemented messages, "
        "P3 produces the most wrong-answer reversals. But P1 does not name a target and P2 through P5 do, "
        "so the result cannot be interpreted as a pure causal ranking of social pressure."
    )
    current = slide("Operational attack strength is not a causal factor ranking", 6, "Observational protocol", note)
    add_text(current, "What the adaptive test supports", 0.78, 1.42, 4.6, 0.35, size=17, color=TEAL, bold=True)
    add_bullets(current, ["Which complete messages trigger failures", "FR and CR under deployed templates", "Failure-mode collection for audit"], 0.78, 1.98, 5.25, 2.65, size=16)
    add_text(current, "What it does not isolate", 6.78, 1.42, 4.6, 0.35, size=17, color=CORAL, bold=True)
    add_bullets(current, ["Authority independent of target naming", "Evidence quality independent of truth", "Cross-model discernment from D alone"], 6.78, 1.98, 5.25, 2.65, size=16)
    add_rect(current, 0.78, 5.18, 11.7, 0.64, LIGHT, line=GRID, radius=True)
    add_text(current, "Use both protocols: one for operational exposure, one for construct-valid inference.", 1.05, 5.36, 11.1, 0.26, size=15, bold=True, align=PP_ALIGN.CENTER)
    add_takeaway(current, "The repaired story keeps the stress test, but narrows what its template ordering can claim.")

    note = (
        "M1 provides a modest improvement. M2 changes the policy much more strongly: false-feedback "
        "adoption falls by fifty points, while true-feedback adoption falls by thirteen points. "
        "That cost matters, but M2 still accepts over eighty-two percent of true feedback."
    )
    current = slide("Independent self-review improves selective updating", 7, "Prompt mitigations", note)
    add_image_contain(current, PAPER_FIGURES / "figure_controlled_mitigations.png", 0.48, 1.12, 12.35, 4.95)
    add_takeaway(current, "M2 is not blanket stubbornness: 82.2% true-feedback adoption versus 23.4% false-feedback adoption.")

    note = (
        "The activation direction looked useful on thirty-six tuning questions, but it did not "
        "transfer to the full observational test. The paired accuracy point estimate is negative, "
        "yet it is not significant after Holm correction. The defensible conclusion is failure to "
        "generalize, not evidence that activation steering is generally harmful."
    )
    current = slide("The tested activation direction does not generalize", 8, "M3 boundary result", note)
    add_text(current, "Tuning subset", 0.92, 1.52, 3.0, 0.35, size=15, color=MUTED, bold=True)
    add_text(current, "D  0.150 → 0.264", 0.9, 2.08, 4.2, 0.58, size=29, color=TEAL, bold=True)
    add_text(current, "36 questions", 0.92, 2.78, 2.2, 0.28, size=12, color=SLATE)
    add_text(current, "Full test", 6.72, 1.52, 3.0, 0.35, size=15, color=MUTED, bold=True)
    add_text(current, "Δ accuracy  −2.31 pp", 6.7, 2.08, 5.15, 0.58, size=29, color=CORAL, bold=True)
    add_text(current, "95% CI −4.47 to −0.19 · Holm p = 0.114", 6.72, 2.78, 4.7, 0.3, size=12, color=SLATE)
    add_rect(current, 0.9, 4.15, 11.55, 1.1, LIGHT, line=GRID, radius=True)
    add_text(current, "Small tuning set + one global direction + shared construction/tuning data", 1.15, 4.49, 11.0, 0.32, size=17, bold=True, align=PP_ALIGN.CENTER)
    add_takeaway(current, "Negative transfer here is a method-instance result, not a verdict on activation steering.")

    note = (
        "We rebuilt M4 to remove the strongest validity threats. The preference split is disjoint "
        "by question and from the benchmark test, every chosen and rejected response is a real model "
        "response, and training is repeated with three seeds. We also add a thousand-question "
        "capability holdout that is disjoint from preference construction and the benchmark."
    )
    current = slide("M4 revalidation removes overlap and template shortcuts", 9, "DPO design", note)
    stages = [
        ("1,908", "real-response\npreference pairs", TEAL),
        ("0", "train–validation\nquestion overlap", BLUE),
        ("3", "independent\ntraining seeds", GOLD),
        ("1,000", "disjoint capability\nquestions", CORAL),
    ]
    for index, (value, label, color) in enumerate(stages):
        x = 0.68 + index * 3.12
        add_text(current, value, x, 1.58, 2.5, 0.58, size=31, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(current, label, x, 2.26, 2.5, 0.72, size=13, color=SLATE, bold=True, align=PP_ALIGN.CENTER)
    add_rect(current, 0.88, 3.65, 11.58, 1.62, LIGHT, line=GRID, radius=True)
    add_text(current, "ROBUSTNESS PAIRS", 1.24, 4.02, 2.65, 0.27, size=11, color=TEAL, bold=True)
    add_text(current, "resist a false target", 1.24, 4.45, 2.65, 0.27, size=16, bold=True)
    add_text(current, "+", 5.08, 4.2, 0.5, 0.4, size=22, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_text(current, "CORRECTABILITY PAIRS", 6.18, 4.02, 3.0, 0.27, size=11, color=CORAL, bold=True)
    add_text(current, "accept a valid correction", 6.18, 4.45, 3.1, 0.27, size=16, bold=True)
    add_text(current, "→  LoRA + DPO", 9.75, 4.18, 2.1, 0.35, size=19, color=NAVY, bold=True)
    add_takeaway(current, "The revalidation tests selective updating, seed stability, and capability regression together.")

    note = (
        f"Across three seeds, the controlled truth-discrimination estimate is "
        f"{float(truth['mean']) * 100:.1f} percentage points with a seed standard deviation of "
        f"{float(truth['sample_sd']) * 100:.1f}. Observational FR averages "
        f"{float(fr['mean']) * 100:.1f} percent and CR averages {float(cr['mean']) * 100:.1f} percent. "
        f"On the independent capability holdout, the mean paired change is "
        f"{float(capability_all['mean']) * 100:+.1f} points. Every seed interval lies inside "
        f"about minus 2.1 to plus 2.8 points. That is not an equivalence proof. The direct "
        f"evidence against stubbornness is correct-feedback adoption, which stays at 96.2 percent."
    )
    current = slide("Three-seed DPO is evaluated on behavior and capability", 10, "M4 result", note)
    add_image_contain(current, PAPER_FIGURES / "figure_m4_revalidation.png", 0.48, 1.08, 12.35, 5.0)
    add_takeaway(
        current,
        f"Mean Δtruth {float(truth['mean']) * 100:.1f} pp · capability change {float(capability_all['mean']) * 100:+.1f} pp",
    )

    note = (
        "Human coding shows that many failures contain newly rationalized content rather than a terse "
        "answer switch. Independent failure-category agreement is moderate and all disagreements are "
        "adjudicated. The extractor's perfect retrospective score does not survive the locked audit. "
        "Independent annotators agree on 96.5 percent of 200 responses, but the frozen extractor matches "
        "only 89.5 percent, with a 95 percent interval from 84.5 to 93.0 percent. It therefore fails the "
        "prespecified 98 percent threshold. Most errors occur when the extractor returns no answer."
    )
    current = slide("The locked audit fails the extraction threshold", 11, "Measurement audit", note)
    add_image_contain(current, ANALYSIS_FIGURES / "figure5_failure_categories.png", 0.52, 1.12, 7.65, 4.9)
    add_text(current, "100 failures", 8.65, 1.45, 2.8, 0.4, size=26, color=TEAL, bold=True)
    add_text(current, "κ = 0.417 · 41 adjudications", 8.65, 2.05, 3.35, 0.3, size=13, color=SLATE)
    add_text(current, "193 / 200", 8.65, 3.15, 2.8, 0.4, size=26, color=BLUE, bold=True)
    add_text(current, "human agreement · κ = 0.962", 8.65, 3.75, 3.65, 0.3, size=13, color=SLATE)
    add_text(current, "179 / 200", 8.65, 4.72, 3.0, 0.42, size=24, color=CORAL, bold=True)
    add_text(current, "extractor match · 89.5% [84.5, 93.0]", 8.65, 5.30, 3.85, 0.3, size=12, color=SLATE)
    add_takeaway(current, "The construct repair stands; automated effect sizes remain provisional.")

    note = (
        "The final story has three layers. Adaptive tests expose operational failures. The factorial "
        "experiment identifies truth-sensitive updating and model heterogeneity. Mitigations must then "
        "be judged on both false- and true-feedback adoption, across seeds, with capability checks. "
        "But measurement validity is now the limiting factor: the locked extractor audit fails its "
        "prespecified threshold, so the design conclusion is stronger than the numerical effect claims."
    )
    current = slide("Measure judgment—not answer persistence", 12, "Conclusion", note)
    add_text(current, "1", 0.78, 1.38, 0.5, 0.45, size=25, color=TEAL, bold=True)
    add_text(current, "Stress tests reveal operational failures", 1.55, 1.38, 5.2, 0.42, size=20, bold=True)
    add_text(current, "2", 0.78, 2.57, 0.5, 0.45, size=25, color=BLUE, bold=True)
    add_text(current, "Factorial contrasts identify truth sensitivity", 1.55, 2.57, 5.8, 0.42, size=20, bold=True)
    add_text(current, "3", 0.78, 3.76, 0.5, 0.45, size=25, color=CORAL, bold=True)
    add_text(current, "Mitigation must preserve warranted revision", 1.55, 3.76, 5.7, 0.42, size=20, bold=True)
    add_rect(current, 7.7, 1.32, 4.55, 3.65, LIGHT, line=GRID, radius=True)
    add_text(current, "Boundaries", 8.08, 1.72, 2.2, 0.34, size=17, color=MUTED, bold=True)
    add_bullets(current, ["English, two-turn factual QA", "Templated feedback and one rationale generator", "DPO on one 7B base model", "Extractor audit: 89.5%, below 98% target"], 8.02, 2.28, 3.9, 2.25, size=13)
    add_takeaway(current, "Construct-valid contrasts still require independently validated extraction.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    script_lines = ["# Speaker script", ""]
    for index, (title, text) in enumerate(scripts, start=1):
        script_lines.extend([f"## Slide {index}: {title}", "", text, ""])
    SCRIPT_OUTPUT.write_text("\n".join(script_lines), encoding="utf-8")


if __name__ == "__main__":
    build()
