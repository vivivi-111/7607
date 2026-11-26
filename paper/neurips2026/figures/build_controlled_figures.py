#!/usr/bin/env python3
"""Build the controlled-evaluation manuscript figures from frozen CSV outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent
SOURCE_DIR = FIG_DIR / "source_data"
SKILL_SCRIPTS = os.environ.get("NATURE_FIGURE_SCRIPTS")
if SKILL_SCRIPTS:
    sys.path.insert(0, SKILL_SCRIPTS)
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional alignment QA outside the Nature figure environment."""
        return None

BASELINE_DIR = ROOT / "analysis/factorial/baseline_four_model"
MITIGATION_DIR = ROOT / "analysis/factorial/qwen2.5-7b-mitigations"
OBSERVATIONAL_CSV = ROOT / "analysis/reextracted/aggregate_all.csv"

COLORS = {
    "incorrect": "#D98270",
    "correct": "#3B7EA1",
    "effect": "#287D6D",
    "neutral": "#777777",
    "baseline": "#8C8C8C",
    "M1": "#6E9ECF",
    "M2": "#D58B45",
}

MODEL_LABELS = {
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-14b": "Qwen2.5-14B",
    "mistral-7b": "Mistral-7B",
}
MODEL_ORDER = list(MODEL_LABELS)
METHOD_ORDER = ["baseline", "M1", "M2"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_figure(fig: plt.Figure, axes: np.ndarray, stem: str) -> None:
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.88, wspace=0.42)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(FIG_DIR / f"{stem}.alignment.json"),
        overlay_svg=str(FIG_DIR / f"{stem}.alignment.svg"),
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        row_groups=[["a", "b", "c"]],
        require_panel_labels=True,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.svg")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600)
    plt.close(fig)


def build_baselines() -> None:
    cells = pd.read_csv(BASELINE_DIR / "cells_all_runs.csv")
    effects = pd.read_csv(BASELINE_DIR / "effects_all_runs.csv")

    adoption = (
        cells.groupby(["run", "feedback_correct"], as_index=False)[
            "feedback_adoption_rate"
        ]
        .mean()
        .query("run in @MODEL_ORDER")
    )
    truth = effects.query(
        "effect == 'truth_discrimination_in_adoption' "
        "and initial_correct == 'all' and authority == 'all' and rationale == 'all'"
    ).copy()
    authority = effects.query(
        "effect == 'authority_by_truth_interaction' "
        "and initial_correct == 'all' and rationale == 'all'"
    ).copy()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    adoption.to_csv(SOURCE_DIR / "figure_controlled_baselines_adoption.csv", index=False)
    truth.to_csv(SOURCE_DIR / "figure_controlled_baselines_truth.csv", index=False)
    authority.to_csv(
        SOURCE_DIR / "figure_controlled_baselines_authority.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4))

    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    for feedback_correct, label, color, marker in [
        (False, "Incorrect feedback", COLORS["incorrect"], "o"),
        (True, "Correct feedback", COLORS["correct"], "s"),
    ]:
        lookup = adoption.query("feedback_correct == @feedback_correct").set_index(
            "run"
        )
        y = [lookup.loc[model, "feedback_adoption_rate"] * 100 for model in MODEL_ORDER]
        ax.plot(x, y, marker=marker, color=color, lw=1.4, ms=4, label=label)
    ax.set_xticks(
        x,
        ["Qwen 1.5B", "Qwen 7B", "Qwen 14B", "Mistral 7B"],
        rotation=30,
        rotation_mode="anchor",
        ha="right",
    )
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Feedback adoption (%)")
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "a")

    ax = axes[1]
    truth = truth.set_index("run").loc[MODEL_ORDER].reset_index()
    y = np.arange(len(MODEL_ORDER))[::-1]
    values = truth["estimate"].to_numpy() * 100
    lo = truth["ci_low"].to_numpy() * 100
    hi = truth["ci_high"].to_numpy() * 100
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - lo, hi - values]),
        fmt="o",
        color=COLORS["effect"],
        ecolor=COLORS["effect"],
        capsize=2,
        lw=1.2,
        ms=4,
    )
    ax.axvline(0, color="#B8B8B8", lw=0.8)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlim(-2, 58)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Truth discrimination (pp)")
    panel_label(ax, "b")

    ax = axes[2]
    authority = authority.set_index("run").loc[MODEL_ORDER].reset_index()
    values = authority["estimate"].to_numpy() * 100
    lo = authority["ci_low"].to_numpy() * 100
    hi = authority["ci_high"].to_numpy() * 100
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - lo, hi - values]),
        fmt="o",
        color=COLORS["incorrect"],
        ecolor=COLORS["incorrect"],
        capsize=2,
        lw=1.2,
        ms=4,
    )
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_xlim(-9, 9)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("Authority × truth (pp)")
    panel_label(ax, "c")

    finish_figure(fig, axes, "figure_controlled_baselines")


def build_mitigations() -> None:
    cells = pd.read_csv(MITIGATION_DIR / "cells_all_runs.csv")
    paired = pd.read_csv(MITIGATION_DIR / "paired_effects.csv")
    observational = pd.read_csv(OBSERVATIONAL_CSV)

    adoption = (
        cells.groupby(["run", "feedback_correct"], as_index=False)[
            "feedback_adoption_rate"
        ]
        .mean()
        .query("run in @METHOD_ORDER")
    )
    paired_truth = paired.query("metric == 'truth_discrimination'").copy()

    model_mask = observational["model_id"].astype(str).str.contains(
        "Qwen2.5-7B-Instruct", regex=False
    )
    obs = observational[
        model_mask
        & observational["mitigation"].isin(["none", "M1", "M2"])
        & (observational["dataset"] == "all")
        & (observational["pushback_type"] == "all")
    ].copy()
    obs["run"] = obs["mitigation"].replace({"none": "baseline"})
    obs = obs.drop_duplicates("run").set_index("run").loc[METHOD_ORDER].reset_index()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    adoption.to_csv(
        SOURCE_DIR / "figure_controlled_mitigations_adoption.csv", index=False
    )
    paired_truth.to_csv(
        SOURCE_DIR / "figure_controlled_mitigations_paired.csv", index=False
    )
    obs.to_csv(
        SOURCE_DIR / "figure_controlled_mitigations_observational.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4))

    ax = axes[0]
    x = np.arange(len(METHOD_ORDER))
    for feedback_correct, label, color, marker in [
        (False, "Incorrect feedback", COLORS["incorrect"], "o"),
        (True, "Correct feedback", COLORS["correct"], "s"),
    ]:
        lookup = adoption.query("feedback_correct == @feedback_correct").set_index(
            "run"
        )
        y = [lookup.loc[method, "feedback_adoption_rate"] * 100 for method in METHOD_ORDER]
        ax.plot(x, y, marker=marker, color=color, lw=1.4, ms=4, label=label)
    ax.set_xticks(x, ["Baseline", "M1", "M2"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Feedback adoption (%)")
    incorrect_last = adoption.query(
        "feedback_correct == False and run == 'M2'"
    )["feedback_adoption_rate"].iloc[0] * 100
    correct_last = adoption.query(
        "feedback_correct == True and run == 'M2'"
    )["feedback_adoption_rate"].iloc[0] * 100
    ax.text(2.18, incorrect_last, "Incorrect", color=COLORS["incorrect"], va="center")
    ax.text(2.18, correct_last, "Correct", color=COLORS["correct"], va="center")
    ax.set_xlim(-0.1, 2.8)
    panel_label(ax, "a")

    ax = axes[1]
    ax.axvline(0, color="#B8B8B8", lw=0.8)
    ax.plot(0, 2, "o", color=COLORS["baseline"], ms=4)
    for y_pos, method in zip([1, 0], ["M1", "M2"]):
        row = paired_truth.loc[paired_truth["run"] == method].iloc[0]
        value = row["paired_difference"] * 100
        lo = row["ci_low"] * 100
        hi = row["ci_high"] * 100
        ax.errorbar(
            value,
            y_pos,
            xerr=np.array([[value - lo], [hi - value]]),
            fmt="o",
            color=COLORS[method],
            ecolor=COLORS[method],
            capsize=2,
            lw=1.2,
            ms=4,
        )
    ax.set_yticks([2, 1, 0], ["Baseline", "M1", "M2"])
    ax.set_xlim(-5, 45)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Paired Δ truth discrimination (pp)")
    panel_label(ax, "b")

    ax = axes[2]
    x = np.arange(len(METHOD_ORDER))
    ax.plot(
        x,
        obs["flip_rate"] * 100,
        marker="o",
        color=COLORS["incorrect"],
        lw=1.4,
        ms=4,
        label="FR",
    )
    ax.plot(
        x,
        obs["correction_rate"] * 100,
        marker="s",
        color=COLORS["correct"],
        lw=1.4,
        ms=4,
        label="CR",
    )
    ax.set_xticks(x, ["Baseline", "M1", "M2"])
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Observational rate (%)")
    ax.legend(loc="lower left", ncol=2, handlelength=1.4)
    panel_label(ax, "c")

    finish_figure(fig, axes, "figure_controlled_mitigations")


if __name__ == "__main__":
    build_baselines()
    build_mitigations()
