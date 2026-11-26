#!/usr/bin/env python3
"""Build the three-seed M4 revalidation figure from frozen analysis outputs."""

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
M4_DIR = ROOT / "analysis/m4_revalidation"
BASELINE_CELLS = ROOT / "analysis/factorial/qwen2.5-7b-mitigations/cells_all_runs.csv"
OBSERVATIONAL = ROOT / "analysis/reextracted/aggregate_all.csv"
SKILL_SCRIPTS = os.environ.get("NATURE_FIGURE_SCRIPTS")
if SKILL_SCRIPTS:
    sys.path.insert(0, SKILL_SCRIPTS)
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional alignment QA outside the Nature figure environment."""
        return None

COLORS = {
    "incorrect": "#D98270",
    "correct": "#3B7EA1",
    "capability": "#287D6D",
    "seed": "#9AA7B2",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
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


def main() -> None:
    factorial = pd.read_csv(M4_DIR / "factorial_cells.csv")
    benchmark = pd.read_csv(M4_DIR / "benchmark_by_seed.csv")
    capability = pd.read_csv(M4_DIR / "independent_capability_regression.csv")
    baseline_cells = pd.read_csv(BASELINE_CELLS)
    observational = pd.read_csv(OBSERVATIONAL)

    baseline_adoption = (
        baseline_cells.query("run == 'baseline'")
        .groupby("feedback_correct")["feedback_adoption_rate"]
        .mean()
    )
    seed_adoption = (
        factorial.groupby(["run", "feedback_correct"], as_index=False)[
            "feedback_adoption_rate"
        ]
        .mean()
    )
    adoption_summary = (
        seed_adoption.groupby("feedback_correct")["feedback_adoption_rate"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    baseline_obs = observational[
        observational["model_id"].astype(str).str.contains(
            "Qwen2.5-7B-Instruct", regex=False
        )
        & (observational["mitigation"] == "none")
        & (observational["dataset"] == "all")
        & (observational["pushback_type"] == "all")
    ].iloc[0]

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    seed_adoption.to_csv(SOURCE_DIR / "figure_m4_controlled_by_seed.csv", index=False)
    adoption_summary.to_csv(
        SOURCE_DIR / "figure_m4_controlled_summary.csv", index=False
    )
    benchmark.to_csv(SOURCE_DIR / "figure_m4_observational_by_seed.csv", index=False)
    capability.to_csv(SOURCE_DIR / "figure_m4_capability_by_seed.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4))
    rng_offsets = np.array([-0.045, 0.0, 0.045])

    ax = axes[0]
    for feedback_correct, color, marker, label in [
        (False, COLORS["incorrect"], "o", "Incorrect feedback"),
        (True, COLORS["correct"], "s", "Correct feedback"),
    ]:
        base = float(baseline_adoption.loc[feedback_correct]) * 100
        values = (
            seed_adoption.query("feedback_correct == @feedback_correct")[
                "feedback_adoption_rate"
            ].to_numpy()
            * 100
        )
        ax.plot([0, 1], [base, values.mean()], color=color, marker=marker, lw=1.4, ms=4, label=label)
        ax.scatter(1 + rng_offsets, values, color=COLORS["seed"], s=10, zorder=2)
        ax.errorbar(
            1,
            values.mean(),
            yerr=values.std(ddof=1),
            color=color,
            capsize=2,
            lw=1.2,
            zorder=3,
        )
    ax.set_xticks([0, 1], ["Baseline", "M4"])
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Controlled adoption (%)")
    ax.legend(loc="center left", handlelength=1.5)
    panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(2)
    for index, (metric, label, color, marker, baseline_value) in enumerate(
        [
            ("FR", "FR", COLORS["incorrect"], "o", float(baseline_obs["flip_rate"])),
            ("CR", "CR", COLORS["correct"], "s", float(baseline_obs["correction_rate"])),
        ]
    ):
        values = benchmark[metric].to_numpy() * 100
        ax.plot(
            x,
            [baseline_value * 100, values.mean()],
            color=color,
            marker=marker,
            lw=1.4,
            ms=4,
            label=label,
        )
        ax.scatter(1 + rng_offsets, values, color=COLORS["seed"], s=10, zorder=2)
        ax.errorbar(
            1,
            values.mean(),
            yerr=values.std(ddof=1),
            color=color,
            capsize=2,
            lw=1.2,
            zorder=3,
        )
    ax.set_xticks(x, ["Baseline", "M4"])
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Observational rate (%)")
    ax.legend(loc="lower left", ncol=2, handlelength=1.4)
    panel_label(ax, "b")

    ax = axes[2]
    datasets = ["all", "mmlu", "gsm8k"]
    labels = ["Overall", "MMLU", "GSM8K"]
    y = np.arange(len(datasets))[::-1]
    ax.axvline(0, color="#777777", lw=0.8)
    for y_pos, dataset in zip(y, datasets, strict=True):
        values = capability.query("dataset == @dataset")["paired_difference"].to_numpy() * 100
        ax.scatter(values, np.full(len(values), y_pos) + rng_offsets, color=COLORS["seed"], s=12, zorder=2)
        ax.errorbar(
            values.mean(),
            y_pos,
            xerr=values.std(ddof=1),
            fmt="o",
            color=COLORS["capability"],
            ecolor=COLORS["capability"],
            capsize=2,
            lw=1.2,
            ms=4,
            zorder=3,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xlabel("Paired capability change (pp)")
    panel_label(ax, "c")

    stem = "figure_m4_revalidation"
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.22, top=0.88, wspace=0.42)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        row_groups=[["a", "b", "c"]],
        json_out=str(FIG_DIR / f"{stem}.alignment.json"),
        overlay_svg=str(FIG_DIR / f"{stem}.alignment.svg"),
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


if __name__ == "__main__":
    main()
