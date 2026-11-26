from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


COLORS = {
    "target": "#3B7EA1",
    "other": "#D9A441",
    "unresolved": "#CBD5E1",
    "M2": "#3B7EA1",
    "M4": "#2A9D78",
    "real": "#7A8B99",
    "fallback": "#D9A441",
    "overlap": "#C85A54",
    "independent": "#2A9D78",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-12, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        json_out=str(output) + ".alignment.json",
        overlay_svg=str(output) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output.with_suffix(".png"), dpi=600, bbox_inches="tight"
    )
    figure.savefig(
        output.with_suffix(".tiff"), dpi=600, bbox_inches="tight"
    )


def plot_construct_and_tradeoff(
    decomposition_path: Path,
    strata_path: Path,
    output: Path,
) -> None:
    decomposition = pd.read_csv(decomposition_path)
    macro = decomposition[
        decomposition["aggregation"] == "five_model_macro"
    ].copy()
    macro["unresolved_rate"] = (
        macro["flip_rate"]
        - macro["target_flip_rate"]
        - macro["other_error_rate"]
    ).clip(lower=0)
    order = ["P1", "P2", "P3", "P4", "P5"]
    macro["pushback_type"] = pd.Categorical(
        macro["pushback_type"], order, ordered=True
    )
    macro = macro.sort_values("pushback_type")

    strata = pd.read_csv(strata_path)
    strata = strata[strata["mitigation"].isin(["M2", "M4"])].copy()

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axis = axes[0]
    positions = np.arange(len(macro))
    bottoms = np.zeros(len(macro))
    segments = [
        ("target_flip_rate", "Target-aligned flip", COLORS["target"]),
        ("other_error_rate", "Other wrong answer", COLORS["other"]),
        ("unresolved_rate", "Invalid / unresolved", COLORS["unresolved"]),
    ]
    for column, label, color in segments:
        values = macro[column].to_numpy() * 100
        axis.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.68,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values
    total = macro["flip_rate"].to_numpy() * 100
    low = macro["flip_rate_ci_low"].to_numpy() * 100
    high = macro["flip_rate_ci_high"].to_numpy() * 100
    axis.errorbar(
        positions,
        total,
        yerr=np.vstack([total - low, high - total]),
        fmt="none",
        ecolor="#273444",
        elinewidth=0.8,
        capsize=2.3,
        capthick=0.8,
        zorder=4,
    )
    axis.set_xticks(
        positions,
        ["P1\n(no target)", "P2", "P3", "P4", "P5"],
    )
    axis.set_ylabel("Initially correct answers that become wrong (%)")
    axis.set_ylim(0, 78)
    axis.set_yticks([0, 20, 40, 60])
    axis.grid(axis="y", color="#E1E6EC", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        ncol=1,
        handlelength=1.2,
        labelspacing=0.35,
    )
    panel_label(axis, "a")

    axis = axes[1]
    categories = [
        ("baseline_first_correct", "Initially correct"),
        ("baseline_first_wrong", "Initially wrong"),
    ]
    y_base = np.array([1.0, 0.0])
    offsets = {"M2": 0.11, "M4": -0.11}
    markers = {"M2": "o", "M4": "D"}
    for method in ("M2", "M4"):
        subset = strata[strata["mitigation"] == method]
        values = []
        lows = []
        highs = []
        for key, _ in categories:
            row = subset[subset["stratum"] == key].iloc[0]
            values.append(row["effect"] * 100)
            lows.append(row["ci_low"] * 100)
            highs.append(row["ci_high"] * 100)
        values_array = np.asarray(values)
        y = y_base + offsets[method]
        axis.errorbar(
            values_array,
            y,
            xerr=np.vstack(
                [values_array - np.asarray(lows), np.asarray(highs) - values_array]
            ),
            fmt=markers[method],
            markersize=5,
            color=COLORS[method],
            ecolor=COLORS[method],
            elinewidth=1.6,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=method,
            zorder=3,
        )
    axis.axvline(0, color="#4B5563", linewidth=0.8, zorder=0)
    axis.grid(axis="x", color="#E1E6EC", linewidth=0.55, zorder=0)
    axis.set_yticks(y_base, [label for _, label in categories])
    axis.set_xlabel("Second-turn accuracy change (percentage points)")
    axis.set_xlim(-28, 31)
    axis.set_xticks([-20, -10, 0, 10, 20, 30])
    axis.tick_params(axis="y", length=0)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        handletextpad=0.4,
    )
    panel_label(axis, "b")

    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.17, top=0.88, wspace=0.34
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output)
    plt.close(figure)


def plot_m4_data_audit(
    preference_manifest_path: Path,
    split_audit_path: Path,
    output: Path,
) -> None:
    preference = json.loads(
        preference_manifest_path.read_text(encoding="utf-8")
    )
    split = json.loads(split_audit_path.read_text(encoding="utf-8"))
    total = preference["train"] + preference["validation"]
    fallback = np.array(
        [preference["chosen_fallback"], preference["rejected_fallback"]]
    )
    real = total - fallback

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axis = axes[0]
    positions = np.arange(2)
    axis.bar(
        positions,
        real / total * 100,
        color=COLORS["real"],
        width=0.58,
        label="Real candidate",
    )
    axis.bar(
        positions,
        fallback / total * 100,
        bottom=real / total * 100,
        color=COLORS["fallback"],
        width=0.58,
        label="Template fallback",
    )
    for x, value, real_value in zip(
        positions, fallback, real, strict=True
    ):
        axis.text(
            x,
            (real_value + value / 2) / total * 100,
            f"{value:,} fallback",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#273444",
        )
    axis.set_xticks(positions, ["Chosen side", "Rejected side"])
    axis.set_ylabel("Preference response sides (%)")
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2
    )
    panel_label(axis, "a")

    axis = axes[1]
    overlap = split["overlapping_questions"]
    independent = split["validation_questions"] - overlap
    axis.barh(
        [0],
        [overlap],
        color=COLORS["overlap"],
        height=0.45,
        label="Also present in train",
    )
    axis.barh(
        [0],
        [independent],
        left=[overlap],
        color=COLORS["independent"],
        height=0.45,
        label="Question-independent",
    )
    axis.text(
        overlap / 2,
        0,
        f"{overlap} ({overlap / (overlap + independent):.1%})",
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
    axis.text(
        overlap + independent / 2,
        0,
        str(independent),
        ha="center",
        va="center",
        color="white",
        fontsize=7,
        fontweight="bold",
    )
    axis.set_xlim(0, overlap + independent)
    axis.set_yticks([])
    axis.set_xlabel("Unique validation questions")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=1
    )
    axis.spines["left"].set_visible(False)
    axis.text(
        0.5,
        -0.37,
        "Pair-level splitting does not provide an\n"
        "independent question-level validation estimate",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="#5B6675",
    )
    panel_label(axis, "b")

    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.25, top=0.78, wspace=0.34
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis/reviewer_reanalysis"),
    )
    parser.add_argument(
        "--preference-manifest",
        type=Path,
        default=Path("artifacts/m4/preferences/manifest.json"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("analysis/figures"),
    )
    args = parser.parse_args()
    plot_construct_and_tradeoff(
        args.analysis_dir / "construct_decomposition.csv",
        args.analysis_dir / "fixed_baseline_strata.csv",
        args.figure_dir / "figure7_construct_and_tradeoff",
    )
    plot_m4_data_audit(
        args.preference_manifest,
        args.analysis_dir / "m4_split_audit.json",
        args.figure_dir / "figureS1_m4_data_audit",
    )
    print("Wrote Figure 7 and Supplementary Figure 1 bundles")


if __name__ == "__main__":
    main()
