from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


MODEL_LABELS = {
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2.5-14b": "Qwen2.5-14B",
    "mistral-7b": "Mistral-7B",
    "qwen3.8-27b-api": "qwen3.8-27B",
}
METHOD_COLORS = {
    "M1": "#8A9BA8",
    "M2": "#3B7EA1",
    "M3": "#C85A54",
    "M4": "#2A9D78",
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
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def label_model(value: str) -> str:
    replacements = {
        **MODEL_LABELS,
    }
    return replacements.get(value, value)


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        json_out=str(output) + ".alignment.json",
        overlay_svg=str(output) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "analysis/reviewer_reanalysis/"
            "clustered_mitigation_effects.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="analysis/figures/figure6_paired_accuracy_effects",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data["difference_pp"] = data["effect"] * 100
    data["low_pp"] = data["ci_low"] * 100
    data["high_pp"] = data["ci_high"] * 100
    mitigation_order = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    model_order = {
        "qwen2.5-1.5b": 1,
        "qwen2.5-7b": 2,
        "qwen2.5-14b": 3,
        "mistral-7b": 4,
        "qwen3.8-27b-api": 5,
    }
    data["sort"] = data["model"].map(model_order) * 10 + data[
        "mitigation"
    ].map(mitigation_order)
    data = data.sort_values("sort")
    labels = [
        f"{label_model(row.model)} {row.mitigation}"
        for row in data.itertuples()
    ]
    positions = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(7.2, 5.3))
    axis.axvline(0, color="#4B5563", linewidth=0.8, zorder=0)
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.55, zorder=0)
    for position, row in zip(positions, data.itertuples(), strict=True):
        color = METHOD_COLORS[row.mitigation]
        axis.plot(
            [row.low_pp, row.high_pp],
            [position, position],
            color=color,
            linewidth=2.1,
            solid_capstyle="round",
            zorder=2,
        )
        axis.scatter(
            row.difference_pp,
            position,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.text(
            38.2,
            position,
            f"{row.difference_pp:+.1f} "
            f"[{row.low_pp:+.1f}, {row.high_pp:+.1f}]",
            va="center",
            ha="right",
            fontsize=6.3,
            color="#273444",
        )
    axis.set(
        xlabel="Change in second-turn accuracy (percentage points)",
        ylabel="",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(-9, 39)
    axis.set_xticks([-5, 0, 5, 10, 15, 20, 25])
    axis.tick_params(axis="y", length=0, pad=5)
    model_values = list(data["model"])
    for index in range(len(model_values) - 1):
        if model_values[index] != model_values[index + 1]:
            axis.axhline(
                index + 0.5, color="#E5E9EF", linewidth=0.7, zorder=0
            )
    axis.text(
        0,
        1.02,
        "Question-clustered bootstrap 95% CI; "
        "n = independent benchmark questions",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#5B6675",
    )
    figure.subplots_adjust(left=0.25, right=0.98, bottom=0.11, top=0.94)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output)
    plt.close(figure)
    print(f"Wrote publication figure bundle at {output}")


if __name__ == "__main__":
    main()
