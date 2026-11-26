from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABEL_ORDER = [
    "direct_concession",
    "fabricated_reasoning",
    "hedging",
    "partial_concession",
    "other",
]
LABEL_NAMES = {
    "direct_concession": "Direct concession",
    "fabricated_reasoning": "Fabricated reasoning",
    "hedging": "Hedging",
    "partial_concession": "Partial concession",
    "other": "Other",
}


def proportions(data: pd.DataFrame, dimension: str) -> pd.DataFrame:
    counts = pd.crosstab(data[dimension], data["final_label"])
    counts = counts.reindex(columns=LABEL_ORDER, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default="analysis/annotation/final/final_annotations.csv",
    )
    parser.add_argument(
        "--output",
        default="analysis/figures/figure5_failure_categories",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.annotations)
    sns.set_theme(style="whitegrid", context="paper")
    colors = sns.color_palette("colorblind", n_colors=len(LABEL_ORDER))
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [0.85, 1.2, 1.4]}
    )

    overall = (
        data["final_label"]
        .value_counts(normalize=True)
        .reindex(LABEL_ORDER, fill_value=0)
    )
    axes[0].barh(
        [LABEL_NAMES[label] for label in LABEL_ORDER],
        overall.values,
        color=colors,
    )
    axes[0].invert_yaxis()
    axes[0].set(
        title="Overall failure categories",
        xlabel="Proportion of adjudicated samples",
        ylabel="Category",
        xlim=(0, 0.8),
    )
    for index, value in enumerate(overall.values):
        axes[0].text(value + 0.01, index, f"{value:.0%}", va="center", fontsize=8)

    for axis, dimension, title in (
        (axes[1], "pushback_type", "Categories by pushback"),
        (axes[2], "model", "Categories by model"),
    ):
        table = proportions(data, dimension)
        table.plot(
            kind="barh",
            stacked=True,
            color=colors,
            width=0.75,
            ax=axis,
            legend=False,
        )
        axis.set(
            title=title,
            xlabel="Within-group proportion",
            ylabel=dimension.replace("_", " ").title(),
            xlim=(0, 1),
        )
        axis.invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in colors
    ]
    figure.legend(
        handles,
        [LABEL_NAMES[label] for label in LABEL_ORDER],
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Adjudicated qualitative failure modes (n=100)", fontsize=14
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
