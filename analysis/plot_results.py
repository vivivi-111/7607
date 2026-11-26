from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def model_label(model_id: str) -> str:
    name = Path(model_id).name
    return name.replace("-Instruct", "").replace("Qwen2.5-", "Qwen-")


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", default="analysis/figures")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.metrics]
    data = pd.concat(frames, ignore_index=True)
    data["model"] = data["model_id"].map(model_label)
    preferred_order = ["Qwen-1.5B", "Qwen-7B", "Qwen-14B"]
    model_order = [
        *[name for name in preferred_order if name in set(data["model"])],
        *sorted(set(data["model"]) - set(preferred_order)),
    ]
    data["model"] = pd.Categorical(
        data["model"], categories=model_order, ordered=True
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    baseline = data[data["mitigation"] == "none"]
    figure1 = (
        baseline.groupby(
            ["model", "pushback_type"], as_index=False, observed=True
        )["flip_rate"]
        .mean()
        .sort_values("pushback_type")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=figure1,
        x="pushback_type",
        y="flip_rate",
        hue="model",
        ax=ax,
    )
    ax.set(title="Wrong-answer flip rate by pushback type", xlabel="Pushback type")
    ax.set_ylabel("Flip rate (proportion)")
    ax.set_ylim(0, 1)
    save(fig, output, "figure1_flip_rate_by_pushback")

    figure2 = data.groupby(
        ["model", "mitigation"], as_index=False, observed=True
    )[
        ["flip_rate", "correction_rate"]
    ].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=figure2,
        x="flip_rate",
        y="correction_rate",
        hue="model",
        style="mitigation",
        s=90,
        ax=ax,
    )
    for row in figure2.itertuples():
        ax.annotate(
            row.mitigation,
            (row.flip_rate, row.correction_rate),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    for model, group in figure2.groupby("model", observed=True):
        baseline_point = group[group["mitigation"] == "none"]
        if baseline_point.empty:
            continue
        start = baseline_point.iloc[0]
        for _, end in group[group["mitigation"] != "none"].iterrows():
            ax.annotate(
                "",
                xy=(end["flip_rate"], end["correction_rate"]),
                xytext=(start["flip_rate"], start["correction_rate"]),
                arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.9},
            )
    ax.set(
        title="Robustness–correctability trade-off",
        xlabel="Flip rate (lower is better)",
        ylabel="Correction rate (higher is better)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    save(fig, output, "figure2_fr_cr_scatter")

    sizes = {"Qwen-1.5B": 1.5, "Qwen-7B": 7.0, "Qwen-14B": 14.0}
    figure3 = baseline[baseline["model"].isin(sizes)].copy()
    figure3["parameters_b"] = figure3["model"].map(sizes)
    figure3 = figure3.groupby("parameters_b", as_index=False)[
        ["flip_rate", "correction_rate", "D"]
    ].mean()
    figure3 = figure3.melt(
        id_vars="parameters_b",
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.lineplot(
        data=figure3,
        x="parameters_b",
        y="value",
        hue="metric",
        marker="o",
        ax=ax,
    )
    ax.set(
        title="Qwen model-size trend",
        xlabel="Model parameters (billions)",
        ylabel="Metric value (proportion)",
    )
    save(fig, output, "figure3_qwen_size_curve")

    figure4 = baseline.pivot_table(
        index="dataset",
        columns="pushback_type",
        values="flip_rate",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        figure4,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="mako",
        cbar_kws={"label": "Flip rate"},
        ax=ax,
    )
    ax.set(
        title="Mean flip rate by dataset and pushback type",
        xlabel="Pushback type",
        ylabel="Dataset",
    )
    save(fig, output, "figure4_dataset_pushback_heatmap")
    print(f"Wrote figures to {output}")


if __name__ == "__main__":
    main()
