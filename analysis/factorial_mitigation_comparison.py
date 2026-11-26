from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.reviewer_reanalysis import (
    cluster_signflip_p,
    clustered_interval,
    holm_adjust,
)
from sycophancy.data import read_jsonl


def parse_runs(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        output[name] = Path(path)
    return output


def question_adoption_components(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["record_id"]].append(row)
    output = {}
    for record_id, values in groups.items():
        correct = [
            float(bool(row["feedback_adopted"]))
            for row in values
            if bool(row["feedback_correct"])
        ]
        incorrect = [
            float(bool(row["feedback_adopted"]))
            for row in values
            if not bool(row["feedback_correct"])
        ]
        if correct and incorrect:
            output[record_id] = {
                "correct_feedback_adoption": mean(correct),
                "incorrect_feedback_adoption": mean(incorrect),
                "truth_discrimination": mean(correct) - mean(incorrect),
            }
    return output


def compare_runs(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = question_adoption_components(baseline_rows)
    candidate = question_adoption_components(candidate_rows)
    common = sorted(set(baseline) & set(candidate))
    output = []
    for offset, metric in enumerate(
        (
            "incorrect_feedback_adoption",
            "correct_feedback_adoption",
            "truth_discrimination",
        )
    ):
        differences = [
            candidate[key][metric] - baseline[key][metric] for key in common
        ]
        low, high = clustered_interval(
            differences, bootstrap, seed + offset
        )
        output.append(
            {
                "metric": metric,
                "n_questions": len(common),
                "baseline_mean": mean(baseline[key][metric] for key in common),
                "candidate_mean": mean(
                    candidate[key][metric] for key in common
                ),
                "paired_difference": mean(differences),
                "ci_low": low,
                "ci_high": high,
                "cluster_signflip_p": cluster_signflip_p(
                    differences, permutations, seed + 10_000 + offset
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    output = []
    for index, (name, path) in enumerate(parse_runs(args.run).items()):
        output.extend(
            {"run": name, **row}
            for row in compare_runs(
                baseline_rows,
                read_jsonl(path),
                args.bootstrap,
                args.permutations,
                args.seed + index * 100,
            )
        )
    adjusted = holm_adjust(
        [float(row["cluster_signflip_p"]) for row in output]
    )
    for row, value in zip(output, adjusted, strict=True):
        row["holm_p"] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
