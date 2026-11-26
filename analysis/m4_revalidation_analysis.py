from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from analysis.factorial_feedback_analysis import analyze_effects, cell_summary
from analysis.reviewer_reanalysis import (
    cluster_signflip_p,
    clustered_interval,
    holm_adjust,
)
from sycophancy.data import read_jsonl
from sycophancy.extract import expected_value, extract_answer


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, received {value!r}")
        parsed[name] = Path(path)
    return parsed


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def reextract(
    rows: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        record = records[row["record_id"]]
        extraction = extract_answer(row["response"], record.get("choices"))
        row["extracted"] = extraction.value
        row["extraction_status"] = extraction.status
        row["is_correct"] = extraction.value == expected_value(record)
        output.append(row)
    return output


def benchmark_summary(
    name: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    first = {
        row["record_id"]: row for row in rows if int(row["round"]) == 1
    }
    second = [row for row in rows if int(row["round"]) == 2]
    robust = [
        row for row in second if bool(first[row["record_id"]]["is_correct"])
    ]
    correctable = [
        row for row in second if not bool(first[row["record_id"]]["is_correct"])
    ]
    first_correct = {key: bool(row["is_correct"]) for key, row in first.items()}
    overall = {
        "run": name,
        "n_questions": len(first),
        "n_second_turn": len(second),
        "first_accuracy": mean(first_correct.values()),
        "FR": mean(not bool(row["is_correct"]) for row in robust),
        "CR": mean(bool(row["is_correct"]) for row in correctable),
        "D": (
            mean(bool(row["is_correct"]) for row in correctable)
            - mean(not bool(row["is_correct"]) for row in robust)
        ),
        "second_accuracy": mean(bool(row["is_correct"]) for row in second),
    }
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    for row in first.values():
        by_dataset[row["dataset"]].append(bool(row["is_correct"]))
    dataset_rows = [
        {
            "run": name,
            "dataset": dataset,
            "n": len(values),
            "first_accuracy": mean(values),
        }
        for dataset, values in sorted(by_dataset.items())
    ]
    return overall, dataset_rows, first_correct


def paired_first_turn(
    name: str,
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    common = sorted(set(baseline) & set(candidate))
    output = []
    for dataset in ("all", "gsm8k", "mmlu", "truthfulqa"):
        keys = [
            key
            for key in common
            if dataset == "all" or records[key]["dataset"] == dataset
        ]
        differences = [
            int(candidate[key]) - int(baseline[key]) for key in keys
        ]
        output.append(
            {
                "run": name,
                "dataset": dataset,
                "n": len(keys),
                "baseline_accuracy": mean(baseline[key] for key in keys),
                "candidate_accuracy": mean(candidate[key] for key in keys),
                "paired_difference": mean(differences),
            }
        )
    return output


def seed_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("first_accuracy", "FR", "CR", "D", "second_accuracy")
    return [
        {
            "metric": metric,
            "n_seeds": len(rows),
            "mean": mean(float(row[metric]) for row in rows),
            "sample_sd": (
                stdev(float(row[metric]) for row in rows)
                if len(rows) > 1
                else 0.0
            ),
            "min": min(float(row[metric]) for row in rows),
            "max": max(float(row[metric]) for row in rows),
        }
        for metric in metrics
    ]


def aggregate_estimates(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    value: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row[value]))
    output = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append(
            {
                **dict(zip(keys, group, strict=True)),
                "n_seeds": len(values),
                "mean": mean(values),
                "sample_sd": stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def capability_comparisons(
    baseline_path: Path,
    runs: dict[str, Path],
    bootstrap: int,
    permutations: int,
) -> list[dict[str, Any]]:
    baseline_rows = read_jsonl(baseline_path)
    baseline = {
        row["record_id"]: bool(row["is_correct"]) for row in baseline_rows
    }
    datasets = {
        row["record_id"]: row["dataset"] for row in baseline_rows
    }
    output = []
    for run_index, (name, path) in enumerate(runs.items()):
        candidate = {
            row["record_id"]: bool(row["is_correct"])
            for row in read_jsonl(path)
        }
        common = sorted(set(baseline) & set(candidate))
        for dataset_index, dataset in enumerate(
            ("all", "gsm8k", "mmlu")
        ):
            keys = [
                key
                for key in common
                if dataset == "all" or datasets[key] == dataset
            ]
            differences = [
                int(candidate[key]) - int(baseline[key]) for key in keys
            ]
            seed = 20261125 + run_index * 100 + dataset_index
            low, high = clustered_interval(
                differences, bootstrap, seed
            )
            output.append(
                {
                    "run": name,
                    "dataset": dataset,
                    "n": len(keys),
                    "baseline_accuracy": mean(
                        baseline[key] for key in keys
                    ),
                    "candidate_accuracy": mean(
                        candidate[key] for key in keys
                    ),
                    "paired_difference": mean(differences),
                    "ci_low": low,
                    "ci_high": high,
                    "cluster_signflip_p": cluster_signflip_p(
                        differences, permutations, seed + 10_000
                    ),
                }
            )
    adjusted = holm_adjust(
        [float(row["cluster_signflip_p"]) for row in output]
    )
    for row, value in zip(output, adjusted, strict=True):
        row["holm_p"] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], required=True)
    parser.add_argument("--factorial-run", action="append", default=[])
    parser.add_argument("--capability-baseline", type=Path)
    parser.add_argument("--capability-run", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    args = parser.parse_args()

    records = {row["id"]: row for row in read_jsonl(args.benchmark)}
    baseline_rows = reextract(read_jsonl(args.baseline), records)
    _, _, baseline_first = benchmark_summary("baseline", baseline_rows)
    summaries = []
    datasets = []
    paired = []
    for name, path in parse_named_paths(args.run).items():
        rows = reextract(read_jsonl(path), records)
        summary, dataset_rows, first = benchmark_summary(name, rows)
        summaries.append(summary)
        datasets.extend(dataset_rows)
        paired.extend(paired_first_turn(name, baseline_first, first, records))

    factorial_cells = []
    factorial_effects = []
    for index, (name, path) in enumerate(
        parse_named_paths(args.factorial_run).items()
    ):
        rows = read_jsonl(path)
        expected = len({row["record_id"] for row in rows}) * 16
        if len(rows) != expected:
            raise ValueError(f"incomplete factorial run {name}: {len(rows)}")
        factorial_cells.extend(
            {"run": name, **row} for row in cell_summary(rows)
        )
        factorial_effects.extend(
            {"run": name, **row}
            for row in analyze_effects(
                rows,
                args.bootstrap,
                args.permutations,
                20261125 + index * 10_000,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summaries, args.output_dir / "benchmark_by_seed.csv")
    write_csv(datasets, args.output_dir / "first_accuracy_by_dataset.csv")
    write_csv(paired, args.output_dir / "paired_first_turn_regression.csv")
    write_csv(seed_aggregate(summaries), args.output_dir / "seed_aggregate.csv")
    write_csv(factorial_cells, args.output_dir / "factorial_cells.csv")
    write_csv(factorial_effects, args.output_dir / "factorial_effects.csv")
    if factorial_effects:
        write_csv(
            aggregate_estimates(
                factorial_effects,
                (
                    "effect",
                    "initial_correct",
                    "feedback_correct",
                    "authority",
                    "rationale",
                ),
                "estimate",
            ),
            args.output_dir / "factorial_effects_seed_aggregate.csv",
        )
    capability_runs = parse_named_paths(args.capability_run)
    if args.capability_baseline and capability_runs:
        capability_rows = capability_comparisons(
            args.capability_baseline,
            capability_runs,
            args.bootstrap,
            args.permutations,
        )
        write_csv(
            capability_rows,
            args.output_dir / "independent_capability_regression.csv",
        )
        write_csv(
            aggregate_estimates(
                capability_rows,
                ("dataset",),
                "paired_difference",
            ),
            args.output_dir / "capability_seed_aggregate.csv",
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runs": list(parse_named_paths(args.run)),
                "factorial_runs": list(parse_named_paths(args.factorial_run)),
                "capability_runs": list(capability_runs),
                "seed_uncertainty": "mean, sample SD, and range across three runs",
                "general_capability_check": (
                    "paired accuracy on 500 MMLU and 500 GSM8K questions "
                    "held out from preference construction and the main benchmark"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
