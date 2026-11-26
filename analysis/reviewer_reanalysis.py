from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sycophancy.data import read_jsonl


RUNS = {
    "qwen2.5-1.5b": {
        "baseline": "results/qwen2.5-1.5b_baseline/dialogues.jsonl",
        "M1": "results/qwen2.5-1.5b_M1/dialogues.jsonl",
        "M2": "results/qwen2.5-1.5b_M2/dialogues.jsonl",
    },
    "qwen2.5-7b": {
        "baseline": "results/qwen2.5-7b_baseline/dialogues.jsonl",
        "M1": "results/qwen2.5-7b_M1/dialogues.jsonl",
        "M2": "results/qwen2.5-7b_M2/dialogues.jsonl",
        "M3": "results/qwen2.5-7b_M3/dialogues.jsonl",
        "M4": "results/qwen2.5-7b_M4/dialogues.jsonl",
    },
    "qwen2.5-14b": {
        "baseline": "results/qwen2.5-14b_baseline/dialogues.jsonl",
        "M1": "results/qwen2.5-14b_M1/dialogues.jsonl",
        "M2": "results/qwen2.5-14b_M2/dialogues.jsonl",
    },
    "mistral-7b": {
        "baseline": "results/mistral-7b_none/dialogues.jsonl",
        "M1": "results/mistral-7b_M1/dialogues.jsonl",
        "M2": "results/mistral-7b_M2/dialogues.jsonl",
    },
    "qwen3.8-27b-api": {
        "baseline": "results/qwen3.8-27b-api_baseline/dialogues.jsonl",
        "M1": "results/qwen3.8-27b-api_M1/dialogues.jsonl",
        "M2": "results/qwen3.8-27b-api_M2/dialogues.jsonl",
    },
}


def run_path(root: Path, results_dir: Path, path: str) -> Path:
    relative = Path(path)
    if relative.parts[0] != "results":
        raise ValueError(f"unexpected run path: {path}")
    return root / results_dir / Path(*relative.parts[1:])


def dialogue_indices(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = read_jsonl(path)
    first = {
        row["record_id"]: row for row in rows if int(row["round"]) == 1
    }
    second = {
        (row["record_id"], row["pushback_type"]): row
        for row in rows
        if int(row["round"]) == 2
    }
    return first, second


def clustered_interval(
    values: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 1_000):
        stop = min(start + 1_000, iterations)
        indices = rng.integers(
            0, len(array), size=(stop - start, len(array))
        )
        samples[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def cluster_signflip_p(
    values: list[float],
    iterations: int,
    seed: int,
) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    for start in range(0, iterations, 10_000):
        size = min(10_000, iterations - start)
        signs = rng.integers(0, 2, size=(size, len(array)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * array).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += size
    return (extreme + 1) / (completed + 1)


def clustered_comparison(
    baseline: dict[tuple[str, str], dict[str, Any]],
    mitigation: dict[tuple[str, str], dict[str, Any]],
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    allowed_records: set[str] | None = None,
) -> dict[str, float | int]:
    common = sorted(set(baseline) & set(mitigation))
    if allowed_records is not None:
        common = [key for key in common if key[0] in allowed_records]
    by_question: dict[str, list[int]] = defaultdict(list)
    baseline_only = 0
    mitigation_only = 0
    for key in common:
        baseline_correct = bool(baseline[key]["is_correct"])
        mitigation_correct = bool(mitigation[key]["is_correct"])
        baseline_only += int(baseline_correct and not mitigation_correct)
        mitigation_only += int(mitigation_correct and not baseline_correct)
        by_question[key[0]].append(
            int(mitigation_correct) - int(baseline_correct)
        )
    question_effects = [
        sum(values) / len(values) for values in by_question.values()
    ]
    low, high = clustered_interval(
        question_effects, bootstrap_iterations, seed
    )
    return {
        "n_pairs": len(common),
        "n_questions": len(by_question),
        "baseline_correct_only": baseline_only,
        "mitigation_correct_only": mitigation_only,
        "effect": float(np.mean(question_effects)),
        "ci_low": low,
        "ci_high": high,
        "cluster_signflip_p": cluster_signflip_p(
            question_effects, permutation_iterations, seed + 10_000
        ),
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def mitigation_effects(
    root: Path,
    results_dir: Path,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_index, (model, paths) in enumerate(RUNS.items()):
        _, baseline = dialogue_indices(
            run_path(root, results_dir, paths["baseline"])
        )
        for method_index, (method, path) in enumerate(paths.items()):
            if method == "baseline":
                continue
            _, mitigation = dialogue_indices(
                run_path(root, results_dir, path)
            )
            comparison = clustered_comparison(
                baseline,
                mitigation,
                bootstrap_iterations,
                permutation_iterations,
                seed + model_index * 100 + method_index,
            )
            output.append(
                {
                    "model": model,
                    "mitigation": method,
                    **comparison,
                }
            )
    adjusted = holm_adjust(
        [float(row["cluster_signflip_p"]) for row in output]
    )
    for row, value in zip(output, adjusted, strict=True):
        row["holm_p"] = value
    return output


def fixed_baseline_strata(
    root: Path,
    results_dir: Path,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_first, baseline_second = dialogue_indices(
        run_path(
            root,
            results_dir,
            RUNS["qwen2.5-7b"]["baseline"],
        )
    )
    output: list[dict[str, Any]] = []
    for method_index, method in enumerate(("M1", "M2", "M3", "M4")):
        _, mitigation_second = dialogue_indices(
            run_path(
                root,
                results_dir,
                RUNS["qwen2.5-7b"][method],
            )
        )
        all_common = set(baseline_second) & set(mitigation_second)
        matched_common = {
            key
            for key in all_common
            if key[1] == "P1"
            or baseline_second[key].get("target")
            == mitigation_second[key].get("target")
        }
        matched_baseline = {
            key: baseline_second[key] for key in matched_common
        }
        matched_mitigation = {
            key: mitigation_second[key] for key in matched_common
        }
        for stratum_index, (stratum, first_correct) in enumerate(
            (("baseline_first_correct", True), ("baseline_first_wrong", False))
        ):
            allowed = {
                record_id
                for record_id, row in baseline_first.items()
                if bool(row["is_correct"]) is first_correct
            }
            comparison = clustered_comparison(
                matched_baseline,
                matched_mitigation,
                bootstrap_iterations,
                permutation_iterations,
                seed + method_index * 10 + stratum_index,
                allowed,
            )
            common = sorted(
                key
                for key in matched_common
                if key[0] in allowed
            )
            baseline_accuracy = float(
                np.mean(
                    [
                        bool(matched_baseline[key]["is_correct"])
                        for key in common
                    ]
                )
            )
            mitigation_accuracy = float(
                np.mean(
                    [
                        bool(matched_mitigation[key]["is_correct"])
                        for key in common
                    ]
                )
            )
            if first_correct:
                transition_metric = "fixed_baseline_FR"
                baseline_transition = 1 - baseline_accuracy
                mitigation_transition = 1 - mitigation_accuracy
                transition_effect = -float(comparison["effect"])
                transition_ci_low = -float(comparison["ci_high"])
                transition_ci_high = -float(comparison["ci_low"])
            else:
                transition_metric = "fixed_baseline_CR"
                baseline_transition = baseline_accuracy
                mitigation_transition = mitigation_accuracy
                transition_effect = float(comparison["effect"])
                transition_ci_low = float(comparison["ci_low"])
                transition_ci_high = float(comparison["ci_high"])
            comparison.update(
                {
                    "model": "qwen2.5-7b",
                    "mitigation": method,
                    "stratum": stratum,
                    "baseline_second_accuracy": baseline_accuracy,
                    "mitigation_second_accuracy": mitigation_accuracy,
                    "transition_metric": transition_metric,
                    "baseline_transition_value": baseline_transition,
                    "mitigation_transition_value": mitigation_transition,
                    "transition_effect": transition_effect,
                    "transition_ci_low": transition_ci_low,
                    "transition_ci_high": transition_ci_high,
                    "excluded_target_mismatch_pairs": len(all_common)
                    - len(matched_common),
                }
            )
            output.append(comparison)
    adjusted = holm_adjust(
        [float(row["cluster_signflip_p"]) for row in output]
    )
    for row, value in zip(output, adjusted, strict=True):
        row["holm_p"] = value
    return output


def construct_decomposition(
    root: Path,
    results_dir: Path,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    per_model: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], dict[str, list[float]]] = {}
    for model, paths in RUNS.items():
        first, second = dialogue_indices(
            run_path(root, results_dir, paths["baseline"])
        )
        for pushback in ("P1", "P2", "P3", "P4", "P5"):
            rows = [
                row
                for (record_id, kind), row in second.items()
                if kind == pushback
                and record_id in first
                and bool(first[record_id]["is_correct"])
            ]
            metrics = {
                "flip_rate": [
                    float(not bool(row["is_correct"])) for row in rows
                ],
                "target_flip_rate": [
                    float(bool(row["is_target_flip"])) for row in rows
                ],
                "other_error_rate": [
                    float(bool(row["is_other_error"])) for row in rows
                ],
            }
            raw[(model, pushback)] = metrics
            per_model.append(
                {
                    "aggregation": "model",
                    "model": model,
                    "pushback_type": pushback,
                    "n_questions": len(rows),
                    **{
                        metric: float(np.mean(values))
                        for metric, values in metrics.items()
                    },
                }
            )
    macro: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    model_groups = {
        "four_open_text_model_macro": [
            model for model in RUNS if model != "qwen3.8-27b-api"
        ],
        "five_model_exploratory_macro": list(RUNS),
    }
    for aggregation, models in model_groups.items():
        for pushback in ("P1", "P2", "P3", "P4", "P5"):
            row: dict[str, Any] = {
                "aggregation": aggregation,
                "model": "+".join(models),
                "pushback_type": pushback,
                "n_questions": sum(
                    len(raw[(model, pushback)]["flip_rate"])
                    for model in models
                ),
            }
            for metric in (
                "flip_rate",
                "target_flip_rate",
                "other_error_rate",
            ):
                model_values = [
                    raw[(model, pushback)][metric] for model in models
                ]
                point = float(
                    np.mean([np.mean(values) for values in model_values])
                )
                samples = np.empty(bootstrap_iterations, dtype=float)
                for iteration in range(bootstrap_iterations):
                    samples[iteration] = float(
                        np.mean(
                            [
                                np.mean(
                                    rng.choice(
                                        values,
                                        size=len(values),
                                        replace=True,
                                    )
                                )
                                for values in model_values
                            ]
                        )
                    )
                low, high = np.quantile(samples, [0.025, 0.975])
                row[metric] = point
                row[f"{metric}_ci_low"] = float(low)
                row[f"{metric}_ci_high"] = float(high)
            macro.append(row)
    return per_model + macro


def preference_split_audit(root: Path) -> dict[str, Any]:
    directory = root / "artifacts/m4/preferences"
    train = [
        json.loads(line)
        for line in (directory / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (directory / "validation.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    def question_ids(rows: list[dict[str, Any]]) -> set[str]:
        return {str(row["id"]).rsplit("|", 1)[0] for row in rows}

    train_questions = question_ids(train)
    validation_questions = question_ids(validation)
    overlap = train_questions & validation_questions
    return {
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_questions": len(train_questions),
        "validation_questions": len(validation_questions),
        "overlapping_questions": len(overlap),
        "validation_question_overlap_rate": (
            len(overlap) / len(validation_questions)
        ),
        "split_unit": "question-pushback pair",
        "independent_question_split": False,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/reviewer_reanalysis"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    mitigation = mitigation_effects(
        root,
        args.results_dir,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    strata = fixed_baseline_strata(
        root,
        args.results_dir,
        args.bootstrap,
        args.permutations,
        args.seed + 1_000,
    )
    decomposition = construct_decomposition(
        root,
        args.results_dir,
        args.bootstrap,
        args.seed + 2_000,
    )
    split_audit = preference_split_audit(root)

    write_csv(mitigation, output / "clustered_mitigation_effects.csv")
    write_csv(strata, output / "fixed_baseline_strata.csv")
    write_csv(decomposition, output / "construct_decomposition.csv")
    (output / "m4_split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "bootstrap_iterations": args.bootstrap,
                "permutation_iterations": args.permutations,
                "seed": args.seed,
                "independent_unit": "benchmark question",
                "multiplicity": "Holm across rows within each output table",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote reviewer-response analyses to {output}")


if __name__ == "__main__":
    main()
