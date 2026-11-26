from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.reviewer_reanalysis import (
    cluster_signflip_p,
    clustered_interval,
    holm_adjust,
)
from sycophancy.data import read_jsonl


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(bool(row[field])) for row in rows]))


def paired_question_effects(
    rows: list[dict[str, Any]],
    left: Callable[[dict[str, Any]], bool],
    right: Callable[[dict[str, Any]], bool],
    outcome: str,
) -> list[float]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["record_id"]].append(row)
    effects = []
    for question_rows in by_question.values():
        left_rows = [row for row in question_rows if left(row)]
        right_rows = [row for row in question_rows if right(row)]
        if left_rows and right_rows:
            effects.append(
                mean(left_rows, outcome) - mean(right_rows, outcome)
            )
    return effects


def summarize_effect(
    name: str,
    values: list[float],
    bootstrap: int,
    permutations: int,
    seed: int,
    **dimensions: Any,
) -> dict[str, Any]:
    low, high = clustered_interval(values, bootstrap, seed)
    return {
        "effect": name,
        **dimensions,
        "n_questions": len(values),
        "estimate": float(np.mean(values)),
        "ci_low": low,
        "ci_high": high,
        "cluster_signflip_p": cluster_signflip_p(
            values, permutations, seed + 10_000
        ),
    }


def _analyze_effects_for_initial(
    rows: list[dict[str, Any]],
    bootstrap: int,
    permutations: int,
    seed: int,
    initial_correct: bool | str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    truth = paired_question_effects(
        rows,
        lambda row: bool(row["feedback_correct"]),
        lambda row: not bool(row["feedback_correct"]),
        "feedback_adopted",
    )
    output.append(
        summarize_effect(
            "truth_discrimination_in_adoption",
            truth,
            bootstrap,
            permutations,
            seed,
            initial_correct=initial_correct,
            feedback_correct="contrast",
            authority="all",
            rationale="all",
        )
    )
    for authority in (False, True):
        for rationale in (False, True):
            subset = [
                row
                for row in rows
                if bool(row["authority"]) is authority
                and bool(row["rationale"]) is rationale
            ]
            values = paired_question_effects(
                subset,
                lambda row: bool(row["feedback_correct"]),
                lambda row: not bool(row["feedback_correct"]),
                "feedback_adopted",
            )
            output.append(
                summarize_effect(
                    "truth_discrimination_in_adoption",
                    values,
                    bootstrap,
                    permutations,
                    seed + 50 + int(authority) * 10 + int(rationale),
                    initial_correct=initial_correct,
                    feedback_correct="contrast",
                    authority=authority,
                    rationale=rationale,
                )
            )
    for feedback_correct in (False, True):
        for rationale in (False, True):
            subset = [
                row
                for row in rows
                if bool(row["feedback_correct"]) is feedback_correct
                and bool(row["rationale"]) is rationale
            ]
            values = paired_question_effects(
                subset,
                lambda row: bool(row["authority"]),
                lambda row: not bool(row["authority"]),
                "feedback_adopted",
            )
            output.append(
                summarize_effect(
                    "authority_effect_on_adoption",
                    values,
                    bootstrap,
                    permutations,
                    seed + 100 + int(feedback_correct) * 10 + int(rationale),
                    initial_correct=initial_correct,
                    feedback_correct=feedback_correct,
                    authority="contrast",
                    rationale=rationale,
                )
            )
    for feedback_correct in (False, True):
        for authority in (False, True):
            subset = [
                row
                for row in rows
                if bool(row["feedback_correct"]) is feedback_correct
                and bool(row["authority"]) is authority
            ]
            values = paired_question_effects(
                subset,
                lambda row: bool(row["rationale"]),
                lambda row: not bool(row["rationale"]),
                "feedback_adopted",
            )
            output.append(
                summarize_effect(
                    "rationale_effect_on_adoption",
                    values,
                    bootstrap,
                    permutations,
                    seed + 200 + int(feedback_correct) * 10 + int(authority),
                    initial_correct=initial_correct,
                    feedback_correct=feedback_correct,
                    authority=authority,
                    rationale="contrast",
                )
            )
    truth_by_authority = {}
    for authority in (False, True):
        subset = [
            row for row in rows if bool(row["authority"]) is authority
        ]
        truth_by_authority[authority] = paired_question_effects(
            subset,
            lambda row: bool(row["feedback_correct"]),
            lambda row: not bool(row["feedback_correct"]),
            "feedback_adopted",
        )
    interaction = [
        authority - neutral
        for authority, neutral in zip(
            truth_by_authority[True],
            truth_by_authority[False],
            strict=True,
        )
    ]
    output.append(
        summarize_effect(
            "authority_by_truth_interaction",
            interaction,
            bootstrap,
            permutations,
            seed + 300,
            initial_correct=initial_correct,
            feedback_correct="contrast",
            authority="contrast",
            rationale="all",
        )
    )
    return output


def analyze_effects(
    rows: list[dict[str, Any]],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = _analyze_effects_for_initial(
        rows, bootstrap, permutations, seed, "all"
    )
    for index, initial_correct in enumerate((False, True), start=1):
        output.extend(
            _analyze_effects_for_initial(
                [
                    row
                    for row in rows
                    if bool(row["initial_correct"]) is initial_correct
                ],
                bootstrap,
                permutations,
                seed + index * 1_000,
                initial_correct,
            )
        )
    adjusted = holm_adjust(
        [float(row["cluster_signflip_p"]) for row in output]
    )
    for row, value in zip(output, adjusted, strict=True):
        row["holm_p"] = value
    return output


def cell_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["initial_correct"],
            row["feedback_correct"],
            row["authority"],
            row["rationale"],
        )
        groups[key].append(row)
    return [
        {
            "initial_correct": key[0],
            "feedback_correct": key[1],
            "authority": key[2],
            "rationale": key[3],
            "n": len(values),
            "valid_answer_rate": float(
                np.mean(
                    [
                        row["extraction_status"] == "ANSWER"
                        for row in values
                    ]
                )
            ),
            "final_accuracy": mean(values, "is_correct"),
            "feedback_adoption_rate": mean(values, "feedback_adopted"),
            "stay_rate": mean(values, "stayed_with_initial"),
            "change_rate": mean(values, "changed_answer"),
        }
        for key, values in sorted(groups.items())
    ]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    expected = len({row["record_id"] for row in rows}) * 16
    if len(rows) != expected:
        raise ValueError(f"incomplete factorial result: {len(rows)} != {expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = cell_summary(rows)
    effects = analyze_effects(
        rows, args.bootstrap, args.permutations, args.seed
    )
    write_csv(cells, args.output_dir / "cells.csv")
    write_csv(effects, args.output_dir / "effects.csv")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "questions": len({row["record_id"] for row in rows}),
                "input_sha256": hashlib.sha256(
                    args.input.read_bytes()
                ).hexdigest(),
                "extraction_status_counts": dict(
                    Counter(row["extraction_status"] for row in rows)
                ),
                "bootstrap_iterations": args.bootstrap,
                "permutation_iterations": args.permutations,
                "independent_unit": "benchmark question",
                "multiplicity": "Holm across reported factorial contrasts",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
