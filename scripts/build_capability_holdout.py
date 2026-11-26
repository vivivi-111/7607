from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from sycophancy.data import read_jsonl, write_jsonl


def stable_id(dataset: str, question: str) -> str:
    digest = hashlib.sha256(question.encode()).hexdigest()[:16]
    return f"capability:{dataset}:{digest}"


def sample_without_overlap(
    rows: list[dict[str, Any]],
    existing_questions: set[str],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if str(row["question"]) not in existing_questions
    ]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < limit:
        raise ValueError(f"only {len(candidates)} non-overlapping rows available")
    return candidates[:limit]


def gsm_answer(value: str) -> str:
    match = re.search(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"cannot parse GSM8K answer: {value!r}")
    return match.group(1).replace("$", "").replace(",", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    existing = {str(row["question"]) for row in read_jsonl(args.benchmark)}
    mmlu = sample_without_overlap(
        list(load_dataset("cais/mmlu", "all", split="test")),
        existing,
        args.per_dataset,
        args.seed,
    )
    gsm8k = sample_without_overlap(
        list(load_dataset("openai/gsm8k", "main", split="test")),
        existing,
        args.per_dataset,
        args.seed + 1,
    )
    output = [
        {
            "id": stable_id("mmlu", row["question"]),
            "dataset": "mmlu",
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "gold_index": int(row["answer"]),
            "gold_answer": row["choices"][int(row["answer"])],
            "split": "capability_holdout",
        }
        for row in mmlu
    ]
    output.extend(
        {
            "id": stable_id("gsm8k", row["question"]),
            "dataset": "gsm8k",
            "subject": "grade_school_math",
            "question": row["question"],
            "choices": None,
            "gold_answer": gsm_answer(row["answer"]),
            "split": "capability_holdout",
        }
        for row in gsm8k
    )
    write_jsonl(output, args.output)
    print(f"Wrote {len(output)} non-overlapping capability questions")


if __name__ == "__main__":
    main()
