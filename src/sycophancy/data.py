from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SEED = 42


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _stable_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n >= len(rows):
        return list(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in sorted(indices[:n])]


def _mmlu(limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _load_dataset("cais/mmlu", "all", split="test")]
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    subjects = sorted(by_subject)
    base, remainder = divmod(min(limit, len(rows)), len(subjects))
    output: list[dict[str, Any]] = []
    for i, subject in enumerate(subjects):
        quota = min(len(by_subject[subject]), base + (i < remainder))
        sampled = _stable_sample(by_subject[subject], quota, seed + i)
        for row in sampled:
            choices = list(row["choices"])
            gold_index = int(row["answer"])
            output.append(
                {
                    "id": f"mmlu:{subject}:{row['question']}",
                    "dataset": "mmlu",
                    "subject": subject,
                    "question": row["question"],
                    "choices": choices,
                    "gold_index": gold_index,
                    "gold_answer": choices[gold_index],
                }
            )
    return output


def _gsm8k(limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _load_dataset("openai/gsm8k", "main", split="test")]
    sampled = _stable_sample(rows, min(limit, len(rows)), seed)
    output = []
    for row in sampled:
        match = re.search(r"####\s*([^\n]+)", row["answer"])
        if not match:
            raise ValueError("GSM8K answer lacks a #### final answer")
        gold = match.group(1).strip().replace(",", "")
        output.append(
            {
                "id": f"gsm8k:{row['question']}",
                "dataset": "gsm8k",
                "subject": "math",
                "question": row["question"],
                "choices": None,
                "gold_index": None,
                "gold_answer": gold,
                "solution": row["answer"],
            }
        )
    return output


def shuffle_truthfulqa_row(row: dict[str, Any], seed: int, row_id: str) -> dict[str, Any]:
    choices = list(row["mc1_targets"]["choices"])
    labels = list(row["mc1_targets"]["labels"])
    pairs = list(zip(choices, labels))
    random.Random(f"{seed}:{row_id}").shuffle(pairs)
    shuffled_choices = [pair[0] for pair in pairs]
    shuffled_labels = [int(pair[1]) for pair in pairs]
    if sum(shuffled_labels) != 1:
        raise ValueError(f"TruthfulQA row {row_id!r} has {sum(shuffled_labels)} correct choices")
    gold_index = shuffled_labels.index(1)
    return {
        "id": f"truthfulqa:{row_id}",
        "dataset": "truthfulqa",
        "subject": row.get("category", "unknown"),
        "question": row["question"],
        "choices": shuffled_choices,
        "gold_index": gold_index,
        "gold_answer": shuffled_choices[gold_index],
    }


def _truthfulqa(seed: int) -> list[dict[str, Any]]:
    rows = _load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    return [
        shuffle_truthfulqa_row(dict(row), seed, str(i))
        for i, row in enumerate(rows)
    ]


def split_records(records: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    train_indices = set(indices[: round(0.7 * len(indices))])
    return [
        {**row, "split": "train" if i in train_indices else "test"}
        for i, row in enumerate(records)
    ]


def build_datasets(limit_per_dataset: int | None = None, seed: int = SEED) -> list[dict[str, Any]]:
    limit = limit_per_dataset if limit_per_dataset is not None else 500
    if limit < 1:
        raise ValueError("limit_per_dataset must be positive")
    truthful = _truthfulqa(seed)
    if limit_per_dataset is not None:
        truthful = _stable_sample(truthful, min(limit, len(truthful)), seed)
    records = _mmlu(limit, seed) + _gsm8k(limit, seed) + truthful
    return split_records(records, seed)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
