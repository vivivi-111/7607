from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sycophancy.data import read_jsonl
from sycophancy.runner import _initial_messages


def fallback_response(
    gold: str, rejected_answer: str, desired: bool
) -> str:
    if desired:
        return (
            "I checked the evidence rather than following the claim automatically. "
            f"The supported answer is {gold}.\nFinal answer: {gold}"
        )
    return (
        "I accepted the unsupported conclusion without resolving the evidence. "
        f"\nFinal answer: {rejected_answer}"
    )


def choose_pair(
    candidates: list[dict[str, Any]],
    gold: str,
    rejected_answer: str,
) -> tuple[str, str, bool, bool]:
    correct = [row for row in candidates if row.get("is_correct")]
    wrong = [
        row
        for row in candidates
        if not row.get("is_correct")
        and str(row.get("extracted")) == str(rejected_answer)
    ]
    if not wrong:
        wrong = [row for row in candidates if not row.get("is_correct")]
    chosen_fallback = not correct
    rejected_fallback = not wrong
    chosen = (
        correct[0]["response"]
        if correct
        else fallback_response(gold, rejected_answer, True)
    )
    rejected = (
        wrong[0]["response"]
        if wrong
        else fallback_response(gold, rejected_answer, False)
    )
    return chosen, rejected, chosen_fallback, rejected_fallback


def grouped_train_validation_split(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_id = str(row["id"]).rsplit("|", 1)[0]
        by_question[question_id].append(row)
    question_ids = sorted(by_question)
    random.Random(seed).shuffle(question_ids)
    target = round(len(rows) * validation_fraction)
    validation_questions: set[str] = set()
    validation_count = 0
    for question_id in question_ids:
        size = len(by_question[question_id])
        if validation_count >= target:
            break
        validation_questions.add(question_id)
        validation_count += size
    train = [
        row
        for row in rows
        if str(row["id"]).rsplit("|", 1)[0] not in validation_questions
    ]
    validation = [
        row
        for row in rows
        if str(row["id"]).rsplit("|", 1)[0] in validation_questions
    ]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=2400)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fallback-policy",
        choices=("template", "drop"),
        default="template",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    benchmark_rows = read_jsonl(args.benchmark)
    records = {
        row["id"]: row
        for row in benchmark_rows
        if row.get("split") == "train"
    }
    test_question_ids = {
        row["id"] for row in benchmark_rows if row.get("split") == "test"
    }
    baseline_rows = read_jsonl(args.baseline)
    baseline_first = {
        row["record_id"]: row for row in baseline_rows if int(row["round"]) == 1
    }
    baseline_second = {
        (row["record_id"], row["pushback_type"]): row
        for row in baseline_rows
        if int(row["round"]) == 2
        and bool(row.get("assignment_consistent", True))
    }
    all_candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in args.candidates:
        for row in read_jsonl(path):
            if int(row["round"]) == 2 and bool(
                row.get("assignment_consistent", True)
            ):
                all_candidates[(row["record_id"], row["pushback_type"])].append(row)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, second in baseline_second.items():
        record_id, pushback_type = key
        record = records.get(record_id)
        first = baseline_first.get(record_id)
        if record is None or first is None:
            continue
        rejected_answer = (
            str(second["target"])
            if second["setting"] == "robustness"
            else str(first["extracted"])
        )
        chosen, rejected, chosen_fallback, rejected_fallback = choose_pair(
            all_candidates[key], str(second["gold"]), rejected_answer
        )
        if args.fallback_policy == "drop" and (
            chosen_fallback or rejected_fallback
        ):
            continue
        messages = _initial_messages(record, "none", None)
        messages.extend(
            [
                {"role": "assistant", "content": first["response"]},
                {
                    "role": "user",
                    "content": second["pushback"]
                    + "\nBriefly reassess and end with `Final answer: X` on its own line.",
                },
            ]
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        groups[(record["dataset"], second["setting"])].append(
            {
                "id": f"{record_id}|{pushback_type}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "dataset": record["dataset"],
                "setting": second["setting"],
                "chosen_fallback": chosen_fallback,
                "rejected_fallback": rejected_fallback,
            }
        )
    rng = random.Random(args.seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    settings = sorted({setting for _, setting in groups})
    per_setting = min(
        args.limit // len(settings),
        *[
            sum(
                len(values)
                for key, values in groups.items()
                if key[1] == setting
            )
            for setting in settings
        ],
    )
    for setting in settings:
        setting_groups = {
            key: values for key, values in groups.items() if key[1] == setting
        }
        keys = sorted(setting_groups)
        chosen_rows: list[dict[str, Any]] = []
        while len(chosen_rows) < per_setting and any(setting_groups.values()):
            for key in keys:
                if setting_groups[key] and len(chosen_rows) < per_setting:
                    chosen_rows.append(setting_groups[key].pop())
        selected.extend(chosen_rows)
    rng.shuffle(selected)
    train, validation = grouped_train_validation_split(
        selected, args.validation_fraction, args.seed
    )
    train_questions = {
        str(row["id"]).rsplit("|", 1)[0] for row in train
    }
    validation_questions = {
        str(row["id"]).rsplit("|", 1)[0] for row in validation
    }
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "train": len(train),
        "validation": len(validation),
        "settings": {
            setting: sum(row["setting"] == setting for row in selected)
            for setting in settings
        },
        "chosen_fallback": sum(row["chosen_fallback"] for row in selected),
        "rejected_fallback": sum(row["rejected_fallback"] for row in selected),
        "split_unit": "question",
        "train_questions": len(train_questions),
        "validation_questions": len(validation_questions),
        "question_overlap": len(train_questions & validation_questions),
        "test_question_overlap": len(
            (train_questions | validation_questions) & test_question_ids
        ),
        "fallback_policy": args.fallback_policy,
        "seed": args.seed,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
