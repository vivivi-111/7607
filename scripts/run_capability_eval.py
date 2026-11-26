from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sycophancy.data import read_jsonl
from sycophancy.extract import expected_value, extract_answer
from sycophancy.models import VLLMBackend
from sycophancy.runner import _initial_messages


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    completed = set()
    if args.output.exists():
        completed = {
            row["record_id"] for row in read_jsonl(args.output)
        }
    pending = [row for row in records if row["id"] not in completed]
    backend = VLLMBackend(
        args.model,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    for start in range(0, len(pending), args.generation_batch_size):
        batch = pending[start : start + args.generation_batch_size]
        responses = backend.generate_batch(
            [_initial_messages(record, "none", None) for record in batch]
        )
        rows = []
        for record, response in zip(batch, responses, strict=True):
            extraction = extract_answer(response, record.get("choices"))
            gold = expected_value(record)
            rows.append(
                {
                    "record_id": record["id"],
                    "dataset": record["dataset"],
                    "model_id": args.model_id,
                    "response": response,
                    "extracted": extraction.value,
                    "extraction_status": extraction.status,
                    "gold": gold,
                    "is_correct": extraction.value == gold,
                }
            )
        append_rows(args.output, rows)
    print(f"Capability results: {args.output}")


if __name__ == "__main__":
    main()
