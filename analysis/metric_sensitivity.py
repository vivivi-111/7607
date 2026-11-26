from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-contains", default="Qwen2.5-7B")
    args = parser.parse_args()
    rows = [
        row
        for row in read_csv(args.aggregate)
        if row["dataset"] == "all"
        and row["pushback_type"] == "all"
        and args.model_contains.casefold() in row["model_id"].casefold()
    ]
    output = []
    for row in rows:
        flip_rate = float(row["flip_rate"])
        correction_rate = float(row["correction_rate"])
        for prevalence_index in range(11):
            initially_correct = prevalence_index / 10
            output.append(
                {
                    "model_id": row["model_id"],
                    "mitigation": row["mitigation"],
                    "initially_correct_prevalence": initially_correct,
                    "FR": flip_rate,
                    "CR": correction_rate,
                    "expected_final_accuracy": (
                        initially_correct * (1 - flip_rate)
                        + (1 - initially_correct) * correction_rate
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, args.output)


if __name__ == "__main__":
    main()
