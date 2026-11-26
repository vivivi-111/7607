from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sycophancy.data import read_jsonl
from sycophancy.metrics import _calculate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = []
    for path in args.dialogues:
        rows = read_jsonl(path)
        first = {
            (row["record_id"], row["model_id"], row["mitigation"]): row
            for row in rows
            if int(row["round"]) == 1
        }
        second = [row for row in rows if int(row["round"]) == 2]
        model_id = next(iter(first))[1]
        mitigation = next(iter(first))[2]
        datasets = sorted({row["dataset"] for row in second})
        pushbacks = sorted({row["pushback_type"] for row in second})
        for dataset in [*datasets, "all"]:
            for pushback in [*pushbacks, "all"]:
                group = [
                    row
                    for row in second
                    if (dataset == "all" or row["dataset"] == dataset)
                    and (
                        pushback == "all"
                        or row["pushback_type"] == pushback
                    )
                ]
                output.append(
                    {
                        "source": path,
                        "model_id": model_id,
                        "mitigation": mitigation,
                        "dataset": dataset,
                        "pushback_type": pushback,
                        **_calculate(group, first),
                    }
                )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} aggregate rows to {destination}")


if __name__ == "__main__":
    main()
