from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from sycophancy.data import read_jsonl


def candidates(paths: list[str]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in paths:
        rows = read_jsonl(path)
        first = {row["record_id"]: row for row in rows if int(row["round"]) == 1}
        for row in rows:
            if (
                int(row["round"]) == 2
                and row.get("setting") == "robustness"
                and row.get("is_target_flip")
            ):
                groups[(Path(path).parent.name, row["pushback_type"])].append(
                    {
                        "run": Path(path).parent.name,
                        "record_id": row["record_id"],
                        "dataset": row["dataset"],
                        "pushback_type": row["pushback_type"],
                        "turn1_response": first[row["record_id"]]["response"],
                        "pushback": row["pushback"],
                        "turn2_response": row["response"],
                        "gold": row["gold"],
                        "target": row["target"],
                        "annotation": "",
                    }
                )
    return groups


def stratified_sample(groups: dict[tuple[str, str], list[dict]], n: int) -> list[dict]:
    rng = random.Random(42)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(groups)
    while len(selected) < n and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())
    return selected


def write(rows: list[dict], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [
        {
            key: re.sub(r"[ \t]+(?=\n|$)", "", value)
            if isinstance(value, str)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(cleaned[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--mitigation", nargs="+", required=True)
    parser.add_argument("--output-dir", default="analysis/qualitative")
    args = parser.parse_args()
    output = Path(args.output_dir)
    baseline = stratified_sample(candidates(args.baseline), 100)
    mitigation = stratified_sample(candidates(args.mitigation), 30)
    write(baseline, str(output / "baseline_failures_100.csv"))
    write(mitigation, str(output / "mitigation_failures_30.csv"))
    print(f"Wrote {len(baseline)} baseline and {len(mitigation)} mitigation cases")


if __name__ == "__main__":
    main()
