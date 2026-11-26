from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import binomtest

from sycophancy.data import read_jsonl


def second_rows(path: str) -> dict[tuple[str, str], dict]:
    return {
        (row["record_id"], row["pushback_type"]): row
        for row in read_jsonl(path)
        if int(row["round"]) == 2
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mitigation", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = second_rows(args.baseline)
    output = []
    for mitigation_path in args.mitigation:
        mitigation = second_rows(mitigation_path)
        common = sorted(set(baseline) & set(mitigation))
        baseline_only = sum(
            bool(baseline[key]["is_correct"])
            and not bool(mitigation[key]["is_correct"])
            for key in common
        )
        mitigation_only = sum(
            bool(mitigation[key]["is_correct"])
            and not bool(baseline[key]["is_correct"])
            for key in common
        )
        discordant = baseline_only + mitigation_only
        p_value = (
            binomtest(
                min(baseline_only, mitigation_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if discordant
            else 1.0
        )
        output.append(
            {
                "mitigation_path": mitigation_path,
                "n_common": len(common),
                "baseline_correct_only": baseline_only,
                "mitigation_correct_only": mitigation_only,
                "second_accuracy_difference": (
                    mitigation_only - baseline_only
                )
                / len(common),
                "mcnemar_exact_p": p_value,
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
    print(f"Wrote {len(output)} comparisons to {destination}")


if __name__ == "__main__":
    main()
