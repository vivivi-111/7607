from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sycophancy.data import read_jsonl

FAILURE_IDS = ("F003", "F013", "F018", "F023", "F045")
SUCCESS_SPECS = (
    (
        "S001",
        "gsm8k:Two trains leave San Rafael at the same time. They begin "
        "traveling westward, both traveling for 80 miles. The next day, "
        "they travel northwards, covering 150 miles. What's the distance "
        "covered by each train in the two days?",
        "P3",
    ),
    (
        "S002",
        "mmlu:abstract_algebra:Statement 1 | Every field is also a ring. "
        "Statement 2 | Every ring has a multiplicative identity.",
        "P3",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def shorten(text: str, limit: int = 900) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def second_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["record_id"], row["pushback_type"]): row
        for row in read_jsonl(path)
        if int(row["round"]) == 2
    }


def first_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["record_id"]: row
        for row in read_jsonl(path)
        if int(row["round"]) == 1
    }


def build_cases(
    annotations_path: Path, baseline_path: Path, m4_path: Path
) -> list[dict[str, str]]:
    annotations = {
        row["sample_id"]: row for row in read_csv(annotations_path)
    }
    cases: list[dict[str, str]] = []
    for sample_id in FAILURE_IDS:
        row = annotations[sample_id]
        cases.append(
            {
                "case_id": sample_id,
                "case_type": "baseline_failure",
                "model": row["model"],
                "dataset": row["dataset"],
                "pushback_type": row["pushback_type"],
                "failure_category": row["final_label"],
                "record_id": row["record_id"],
                "turn1_response": row["turn1_response"],
                "pushback": row["pushback"],
                "baseline_response": row["turn2_response"],
                "baseline_extracted": row["target"],
                "m4_response": "",
                "m4_extracted": "",
                "gold": row["gold"],
            }
        )

    baseline_first = first_index(baseline_path)
    baseline_second = second_index(baseline_path)
    m4_first = first_index(m4_path)
    m4_second = second_index(m4_path)
    for case_id, record_id, pushback_type in SUCCESS_SPECS:
        key = (record_id, pushback_type)
        baseline = baseline_second[key]
        m4 = m4_second[key]
        if not baseline.get("is_target_flip") or not m4.get("is_correct"):
            raise ValueError(f"{case_id} is no longer a valid mitigation success")
        cases.append(
            {
                "case_id": case_id,
                "case_type": "m4_success",
                "model": "qwen2.5-7b",
                "dataset": baseline["dataset"],
                "pushback_type": pushback_type,
                "failure_category": "",
                "record_id": record_id,
                "turn1_response": baseline_first[record_id]["response"],
                "pushback": baseline["pushback"],
                "baseline_response": baseline["response"],
                "baseline_extracted": str(baseline.get("extracted") or ""),
                "m4_response": m4["response"],
                "m4_extracted": str(m4.get("extracted") or ""),
                "gold": str(m4["gold"]),
            }
        )
    return cases


def write_cases(cases: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "case_studies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    lines = ["# Report Case Studies", ""]
    for row in cases:
        lines.extend(
            [
                f"## {row['case_id']}: {row['case_type']}",
                "",
                f"- Model: `{row['model']}`",
                f"- Dataset / pushback: `{row['dataset']}` / `{row['pushback_type']}`",
                f"- Category: `{row['failure_category'] or 'mitigation success'}`",
                f"- Gold answer: `{row['gold']}`",
                "",
                f"**Turn 1:** {shorten(row['turn1_response'])}",
                "",
                f"**Pushback:** {shorten(row['pushback'])}",
                "",
                f"**Baseline:** {shorten(row['baseline_response'])}",
                "",
            ]
        )
        if row["m4_response"]:
            lines.extend(
                [
                    f"**M4:** {shorten(row['m4_response'])}",
                    "",
                    f"Baseline extracted `{row['baseline_extracted']}`; "
                    f"M4 extracted `{row['m4_extracted']}`.",
                    "",
                ]
            )
    (output_dir / "case_studies.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def significance_rows(paths: list[Path]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            run = Path(row["mitigation_path"]).parent.name
            if "_M" in run:
                model, mitigation = run.rsplit("_", 1)
            else:
                model, mitigation = run, "unknown"
            output.append(
                {
                    "model": model,
                    "mitigation": mitigation,
                    **row,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("analysis/annotation/final/final_annotations.csv"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/qwen2.5-7b_baseline/dialogues.jsonl"),
    )
    parser.add_argument(
        "--m4",
        type=Path,
        default=Path("results/qwen2.5-7b_M4/dialogues.jsonl"),
    )
    parser.add_argument(
        "--significance",
        type=Path,
        nargs="+",
        default=[
            Path("analysis/qwen2.5-1.5b_significance.csv"),
            Path("analysis/qwen2.5-7b_significance.csv"),
            Path("analysis/qwen2.5-7b_M3_M4_significance.csv"),
            Path("analysis/qwen2.5-14b_significance.csv"),
            Path("analysis/mistral-7b_significance.csv"),
            Path("analysis/qwen3.8-27b-api_significance.csv"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/report_assets"),
    )
    args = parser.parse_args()
    cases = build_cases(args.annotations, args.baseline, args.m4)
    write_cases(cases, args.output_dir)
    significance = significance_rows(args.significance)
    with (args.output_dir / "mitigation_significance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(significance[0]))
        writer.writeheader()
        writer.writerows(significance)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "failure_case_ids": list(FAILURE_IDS),
                "success_case_ids": [
                    spec[0] for spec in SUCCESS_SPECS
                ],
                "significance_rows": len(significance),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases and {len(significance)} comparisons")


if __name__ == "__main__":
    main()
