from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from sycophancy.data import read_jsonl

STATUSES = ("ANSWER", "NONE", "HEDGE")
VISIBLE_COLUMNS = (
    "sample_id",
    "dataset",
    "round",
    "question",
    "choices",
    "turn1_response",
    "pushback",
    "response",
    "human_status",
    "human_value",
    "notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_paths(aggregate: Path) -> list[Path]:
    paths: list[Path] = []
    for row in read_csv(aggregate):
        path = Path(row["source"])
        if path not in paths:
            paths.append(path)
    return paths


def format_choices(choices: list[Any] | None) -> str:
    return "\n".join(
        f"{chr(65 + index)}. {choice}"
        for index, choice in enumerate(choices or [])
    )


def excluded_ids(
    mapping_paths: list[Path], qualitative_paths: list[Path]
) -> tuple[set[str], set[tuple[str, str, str]]]:
    record_ids = set()
    for mapping_path in mapping_paths:
        record_ids.update(
            row["record_id"]
            for row in read_csv(mapping_path)
            if row.get("record_id")
        )
    second_keys: set[tuple[str, str, str]] = set()
    for path in qualitative_paths:
        for row in read_csv(path):
            record_ids.add(row["record_id"])
            second_keys.add(
                (row["run"], row["record_id"], row["pushback_type"])
            )
    return record_ids, second_keys


def candidate_rows(
    paths: list[Path],
    benchmark: dict[str, dict[str, Any]],
    excluded_records: set[str],
    excluded_second: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        run = path.parent.name
        first_responses: dict[str, str] = {}
        for row in read_jsonl(path):
            round_number = int(row["round"])
            if row["record_id"] in excluded_records:
                continue
            if round_number == 1:
                first_responses[row["record_id"]] = row["response"]
            elif (
                run,
                row["record_id"],
                row.get("pushback_type", ""),
            ) in excluded_second:
                continue
            record = benchmark.get(row["record_id"])
            if record is None:
                continue
            candidates.append(
                {
                    **row,
                    "run": run,
                    "round": round_number,
                    "question": record["question"],
                    "choices_list": record.get("choices") or [],
                    "turn1_response": (
                        ""
                        if round_number == 1
                        else first_responses.get(row["record_id"], "")
                    ),
                }
            )
    return candidates


def stratified_sample(
    rows: list[dict[str, Any]],
    per_round: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    status_weights = {"ANSWER": 0.8, "NONE": 0.15, "HEDGE": 0.05}
    for round_number in (1, 2):
        round_rows = [
            row for row in rows if int(row["round"]) == round_number
        ]
        quotas = {
            status: round(per_round * weight)
            for status, weight in status_weights.items()
        }
        quotas["ANSWER"] += per_round - sum(quotas.values())
        for status in STATUSES:
            groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
                list
            )
            for row in round_rows:
                if row["extraction_status"] == status:
                    groups[(row["run"], row["dataset"])].append(row)
            for values in groups.values():
                rng.shuffle(values)
            keys = sorted(groups)
            status_rows: list[dict[str, Any]] = []
            while len(status_rows) < quotas[status] and any(groups.values()):
                for key in keys:
                    if groups[key] and len(status_rows) < quotas[status]:
                        status_rows.append(groups[key].pop())
            if len(status_rows) != quotas[status]:
                raise ValueError(
                    f"not enough round={round_number} status={status} rows"
                )
            selected.extend(status_rows)
    rng.shuffle(selected)
    return selected


def packet_rows(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    visible: list[dict[str, str]] = []
    private: list[dict[str, str]] = []
    for index, row in enumerate(selected, 1):
        sample_id = f"E{index:03d}"
        visible.append(
            {
                "sample_id": sample_id,
                "dataset": row["dataset"],
                "round": str(row["round"]),
                "question": row["question"],
                "choices": format_choices(row["choices_list"]),
                "turn1_response": row["turn1_response"],
                "pushback": row.get("pushback", ""),
                "response": row["response"],
                "human_status": "",
                "human_value": "",
                "notes": "",
            }
        )
        private.append(
            {
                "sample_id": sample_id,
                "run": row["run"],
                "record_id": row["record_id"],
                "model_id": row["model_id"],
                "mitigation": row["mitigation"],
                "round": str(row["round"]),
                "pushback_type": row.get("pushback_type", ""),
                "dataset": row["dataset"],
                "choices_json": json.dumps(
                    row["choices_list"], ensure_ascii=False
                ),
                "auto_status": row["extraction_status"],
                "auto_value": "" if row.get("extracted") is None else str(
                    row["extracted"]
                ),
                "checkpoint_key": row["checkpoint_key"],
            }
        )
    return visible, private


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(rows: list[dict[str, str]], path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "extraction_audit"
    sheet.append(list(rows[0]))
    for row in rows:
        sheet.append(list(row.values()))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    widths = {
        "sample_id": 12,
        "dataset": 14,
        "round": 8,
        "question": 45,
        "choices": 45,
        "turn1_response": 55,
        "pushback": 40,
        "response": 60,
        "human_status": 18,
        "human_value": 18,
        "notes": 36,
    }
    for index, column in enumerate(rows[0], 1):
        sheet.column_dimensions[get_column_letter(index)].width = widths[column]
    status_column = list(rows[0]).index("human_status") + 1
    validation = DataValidation(
        type="list", formula1=f'"{",".join(STATUSES)}"', allow_blank=False
    )
    validation.showErrorMessage = True
    sheet.add_data_validation(validation)
    validation.add(
        f"{get_column_letter(status_column)}2:"
        f"{get_column_letter(status_column)}{len(rows) + 1}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def build_packets(
    aggregate: Path,
    benchmark_path: Path,
    mapping_path: Path,
    qualitative_paths: list[Path],
    output_dir: Path,
    seed: int = 42,
    exclusion_mappings: list[Path] | None = None,
) -> dict[str, Any]:
    benchmark = {
        row["id"]: row
        for row in read_jsonl(benchmark_path)
        if row.get("split") == "test"
    }
    excluded_records, excluded_second = excluded_ids(
        [mapping_path, *(exclusion_mappings or [])], qualitative_paths
    )
    candidates = candidate_rows(
        source_paths(aggregate),
        benchmark,
        excluded_records,
        excluded_second,
    )
    selected = stratified_sample(candidates, seed=seed)
    visible, private = packet_rows(selected)
    for name, rows in (
        ("annotator_A", visible),
        ("annotator_B", visible),
    ):
        ordered = list(rows)
        random.Random(seed if name.endswith("A") else seed + 1).shuffle(ordered)
        write_csv(ordered, output_dir / f"{name}.csv")
        write_xlsx(ordered, output_dir / f"{name}.xlsx")
    write_csv(private, output_dir / "private_mapping.csv")
    manifest = {
        "audit_status": "locked_before_annotation",
        "n": len(visible),
        "seed": seed,
        "source_sha256": {
            "aggregate": hashlib.sha256(aggregate.read_bytes()).hexdigest(),
            "benchmark": hashlib.sha256(
                benchmark_path.read_bytes()
            ).hexdigest(),
            "private_mapping": hashlib.sha256(
                (output_dir / "private_mapping.csv").read_bytes()
            ).hexdigest(),
            "annotator_A": hashlib.sha256(
                (output_dir / "annotator_A.xlsx").read_bytes()
            ).hexdigest(),
            "annotator_B": hashlib.sha256(
                (output_dir / "annotator_B.xlsx").read_bytes()
            ).hexdigest(),
            "extractor": hashlib.sha256(
                (
                    Path(__file__).parents[1]
                    / "src/sycophancy/extract.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "excluded_mapping_files": [
            str(path)
            for path in [mapping_path, *(exclusion_mappings or [])]
        ],
        "round_counts": {
            str(round_number): sum(
                row["round"] == str(round_number) for row in visible
            )
            for round_number in (1, 2)
        },
        "auto_status_counts": {
            status: sum(row["auto_status"] == status for row in private)
            for status in STATUSES
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate", type=Path, default=Path("analysis/aggregate_summary.csv")
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/processed/benchmark.jsonl")
    )
    parser.add_argument(
        "--annotation-mapping",
        type=Path,
        default=Path("analysis/annotation/private_mapping.csv"),
    )
    parser.add_argument(
        "--qualitative",
        type=Path,
        nargs="+",
        default=[
            Path("analysis/qualitative/baseline_failures_100.csv"),
            Path("analysis/qualitative/mitigation_failures_30.csv"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/extraction_audit"),
    )
    parser.add_argument(
        "--exclude-mapping",
        type=Path,
        action="append",
        default=[],
        help="exclude every record present in an earlier audit mapping",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_packets(
        args.aggregate,
        args.benchmark,
        args.annotation_mapping,
        args.qualitative,
        args.output_dir,
        args.seed,
        args.exclude_mapping,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
