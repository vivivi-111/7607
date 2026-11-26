from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from analysis.build_extraction_audit_packets import STATUSES, VISIBLE_COLUMNS
from analysis.score_annotations import cohen_kappa, read_annotations
from sycophancy.extract import extract_answer, normalize_number


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical(
    status: str, value: str, dataset: str
) -> tuple[str, str]:
    status = status.strip().upper()
    value = value.strip()
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}")
    if status != "ANSWER":
        return status, ""
    if not value:
        raise ValueError("ANSWER requires a human_value")
    if dataset == "gsm8k":
        normalized = normalize_number(value)
        if normalized is None:
            raise ValueError(f"invalid GSM8K numeric answer {value!r}")
        return status, normalized
    label = value.upper().strip("(). ")
    if len(label) != 1 or not label.isalpha():
        raise ValueError(f"invalid multiple-choice answer {value!r}")
    return status, label


def index_rows(
    rows: list[dict[str, str]],
    mapping: dict[str, dict[str, str]],
    annotator: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id or sample_id not in mapping:
            raise ValueError(
                f"{annotator}: invalid sample_id at row {row_number}: {sample_id!r}"
            )
        if sample_id in indexed:
            raise ValueError(f"{annotator}: duplicate sample_id {sample_id}")
        status, value = canonical(
            str(row.get("human_status") or ""),
            str(row.get("human_value") or ""),
            mapping[sample_id]["dataset"],
        )
        indexed[sample_id] = {
            **row,
            "sample_id": sample_id,
            "human_status": status,
            "human_value": value,
        }
    if set(indexed) != set(mapping):
        raise ValueError(
            f"{annotator}: missing IDs {sorted(set(mapping) - set(indexed))}"
        )
    return indexed


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def agreement_summary(
    indexed_a: dict[str, dict[str, str]],
    indexed_b: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    sample_ids = sorted(indexed_a)
    labels_a = [
        f"{indexed_a[sample_id]['human_status']}:{indexed_a[sample_id]['human_value']}"
        for sample_id in sample_ids
    ]
    labels_b = [
        f"{indexed_b[sample_id]['human_status']}:{indexed_b[sample_id]['human_value']}"
        for sample_id in sample_ids
    ]
    disagreements = [
        sample_id
        for sample_id, label_a, label_b in zip(
            sample_ids, labels_a, labels_b, strict=True
        )
        if label_a != label_b
    ]
    agreements = len(sample_ids) - len(disagreements)
    return (
        {
            "n": len(sample_ids),
            "agreements": agreements,
            "disagreements": len(disagreements),
            "raw_agreement": agreements / len(sample_ids),
            "cohen_kappa": cohen_kappa(labels_a, labels_b),
            "status_counts_A": dict(
                sorted(
                    Counter(
                        row["human_status"] for row in indexed_a.values()
                    ).items()
                )
            ),
            "status_counts_B": dict(
                sorted(
                    Counter(
                        row["human_status"] for row in indexed_b.values()
                    ).items()
                )
            ),
        },
        disagreements,
    )


def write_adjudication(
    disagreements: list[str],
    indexed_a: dict[str, dict[str, str]],
    indexed_b: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    columns = (
        *VISIBLE_COLUMNS[:-3],
        "human_status_A",
        "human_value_A",
        "human_status_B",
        "human_value_B",
        "notes_A",
        "notes_B",
        "final_status",
        "final_value",
        "adjudication_notes",
    )
    rows = []
    for sample_id in disagreements:
        row_a = indexed_a[sample_id]
        row_b = indexed_b[sample_id]
        rows.append(
            {
                **{
                    column: row_a.get(column, "")
                    for column in VISIBLE_COLUMNS[:-3]
                },
                "human_status_A": row_a["human_status"],
                "human_value_A": row_a["human_value"],
                "human_status_B": row_b["human_status"],
                "human_value_B": row_b["human_value"],
                "notes_A": row_a.get("notes", ""),
                "notes_B": row_b.get("notes", ""),
                "final_status": "",
                "final_value": "",
                "adjudication_notes": "",
            }
        )
    with (output_dir / "adjudication.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "adjudication"
    sheet.append(list(columns))
    for row in rows:
        sheet.append([row[column] for column in columns])
    sheet.freeze_panes = "A2"
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for index, column in enumerate(columns, 1):
        width = 55 if column in {"response", "turn1_response"} else 24
        sheet.column_dimensions[get_column_letter(index)].width = width
    status_column = columns.index("final_status") + 1
    validation = DataValidation(
        type="list", formula1=f'"{",".join(STATUSES)}"', allow_blank=False
    )
    sheet.add_data_validation(validation)
    validation.add(
        f"{get_column_letter(status_column)}2:"
        f"{get_column_letter(status_column)}{len(rows) + 1}"
    )
    workbook.save(output_dir / "adjudication.xlsx")


def final_gold(
    sample_ids: list[str],
    indexed_a: dict[str, dict[str, str]],
    indexed_b: dict[str, dict[str, str]],
    mapping: dict[str, dict[str, str]],
    adjudication_rows: list[dict[str, str]] | None,
) -> dict[str, tuple[str, str]]:
    gold: dict[str, tuple[str, str]] = {}
    disagreements: set[str] = set()
    for sample_id in sample_ids:
        a = (
            indexed_a[sample_id]["human_status"],
            indexed_a[sample_id]["human_value"],
        )
        b = (
            indexed_b[sample_id]["human_status"],
            indexed_b[sample_id]["human_value"],
        )
        if a == b:
            gold[sample_id] = a
        else:
            disagreements.add(sample_id)
    if not disagreements:
        return gold
    if adjudication_rows is None:
        raise ValueError("adjudication is required to compute extraction accuracy")
    adjudicated: set[str] = set()
    for row in adjudication_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in disagreements or sample_id in adjudicated:
            raise ValueError(f"invalid adjudication sample_id {sample_id!r}")
        gold[sample_id] = canonical(
            str(row.get("final_status") or ""),
            str(row.get("final_value") or ""),
            mapping[sample_id]["dataset"],
        )
        adjudicated.add(sample_id)
    if adjudicated != disagreements:
        raise ValueError(
            f"missing adjudications: {sorted(disagreements - adjudicated)}"
        )
    return gold


def accuracy_rows(
    gold: dict[str, tuple[str, str]],
    mapping: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for sample_id in sorted(gold):
        auto = canonical(
            mapping[sample_id]["auto_status"],
            mapping[sample_id]["auto_value"],
            mapping[sample_id]["dataset"],
        )
        expected = gold[sample_id]
        rows.append(
            {
                **mapping[sample_id],
                "gold_status": expected[0],
                "gold_value": expected[1],
                "is_extraction_correct": str(auto == expected),
            }
        )
    correct = sum(row["is_extraction_correct"] == "True" for row in rows)
    low, high = wilson_interval(correct, len(rows))
    summary = {
        "n": len(rows),
        "correct": correct,
        "errors": len(rows) - correct,
        "accuracy": correct / len(rows),
        "wilson_95_ci_low": low,
        "wilson_95_ci_high": high,
        "passes_98_percent": correct / len(rows) >= 0.98,
    }
    return rows, summary


def reextract_mapping(
    indexed_rows: dict[str, dict[str, str]],
    mapping: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    refreshed: dict[str, dict[str, str]] = {}
    for sample_id, private_row in mapping.items():
        visible_row = indexed_rows[sample_id]
        choices = []
        for index, line in enumerate(
            str(visible_row.get("choices") or "").splitlines()
        ):
            prefix = f"{chr(65 + index)}. "
            choices.append(line[len(prefix):] if line.startswith(prefix) else line)
        extraction = extract_answer(
            str(visible_row.get("response") or ""),
            choices or None,
        )
        refreshed[sample_id] = {
            **private_row,
            "auto_status": extraction.status,
            "auto_value": extraction.value or "",
        }
    return refreshed


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument(
        "--reextract-current",
        action="store_true",
        help="rerun the current extractor on workbook responses before scoring",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/extraction_audit/scores"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        row["sample_id"]: row for row in read_csv(args.mapping)
    }
    indexed_a = index_rows(
        read_annotations(args.annotator_a), mapping, "annotator A"
    )
    indexed_b = index_rows(
        read_annotations(args.annotator_b), mapping, "annotator B"
    )
    if args.reextract_current:
        mapping = reextract_mapping(indexed_a, mapping)
    summary, disagreements = agreement_summary(indexed_a, indexed_b)
    write_adjudication(
        disagreements, indexed_a, indexed_b, args.output_dir
    )
    if args.adjudication:
        gold = final_gold(
            sorted(mapping),
            indexed_a,
            indexed_b,
            mapping,
            read_annotations(args.adjudication),
        )
        rows, accuracy = accuracy_rows(gold, mapping)
        summary["extraction"] = accuracy
        write_csv(rows, args.output_dir / "final_audit.csv")
        errors = [
            row for row in rows if row["is_extraction_correct"] == "False"
        ]
        errors_path = args.output_dir / "extraction_errors.csv"
        if errors:
            write_csv(errors, errors_path)
        elif errors_path.exists():
            errors_path.unlink()
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
