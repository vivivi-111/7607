from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analysis.build_annotation_packets import LABELS, VISIBLE_SOURCE_COLUMNS


def read_annotations(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise SystemExit(
                'Excel input requires openpyxl; install with pip install -e ".[annotation]"'
            ) from exc
        sheet = load_workbook(path, read_only=True, data_only=True).active
        values = sheet.iter_rows(values_only=True)
        headers = [str(value) if value is not None else "" for value in next(values)]
        return [
            {
                header: "" if value is None else str(value).strip()
                for header, value in zip(headers, row, strict=False)
            }
            for row in values
            if any(value is not None and str(value).strip() for value in row)
        ]
    raise ValueError(f"Unsupported annotation format: {path.suffix}")


def index_and_validate(
    rows: list[dict[str, str]], annotator: str
) -> dict[str, dict[str, str]]:
    required = {"sample_id", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{annotator}: required columns are sample_id and label")
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        sample_id = row.get("sample_id", "").strip()
        label = row.get("label", "").strip()
        if not sample_id:
            raise ValueError(f"{annotator}: empty sample_id at row {row_number}")
        if sample_id in indexed:
            raise ValueError(f"{annotator}: duplicate sample_id {sample_id}")
        if label not in LABELS:
            raise ValueError(
                f"{annotator}: invalid or empty label {label!r} for {sample_id}"
            )
        indexed[sample_id] = {**row, "sample_id": sample_id, "label": label}
    return indexed


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Kappa requires equally sized, non-empty label sequences")
    count = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / count
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    expected = (
        sum(counts_a[label] * counts_b[label] for label in categories) / count**2
    )
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def score_rows(
    rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]
) -> tuple[dict[str, object], list[dict[str, str]]]:
    indexed_a = index_and_validate(rows_a, "annotator A")
    indexed_b = index_and_validate(rows_b, "annotator B")
    ids_a, ids_b = set(indexed_a), set(indexed_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)
        only_b = sorted(ids_b - ids_a)
        raise ValueError(
            f"sample_id sets differ; only A: {only_a}; only B: {only_b}"
        )

    sample_ids = sorted(ids_a)
    labels_a = [indexed_a[sample_id]["label"] for sample_id in sample_ids]
    labels_b = [indexed_b[sample_id]["label"] for sample_id in sample_ids]
    agreements = sum(a == b for a, b in zip(labels_a, labels_b, strict=True))
    disagreements = [
        {
            "sample_id": sample_id,
            **{
                column: indexed_a[sample_id].get(column, "")
                for column in VISIBLE_SOURCE_COLUMNS
            },
            "label_A": indexed_a[sample_id]["label"],
            "label_B": indexed_b[sample_id]["label"],
            "notes_A": indexed_a[sample_id].get("notes", ""),
            "notes_B": indexed_b[sample_id].get("notes", ""),
        }
        for sample_id in sample_ids
        if indexed_a[sample_id]["label"] != indexed_b[sample_id]["label"]
    ]
    summary: dict[str, object] = {
        "n": len(sample_ids),
        "agreements": agreements,
        "disagreements": len(disagreements),
        "raw_agreement": agreements / len(sample_ids),
        "cohen_kappa": cohen_kappa(labels_a, labels_b),
        "label_counts_A": dict(sorted(Counter(labels_a).items())),
        "label_counts_B": dict(sorted(Counter(labels_b).items())),
    }
    return summary, disagreements


def write_scores(
    summary: dict[str, object],
    disagreements: list[dict[str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = (
        "sample_id",
        *VISIBLE_SOURCE_COLUMNS,
        "label_A",
        "label_B",
        "notes_A",
        "notes_B",
    )
    with (output_dir / "disagreements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(disagreements)
    adjudication_columns = (*columns, "final_label", "adjudication_notes")
    with (output_dir / "adjudication.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=adjudication_columns)
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "final_label": "",
                "adjudication_notes": "",
            }
            for row in disagreements
        )
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "adjudication"
    sheet.append(list(adjudication_columns))
    for row in disagreements:
        sheet.append(
            [
                row.get(column, "")
                if column not in {"final_label", "adjudication_notes"}
                else ""
                for column in adjudication_columns
            ]
        )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for index, column in enumerate(adjudication_columns, 1):
        width = (
            55
            if column in {"turn1_response", "turn2_response"}
            else 35
            if column in {"pushback", "adjudication_notes"}
            else 20
        )
        sheet.column_dimensions[get_column_letter(index)].width = width
    final_label_column = adjudication_columns.index("final_label") + 1
    validation = DataValidation(
        type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True
    )
    validation.showErrorMessage = True
    sheet.add_data_validation(validation)
    validation.add(
        f"{get_column_letter(final_label_column)}2:"
        f"{get_column_letter(final_label_column)}{len(disagreements) + 1}"
    )
    workbook.save(output_dir / "adjudication.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score two returned annotation files")
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/annotation/scores")
    )
    args = parser.parse_args()
    summary, disagreements = score_rows(
        read_annotations(args.annotator_a), read_annotations(args.annotator_b)
    )
    write_scores(summary, disagreements, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
