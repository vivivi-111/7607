from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Iterable

LABELS = (
    "direct_concession",
    "fabricated_reasoning",
    "hedging",
    "partial_concession",
    "other",
)
VISIBLE_SOURCE_COLUMNS = (
    "dataset",
    "pushback_type",
    "turn1_response",
    "pushback",
    "turn2_response",
    "gold",
    "target",
)
PACKET_COLUMNS = ("sample_id", *VISIBLE_SOURCE_COLUMNS, "label", "notes")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def model_from_row(row: dict[str, str]) -> str:
    if row.get("model"):
        return row["model"]
    return re.sub(r"_(?:none|baseline|M[12])$", "", row.get("run", ""))


def make_packet_rows(
    rows: Iterable[dict[str, str]], prefix: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    packets: list[dict[str, str]] = []
    mappings: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        sample_id = f"{prefix}{index:03d}"
        packets.append(
            {
                "sample_id": sample_id,
                **{column: row.get(column, "") for column in VISIBLE_SOURCE_COLUMNS},
                "label": "",
                "notes": "",
            }
        )
        mappings.append(
            {
                "sample_id": sample_id,
                "set": "formal" if prefix == "F" else "practice",
                "run": row.get("run", ""),
                "model": model_from_row(row),
                "record_id": row.get("record_id", ""),
            }
        )
    return packets, mappings


def select_practice_rows(
    candidates: Iterable[dict[str, str]],
    formal_rows: Iterable[dict[str, str]],
    count: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    formal_ids = {row["record_id"] for row in formal_rows}
    eligible = [row for row in candidates if row.get("record_id") not in formal_ids]
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} non-overlapping practice rows; need {count}"
        )
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def ordered(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(rows: list[dict[str, str]], path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError as exc:
        raise SystemExit(
            'Excel output requires openpyxl; install with pip install -e ".[annotation]"'
        ) from exc

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
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
        "sample_id": 14,
        "dataset": 16,
        "pushback_type": 15,
        "turn1_response": 55,
        "pushback": 35,
        "turn2_response": 55,
        "gold": 12,
        "target": 12,
        "label": 28,
        "notes": 45,
    }
    for index, name in enumerate(rows[0], 1):
        sheet.column_dimensions[get_column_letter(index)].width = widths.get(name, 20)

    label_column = list(rows[0]).index("label") + 1
    validation = DataValidation(
        type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True
    )
    validation.error = "请选择下拉列表中的标签"
    validation.errorTitle = "无效标签"
    validation.showErrorMessage = True
    sheet.add_data_validation(validation)
    validation.add(
        f"{get_column_letter(label_column)}2:"
        f"{get_column_letter(label_column)}{len(rows) + 1}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def build_packets(
    formal_source: Path, practice_source: Path, output_dir: Path, seed: int = 42
) -> None:
    formal_source_rows = read_csv(formal_source)
    if len(formal_source_rows) != 100:
        raise ValueError(
            f"Formal source must contain exactly 100 rows, got {len(formal_source_rows)}"
        )
    practice_source_rows = read_csv(practice_source)
    practice_selected = select_practice_rows(
        practice_source_rows, formal_source_rows, seed=seed
    )
    formal, formal_mapping = make_packet_rows(formal_source_rows, "F")
    practice, practice_mapping = make_packet_rows(practice_selected, "P")

    annotator_a = ordered(formal, seed)
    annotator_b = ordered(formal, seed + 1)
    for name, rows in (
        ("annotator_A", annotator_a),
        ("annotator_B", annotator_b),
        ("practice_20", practice),
    ):
        write_csv(rows, output_dir / f"{name}.csv")
        write_xlsx(rows, output_dir / f"{name}.xlsx")
    write_csv(formal_mapping + practice_mapping, output_dir / "private_mapping.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded annotation packets")
    parser.add_argument(
        "--formal-source",
        type=Path,
        default=Path("analysis/qualitative/baseline_failures_100.csv"),
    )
    parser.add_argument(
        "--practice-source",
        type=Path,
        default=Path("analysis/qualitative/mitigation_failures_30.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/annotation")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_packets(
        args.formal_source, args.practice_source, args.output_dir, seed=args.seed
    )
    print(f"Wrote annotation packets to {args.output_dir}")


if __name__ == "__main__":
    main()
