from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analysis.build_annotation_packets import LABELS, VISIBLE_SOURCE_COLUMNS
from analysis.score_annotations import (
    index_and_validate,
    read_annotations,
    score_rows,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_summary(
    rows: list[dict[str, str]], dimension: str
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for value in sorted({row[dimension] for row in rows}):
        group = [row for row in rows if row[dimension] == value]
        counts = Counter(row["final_label"] for row in group)
        output[value] = {
            "n": len(group),
            "label_counts": dict(sorted(counts.items())),
            "label_proportions": {
                label: count / len(group)
                for label, count in sorted(counts.items())
            },
        }
    return output


def finalize_rows(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    adjudication_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    indexed_a = index_and_validate(rows_a, "annotator A")
    indexed_b = index_and_validate(rows_b, "annotator B")
    if set(indexed_a) != set(indexed_b):
        raise ValueError("annotator sample_id sets differ")
    agreement_summary, disagreements = score_rows(rows_a, rows_b)
    disagreement_ids = {row["sample_id"] for row in disagreements}

    adjudication: dict[str, dict[str, str]] = {}
    for row in adjudication_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("adjudication contains an empty sample_id")
        if sample_id in adjudication:
            raise ValueError(f"duplicate adjudication sample_id {sample_id}")
        final_label = str(row.get("final_label") or "").strip()
        if final_label not in LABELS:
            raise ValueError(
                f"invalid or empty final_label {final_label!r} for {sample_id}"
            )
        adjudication[sample_id] = {
            **row,
            "sample_id": sample_id,
            "final_label": final_label,
        }
    if set(adjudication) != disagreement_ids:
        missing = sorted(disagreement_ids - set(adjudication))
        extra = sorted(set(adjudication) - disagreement_ids)
        raise ValueError(
            f"adjudication IDs must equal disagreement IDs; missing={missing}, extra={extra}"
        )

    mapping = {
        row["sample_id"]: row
        for row in mapping_rows
        if row.get("set") == "formal"
    }
    if not set(indexed_a).issubset(mapping):
        raise ValueError(
            f"private mapping is missing IDs: {sorted(set(indexed_a) - set(mapping))}"
        )

    final: list[dict[str, str]] = []
    for sample_id in sorted(indexed_a):
        row_a = indexed_a[sample_id]
        row_b = indexed_b[sample_id]
        agreed = row_a["label"] == row_b["label"]
        decision = adjudication.get(sample_id, {})
        final.append(
            {
                "sample_id": sample_id,
                **{
                    column: row_a.get(column, "")
                    for column in VISIBLE_SOURCE_COLUMNS
                },
                "run": mapping[sample_id].get("run", ""),
                "model": mapping[sample_id].get("model", ""),
                "record_id": mapping[sample_id].get("record_id", ""),
                "label_A": row_a["label"],
                "label_B": row_b["label"],
                "final_label": (
                    row_a["label"] if agreed else decision["final_label"]
                ),
                "resolution": "agreement" if agreed else "adjudicated",
                "notes_A": row_a.get("notes", ""),
                "notes_B": row_b.get("notes", ""),
                "adjudication_notes": str(
                    decision.get("adjudication_notes") or ""
                ).strip(),
            }
        )

    counts = Counter(row["final_label"] for row in final)
    summary: dict[str, object] = {
        **agreement_summary,
        "adjudicated": len(disagreement_ids),
        "final_label_counts": dict(sorted(counts.items())),
        "final_label_proportions": {
            label: count / len(final) for label, count in sorted(counts.items())
        },
        "by_model": grouped_summary(final, "model"),
        "by_pushback_type": grouped_summary(final, "pushback_type"),
        "by_dataset": grouped_summary(final, "dataset"),
    }
    return final, summary


def write_outputs(
    rows: list[dict[str, str]], summary: dict[str, object], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "final_annotations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge agreements with adjudicated annotation disagreements"
    )
    parser.add_argument("annotator_a", type=Path)
    parser.add_argument("annotator_b", type=Path)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/annotation/final")
    )
    args = parser.parse_args()
    rows, summary = finalize_rows(
        read_annotations(args.annotator_a),
        read_annotations(args.annotator_b),
        read_annotations(args.adjudication),
        read_csv(args.mapping),
    )
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
