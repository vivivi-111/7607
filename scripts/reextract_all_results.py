"""Re-extract every reported run without mutating the original result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analysis.reviewer_reanalysis import RUNS
from scripts.reextract_results import reextract_rows
from sycophancy.data import read_jsonl
from sycophancy.metrics import _calculate, metrics_to_csv


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def aggregate(paths: list[Path], output_path: Path) -> None:
    output: list[dict[str, Any]] = []
    for path in paths:
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
                        "source": str(path),
                        "model_id": model_id,
                        "mitigation": mitigation,
                        "dataset": dataset,
                        "pushback_type": pushback,
                        **_calculate(group, first),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/processed/benchmark.jsonl"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/results_reextracted"),
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("analysis/reextracted"),
    )
    parser.add_argument("--bootstrap", type=int, default=1_000)
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = (root / args.output_root).resolve()
    analysis_output = (root / args.analysis_output).resolve()
    records = {
        row["id"]: row for row in read_jsonl(root / args.benchmark)
    }
    run_paths = sorted(
        {
            Path(path)
            for methods in RUNS.values()
            for path in methods.values()
        }
    )
    full_paths: list[Path] = []
    consistent_paths: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for relative in run_paths:
        source = root / relative
        rows, summary = reextract_rows(read_jsonl(source), records)
        run_name = relative.parent.name
        run_output = output_root / run_name
        full_path = run_output / "dialogues.jsonl"
        consistent_path = run_output / "dialogues_design_consistent.jsonl"
        write_jsonl(rows, full_path)
        write_jsonl(
            [
                row
                for row in rows
                if int(row["round"]) == 1
                or bool(row["assignment_consistent"])
            ],
            consistent_path,
        )
        metrics_to_csv(
            full_path,
            analysis_output / f"{run_name}_all.csv",
            bootstrap_iterations=args.bootstrap,
        )
        metrics_to_csv(
            consistent_path,
            analysis_output / f"{run_name}_design_consistent.csv",
            bootstrap_iterations=args.bootstrap,
        )
        full_paths.append(full_path)
        consistent_paths.append(consistent_path)
        summaries.append(
            {
                "run": run_name,
                "source": str(relative),
                **summary,
            }
        )
    aggregate(full_paths, analysis_output / "aggregate_all.csv")
    aggregate(
        consistent_paths,
        analysis_output / "aggregate_design_consistent.csv",
    )
    totals = {
        key: sum(int(row[key]) for row in summaries)
        for key in (
            "rows",
            "changed_extractions",
            "changed_correctness",
            "assignment_inconsistent",
        )
    }
    (analysis_output / "reextraction_summary.json").write_text(
        json.dumps(
            {"runs": summaries, "totals": totals},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
