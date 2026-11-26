from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_runs(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"expected NAME=DIR, received {value!r}")
        output[name] = Path(path)
    return output


def macro_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    dimensions = (
        "effect",
        "initial_correct",
        "feedback_correct",
        "authority",
        "rationale",
    )
    for row in rows:
        groups[tuple(str(row[key]) for key in dimensions)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        estimates = [float(row["estimate"]) for row in values]
        output.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "n_models": len(values),
                "model_macro_mean": mean(estimates),
                "model_min": min(estimates),
                "model_max": max(estimates),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    effects = []
    cells = []
    for name, directory in parse_runs(args.run).items():
        effects.extend(
            {"run": name, **row} for row in read_csv(directory / "effects.csv")
        )
        cells.extend(
            {"run": name, **row} for row in read_csv(directory / "cells.csv")
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(effects, args.output_dir / "effects_all_runs.csv")
    write_csv(cells, args.output_dir / "cells_all_runs.csv")
    write_csv(
        macro_effects(effects),
        args.output_dir / "effects_model_macro_descriptive.csv",
    )


if __name__ == "__main__":
    main()
