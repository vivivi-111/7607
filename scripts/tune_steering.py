from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sycophancy.data import read_jsonl
from sycophancy.metrics import _calculate
from sycophancy.prompts import load_justification_cache
from sycophancy.runner import run_dialogue
from sycophancy.steering import SteeringBackend


def stratified_records(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


def aggregate(path: Path) -> dict[str, float | int]:
    rows = read_jsonl(path)
    first = {
        (row["record_id"], row["model_id"], row["mitigation"]): row
        for row in rows
        if int(row["round"]) == 1
    }
    second = [row for row in rows if int(row["round"]) == 2]
    return _calculate(second, first)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--justification-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1, 2, 4])
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()

    records = stratified_records(
        [
            row
            for row in read_jsonl(args.benchmark)
            if row.get("split") == "train"
        ],
        args.limit,
        args.seed,
    )
    backend = SteeringBackend(
        args.model,
        args.directions,
        args.layers[0],
        0,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_justification_cache(args.justification_cache)
    candidates = [(args.layers[0], 0.0), *[
        (layer, alpha) for layer in args.layers for alpha in args.alphas
    ]]
    results = []
    for layer, alpha in candidates:
        backend.set_steering(layer, alpha)
        name = f"layer{layer}_alpha{alpha:g}"
        path = run_dialogue(
            records,
            backend,
            output_dir / name,
            args.model,
            mitigation="M3",
            justification_cache=cache,
        )
        metrics = aggregate(path)
        results.append({"layer": layer, "alpha": alpha, **metrics})
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    baseline_accuracy = float(results[0]["first_accuracy"])
    eligible = [
        row
        for row in results[1:]
        if float(row["first_accuracy"]) >= baseline_accuracy - 0.02
    ]
    pool = eligible or results[1:]
    best = max(pool, key=lambda row: (float(row["D"]), float(row["second_accuracy"])))
    selection = {
        "selection_rule": "max D with first_accuracy no more than 0.02 below alpha=0",
        "baseline_first_accuracy": baseline_accuracy,
        "eligible_count": len(eligible),
        "best": best,
        "candidates": results,
        "record_ids": [row["id"] for row in records],
        "seed": args.seed,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Selected layer={best['layer']} alpha={best['alpha']}")


if __name__ == "__main__":
    main()
