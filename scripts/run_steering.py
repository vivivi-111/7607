from __future__ import annotations

import argparse
import json

from sycophancy.data import read_jsonl
from sycophancy.prompts import load_justification_cache
from sycophancy.runner import run_dialogue
from sycophancy.steering import SteeringBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--justification-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    args = parser.parse_args()

    selection = json.loads(open(args.selection, encoding="utf-8").read())
    best = selection["best"]
    backend = SteeringBackend(
        args.model,
        args.directions,
        int(best["layer"]),
        float(best["alpha"]),
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    records = read_jsonl(args.benchmark)
    if args.split != "all":
        records = [row for row in records if row.get("split") == args.split]
    output = run_dialogue(
        records,
        backend,
        args.output_dir,
        args.model,
        mitigation="M3",
        justification_cache=load_justification_cache(args.justification_cache),
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
