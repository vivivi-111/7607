from __future__ import annotations

import argparse
from pathlib import Path

from sycophancy.data import read_jsonl
from sycophancy.models import VLLMBackend
from sycophancy.prompts import load_justification_cache
from sycophancy.runner import run_dialogue


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run none, M1, and M2 with one shared vLLM model load."
    )
    result.add_argument("--model", required=True)
    result.add_argument("--input", required=True)
    result.add_argument("--output-prefix", required=True)
    result.add_argument("--split", choices=("train", "test", "all"), default="test")
    result.add_argument("--justification-cache")
    result.add_argument("--max-tokens", type=int, default=1024)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    return result


def main() -> None:
    args = parser().parse_args()
    records = read_jsonl(args.input)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if not records:
        raise ValueError(f"no records found for split={args.split!r}")

    backend = VLLMBackend(
        args.model,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
    )
    justification_cache = load_justification_cache(args.justification_cache)
    for mitigation in ("none", "M1", "M2"):
        output_dir = Path(f"{args.output_prefix}_{mitigation}")
        output = run_dialogue(
            records,
            backend,
            output_dir,
            args.model,
            mitigation=mitigation,
            justification_cache=justification_cache,
        )
        print(f"{mitigation}: {output}", flush=True)


if __name__ == "__main__":
    main()
