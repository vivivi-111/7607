"""Run the controlled factorial feedback experiment."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

from sycophancy.data import read_jsonl
from sycophancy.factorial import run_factorial
from sycophancy.models import TransformersBackend, VLLMBackend
from sycophancy.prompts import load_justification_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--backend", choices=("vllm", "hf"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--mitigation", choices=("none", "M1", "M2", "M4"), default="none")
    parser.add_argument("--justification-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"another factorial runner is already writing to {output_dir}"
        ) from error

    records = read_jsonl(args.input)
    if args.split != "all":
        records = [
            record for record in records if record.get("split") == args.split
        ]
    if args.limit is not None:
        records = records[: args.limit]
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )
    else:
        backend = TransformersBackend(
            args.model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
    output = run_factorial(
        records=records,
        backend=backend,
        output_dir=args.output_dir,
        model_id=args.model_id or args.model,
        mitigation=args.mitigation,
        justification_cache=load_justification_cache(
            args.justification_cache
        ),
        generation_batch_size=args.generation_batch_size,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
