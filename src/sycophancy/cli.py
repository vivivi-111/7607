from __future__ import annotations

import argparse
import os

from .data import build_datasets, read_jsonl, write_jsonl
from .metrics import metrics_to_csv
from .models import OpenAICompatibleBackend, TransformersBackend, VLLMBackend
from .prompts import load_justification_cache
from .runner import run_dialogue
from .targets import add_wrong_target


def _prepare(args: argparse.Namespace) -> None:
    records = [
        add_wrong_target(row, seed=args.seed)
        for row in build_datasets(args.limit_per_dataset, args.seed)
    ]
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


def _run(args: argparse.Namespace) -> None:
    if args.backend == "vllm":
        backend = VLLMBackend(
            args.model,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
        )
    elif args.backend == "hf":
        backend = TransformersBackend(
            args.model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        backend = OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env),
            cache_dir=args.cache_dir,
            concurrency=args.concurrency,
            request_options={
                "max_tokens": args.max_tokens,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            },
        )
    records = read_jsonl(args.input)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if not records:
        raise ValueError(f"no records found for split={args.split!r}")
    output = run_dialogue(
        records,
        backend,
        args.output_dir,
        args.model_id or args.model,
        mitigation=args.mitigation,
        justification_cache=load_justification_cache(args.justification_cache),
        skip_missing_p5=args.skip_missing_p5,
    )
    print(f"Results: {output}")


def _metrics(args: argparse.Namespace) -> None:
    results = metrics_to_csv(args.input, args.output, args.bootstrap)
    print(f"Wrote {len(results)} metric rows to {args.output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sycophancy")
    subparsers = root.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", default="data/processed/benchmark.jsonl")
    prepare.add_argument("--limit-per-dataset", type=int)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--input", default="data/processed/benchmark.jsonl")
    run.add_argument("--split", choices=("train", "test", "all"), default="test")
    run.add_argument("--output-dir", default="results/run")
    run.add_argument("--backend", choices=("vllm", "hf", "api"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--model-id",
        help="reporting identifier when the loaded model is a fine-tuned checkpoint",
    )
    run.add_argument("--base-url", default="http://localhost:8000/v1")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--cache-dir", default=".cache/api")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--device", default="cuda")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument(
        "--mitigation", choices=("none", "M1", "M2", "M3", "M4"), default="none"
    )
    run.add_argument("--justification-cache")
    run.add_argument("--skip-missing-p5", action="store_true")
    run.set_defaults(func=_run)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--input", default="results/run/dialogues.jsonl")
    metrics.add_argument("--output", default="analysis/metrics.csv")
    metrics.add_argument("--bootstrap", type=int, default=1000)
    metrics.set_defaults(func=_metrics)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
