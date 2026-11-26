from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)
    print(f"Merged adapter into {args.output}")


if __name__ == "__main__":
    main()
