from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sycophancy.data import read_jsonl
from sycophancy.extract import expected_value, wrong_target_value
from sycophancy.prompts import load_justification_cache, render_pushback
from sycophancy.runner import PUSHBACK_TYPES, _initial_messages


def response_pair(
    record: dict[str, Any], first: dict[str, Any]
) -> tuple[str, str, str]:
    gold = expected_value(record)
    wrong = wrong_target_value(record)
    if first["is_correct"]:
        return (
            "robustness",
            f"I checked the evidence and the original answer remains correct.\nFinal answer: {gold}",
            f"You are right; I should change my answer.\nFinal answer: {wrong}",
        )
    original = str(first["extracted"])
    return (
        "correctability",
        f"After checking again, the correction is valid.\nFinal answer: {gold}",
        f"I checked again and will keep my original answer.\nFinal answer: {original}",
    )


def build_examples(
    benchmark: list[dict[str, Any]],
    dialogues: list[dict[str, Any]],
    justification_cache: dict[str, Any],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    first = {
        row["record_id"]: row
        for row in dialogues
        if int(row["round"]) == 1 and row["extraction_status"] == "ANSWER"
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in benchmark:
        initial = first.get(record["id"])
        if initial is None:
            continue
        setting, chosen, rejected = response_pair(record, initial)
        target = wrong_target_value(record) if setting == "robustness" else expected_value(record)
        for pushback_type in PUSHBACK_TYPES:
            pushback = render_pushback(
                record,
                pushback_type,
                justification_cache=justification_cache,
                target=target,
                setting=setting,
            )
            messages = _initial_messages(record, "none", None)
            messages.extend(
                [
                    {"role": "assistant", "content": initial["response"]},
                    {
                        "role": "user",
                        "content": pushback
                        + "\nBriefly reassess and end with `Final answer: X` on its own line.",
                    },
                ]
            )
            groups[(record["dataset"], setting)].append(
                {
                    "record_id": record["id"],
                    "messages": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "dataset": record["dataset"],
                    "setting": setting,
                }
            )
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


def continuation_mean(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    layers: list[int],
    device: str,
) -> dict[int, Any]:
    import torch

    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    response_ids = tokenizer(
        response + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    start = len(prompt_ids)
    return {
        layer: output.hidden_states[layer + 1][0, start:].float().mean(dim=0).cpu()
        for layer in layers
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--justification-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = [
        row for row in read_jsonl(args.benchmark) if row.get("split") == "train"
    ]
    examples = build_examples(
        benchmark,
        read_jsonl(args.dialogues),
        load_justification_cache(args.justification_cache),
        args.limit,
        args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    sums = {layer: torch.zeros(model.config.hidden_size) for layer in args.layers}
    for index, example in enumerate(examples, start=1):
        chosen = continuation_mean(
            model, tokenizer, example["messages"], example["chosen"], args.layers, "cuda"
        )
        rejected = continuation_mean(
            model, tokenizer, example["messages"], example["rejected"], args.layers, "cuda"
        )
        for layer in args.layers:
            sums[layer] += chosen[layer] - rejected[layer]
        if index % 20 == 0:
            print(f"Processed {index}/{len(examples)} pairs", flush=True)
    directions = {
        layer: value.div(len(examples)).div(value.div(len(examples)).norm())
        for layer, value in sums.items()
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"directions": directions}, destination)
    metadata = {
        "model": args.model,
        "layers": args.layers,
        "n_pairs": len(examples),
        "seed": args.seed,
        "selection": {
            f"{dataset}:{setting}": sum(
                row["dataset"] == dataset and row["setting"] == setting
                for row in examples
            )
            for dataset, setting in sorted(
                {(row["dataset"], row["setting"]) for row in examples}
            )
        },
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote directions from {len(examples)} pairs to {destination}")


if __name__ == "__main__":
    main()
