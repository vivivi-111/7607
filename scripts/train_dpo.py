from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = DPOConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=Dataset.from_list(load_rows(args.train)),
        eval_dataset=Dataset.from_list(load_rows(args.validation)),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    adapter_dir = output / "best_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        "train": train_result.metrics,
        "validation": trainer.evaluate(),
        "config": vars(args),
    }
    (output / "training_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"Saved best adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
