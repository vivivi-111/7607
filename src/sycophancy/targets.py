from __future__ import annotations

import hashlib
import random
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Protocol


class ReferenceScorer(Protocol):
    def __call__(self, record: dict[str, Any], candidate_indices: list[int]) -> int: ...


def _seed_for(record_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choice_wrong_target(
    record: dict[str, Any],
    seed: int = 42,
    scorer: ReferenceScorer | None = None,
) -> dict[str, Any]:
    choices = record.get("choices")
    gold_index = record.get("gold_index")
    if not choices or gold_index is None:
        raise ValueError("choice_wrong_target requires choices and gold_index")
    candidates = [i for i in range(len(choices)) if i != int(gold_index)]
    if not candidates:
        raise ValueError("no incorrect choice is available")
    if scorer is not None:
        selected = int(scorer(record, candidates))
        if selected not in candidates:
            raise ValueError("reference scorer selected a non-candidate")
    else:
        selected = random.Random(_seed_for(record["id"], seed)).choice(candidates)
    return {
        "wrong_target": choices[selected],
        "wrong_target_index": selected,
        "wrong_target_label": chr(65 + selected),
        "wrong_target_strategy": "reference_scorer" if scorer else "deterministic_random",
    }


def gsm8k_wrong_target(record: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    raw = str(record["gold_answer"]).strip().replace(",", "")
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        raise ValueError(f"cannot perturb non-numeric GSM8K answer: {raw!r}")
    try:
        gold = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"invalid GSM8K number: {raw!r}") from exc
    rng = random.Random(_seed_for(record["id"], seed))
    if gold == gold.to_integral_value():
        magnitude = max(Decimal(1), abs(gold) // 10)
        delta = magnitude * (-1 if rng.randrange(2) else 1)
        target = gold + delta
        rendered = str(int(target))
    else:
        scale = Decimal(1).scaleb(gold.as_tuple().exponent)
        delta = scale * rng.choice([-2, -1, 1, 2])
        target = gold + delta
        rendered = format(target, "f")
    if target == gold:
        raise AssertionError("numeric perturbation must differ from gold")
    return {
        "wrong_target": rendered,
        "wrong_target_index": None,
        "wrong_target_label": rendered,
        "wrong_target_strategy": "deterministic_numeric_perturbation",
    }


def add_wrong_target(
    record: dict[str, Any],
    seed: int = 42,
    scorer: Callable[[dict[str, Any], list[int]], int] | None = None,
) -> dict[str, Any]:
    target = (
        gsm8k_wrong_target(record, seed)
        if record.get("dataset") == "gsm8k"
        else choice_wrong_target(record, seed, scorer)
    )
    return {**record, **target}
