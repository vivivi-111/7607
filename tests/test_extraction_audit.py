from __future__ import annotations

import pytest

from analysis.build_extraction_audit_packets import (
    packet_rows,
    stratified_sample,
)
from analysis.score_extraction_audit import (
    accuracy_rows,
    canonical,
    reextract_mapping,
    wilson_interval,
)


def test_stratified_audit_sample_has_round_and_status_quotas() -> None:
    rows = []
    for round_number in (1, 2):
        for status, count in (("ANSWER", 30), ("NONE", 10), ("HEDGE", 5)):
            for index in range(count):
                rows.append(
                    {
                        "checkpoint_key": f"{round_number}-{status}-{index}",
                        "round": round_number,
                        "extraction_status": status,
                        "dataset": ("mmlu", "gsm8k", "truthfulqa")[index % 3],
                        "run": f"run-{index % 2}",
                    }
                )
    selected = stratified_sample(rows, per_round=20, seed=7)
    assert len(selected) == 40
    for round_number in (1, 2):
        group = [row for row in selected if row["round"] == round_number]
        assert sum(row["extraction_status"] == "ANSWER" for row in group) == 16
        assert sum(row["extraction_status"] == "NONE" for row in group) == 3
        assert sum(row["extraction_status"] == "HEDGE" for row in group) == 1


def test_packet_rows_hide_automatic_extraction() -> None:
    source = [
        {
            "dataset": "mmlu",
            "round": 2,
            "question": "Question?",
            "choices_list": ["one", "two"],
            "turn1_response": "first",
            "pushback": "really?",
            "response": "Final answer: B",
            "run": "secret-run",
            "record_id": "secret-id",
            "model_id": "secret-model",
            "mitigation": "none",
            "pushback_type": "P1",
            "extraction_status": "ANSWER",
            "extracted": "B",
            "checkpoint_key": "secret-key",
        }
    ]
    visible, private = packet_rows(source)
    assert visible[0]["sample_id"] == "E001"
    assert "auto_status" not in visible[0]
    assert "model_id" not in visible[0]
    assert private[0]["auto_status"] == "ANSWER"
    assert private[0]["checkpoint_key"] == "secret-key"


def test_canonicalizes_choice_and_numeric_answers() -> None:
    assert canonical("answer", "(b)", "mmlu") == ("ANSWER", "B")
    assert canonical("ANSWER", "$1,200.0", "gsm8k") == ("ANSWER", "1200")
    assert canonical("hedge", "ignored", "truthfulqa") == ("HEDGE", "")
    with pytest.raises(ValueError, match="requires"):
        canonical("ANSWER", "", "mmlu")


def test_accuracy_and_wilson_threshold() -> None:
    mapping = {
        f"E{index:03d}": {
            "sample_id": f"E{index:03d}",
            "dataset": "mmlu",
            "auto_status": "ANSWER",
            "auto_value": "A" if index <= 196 else "B",
        }
        for index in range(1, 201)
    }
    gold = {
        sample_id: ("ANSWER", "A") for sample_id in mapping
    }
    rows, summary = accuracy_rows(gold, mapping)
    assert len(rows) == 200
    assert summary["accuracy"] == 0.98
    assert summary["passes_98_percent"] is True
    low, high = wilson_interval(196, 200)
    assert 0 < low < 0.98 < high < 1


def test_reextract_mapping_uses_current_extractor() -> None:
    mapping = {
        "E001": {
            "sample_id": "E001",
            "dataset": "mmlu",
            "auto_status": "NONE",
            "auto_value": "",
        }
    }
    rows = {
        "E001": {
            "choices": "A. one\nB. two",
            "response": "Final answer: B",
        }
    }
    refreshed = reextract_mapping(rows, mapping)
    assert refreshed["E001"]["auto_status"] == "ANSWER"
    assert refreshed["E001"]["auto_value"] == "B"
