from __future__ import annotations

from sycophancy.factorial import (
    alternative_wrong_value,
    build_messages,
    factorial_conditions,
    factorial_eligible,
    feedback_text,
)


RECORD = {
    "id": "mmlu:test",
    "dataset": "mmlu",
    "subject": "logic",
    "question": "Which option is correct?",
    "choices": ["feedback wrong", "correct", "initial wrong", "other"],
    "gold_index": 1,
    "wrong_target_index": 0,
}


def test_factorial_design_contains_all_sixteen_cells() -> None:
    conditions = factorial_conditions()
    assert len(conditions) == 16
    assert len({row["condition_id"] for row in conditions}) == 16
    for field in (
        "initial_correct",
        "feedback_correct",
        "authority",
        "rationale",
    ):
        assert {row[field] for row in conditions} == {False, True}


def test_initial_and_feedback_truth_are_crossed_independently() -> None:
    cache = {"mmlu:test": {"correct": "B is supported.", "wrong": "A seems plausible."}}
    condition = {
        "condition_id": "test",
        "initial_correct": False,
        "feedback_correct": True,
        "authority": True,
        "rationale": True,
    }
    messages, initial, target = build_messages(
        RECORD, condition, cache, "none"
    )
    assert initial == "C"
    assert target == "B"
    assert "professor of logic" in messages[-1]["content"]
    assert "B is supported" in messages[-1]["content"]
    assert messages[-2]["content"].endswith("Final answer: C")


def test_initial_and_feedback_wrong_answers_are_distinct() -> None:
    assert alternative_wrong_value(RECORD) == "C"
    condition = {
        "condition_id": "test",
        "initial_correct": False,
        "feedback_correct": False,
        "authority": False,
        "rationale": False,
    }
    _, initial, target = build_messages(RECORD, condition, {}, "none")
    assert initial == "C"
    assert target == "A"


def test_numeric_initial_wrong_is_distinct() -> None:
    record = {
        "id": "gsm8k:test",
        "dataset": "gsm8k",
        "question": "Compute it.",
        "gold_answer": "11",
        "wrong_target": "10",
    }
    assert alternative_wrong_value(record) == "12"


def test_binary_question_is_ineligible_for_crossed_wrong_targets() -> None:
    assert not factorial_eligible({**RECORD, "choices": ["yes", "no"]})
    assert factorial_eligible(RECORD)


def test_authority_and_rationale_are_separate_manipulations() -> None:
    plain = feedback_text(RECORD, "B", False, False, None)
    authority = feedback_text(RECORD, "B", True, False, None)
    reasoned = feedback_text(RECORD, "B", False, True, "Because evidence.")
    assert "professor" not in plain
    assert "professor" in authority
    assert "Because evidence" in reasoned
