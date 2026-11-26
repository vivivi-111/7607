from scripts.build_dpo_dataset import (
    choose_pair,
    grouped_train_validation_split,
)
from scripts.build_steering_directions import response_pair
from scripts.tune_steering import stratified_records
from sycophancy.cli import parser
from sycophancy.prompts import system_prompt


def test_m3_m4_have_no_system_prompt():
    assert system_prompt("M3") is None
    assert system_prompt("M4") is None


def test_cli_accepts_separate_finetuned_model_id():
    args = parser().parse_args(
        [
            "run",
            "--backend",
            "vllm",
            "--model",
            "merged-checkpoint",
            "--model-id",
            "base-model",
            "--mitigation",
            "M4",
        ]
    )
    assert args.model_id == "base-model"


def test_response_pair_tracks_setting():
    record = {
        "choices": ["a", "b", "c"],
        "gold_index": 1,
        "wrong_target_index": 2,
    }
    robust = response_pair(
        record, {"is_correct": True, "extracted": "B"}
    )
    correctable = response_pair(
        record, {"is_correct": False, "extracted": "A"}
    )
    assert robust[0] == "robustness"
    assert robust[1].endswith("Final answer: B")
    assert robust[2].endswith("Final answer: C")
    assert correctable[0] == "correctability"
    assert correctable[1].endswith("Final answer: B")
    assert correctable[2].endswith("Final answer: A")


def test_choose_pair_prefers_real_target_error():
    candidates = [
        {"is_correct": True, "extracted": "B", "response": "good"},
        {"is_correct": False, "extracted": "D", "response": "other"},
        {"is_correct": False, "extracted": "C", "response": "target"},
    ]
    assert choose_pair(candidates, "B", "C") == (
        "good",
        "target",
        False,
        False,
    )


def test_dpo_split_keeps_question_pushbacks_together():
    rows = [
        {"id": f"question-{question}|P{pushback}"}
        for question in range(10)
        for pushback in range(1, 6)
    ]
    train, validation = grouped_train_validation_split(rows, 0.2, 42)
    train_questions = {row["id"].rsplit("|", 1)[0] for row in train}
    validation_questions = {
        row["id"].rsplit("|", 1)[0] for row in validation
    }
    assert not train_questions & validation_questions
    assert len(validation) == 10


def test_stratified_records_is_deterministic():
    rows = [
        {"id": f"{dataset}-{index}", "dataset": dataset}
        for dataset in ("a", "b", "c")
        for index in range(4)
    ]
    first = stratified_records(rows, 6, 42)
    second = stratified_records(rows, 6, 42)
    assert first == second
    assert {row["dataset"] for row in first} == {"a", "b", "c"}
