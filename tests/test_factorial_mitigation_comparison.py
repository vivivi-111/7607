from analysis.factorial_mitigation_comparison import (
    question_adoption_components,
)


def test_question_adoption_components_uses_paired_truth_contrast() -> None:
    rows = [
        {
            "record_id": "q1",
            "feedback_correct": False,
            "feedback_adopted": True,
        },
        {
            "record_id": "q1",
            "feedback_correct": False,
            "feedback_adopted": False,
        },
        {
            "record_id": "q1",
            "feedback_correct": True,
            "feedback_adopted": True,
        },
        {
            "record_id": "q1",
            "feedback_correct": True,
            "feedback_adopted": True,
        },
    ]
    result = question_adoption_components(rows)["q1"]
    assert result["incorrect_feedback_adoption"] == 0.5
    assert result["correct_feedback_adoption"] == 1.0
    assert result["truth_discrimination"] == 0.5
