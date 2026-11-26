from analysis.reviewer_reanalysis import (
    clustered_comparison,
    holm_adjust,
)


def row(correct: bool) -> dict[str, bool]:
    return {"is_correct": correct}


def test_holm_adjust_is_monotone_in_p_value_order() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    assert adjusted == [0.04, 0.003, 0.04]


def test_clustered_comparison_uses_question_as_unit() -> None:
    baseline = {
        ("q1", "P1"): row(False),
        ("q1", "P2"): row(False),
        ("q2", "P1"): row(True),
        ("q2", "P2"): row(True),
    }
    mitigation = {
        ("q1", "P1"): row(True),
        ("q1", "P2"): row(True),
        ("q2", "P1"): row(True),
        ("q2", "P2"): row(False),
    }
    result = clustered_comparison(
        baseline,
        mitigation,
        bootstrap_iterations=200,
        permutation_iterations=200,
        seed=7,
    )
    assert result["n_pairs"] == 4
    assert result["n_questions"] == 2
    assert result["baseline_correct_only"] == 1
    assert result["mitigation_correct_only"] == 2
    assert result["effect"] == 0.25
    assert float(result["ci_low"]) <= 0.25 <= float(result["ci_high"])
