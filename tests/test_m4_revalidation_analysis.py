from analysis.m4_revalidation_analysis import aggregate_estimates


def test_aggregate_estimates_reports_seed_uncertainty() -> None:
    rows = [
        {"dataset": "all", "paired_difference": -0.01},
        {"dataset": "all", "paired_difference": 0.00},
        {"dataset": "all", "paired_difference": 0.01},
        {"dataset": "mmlu", "paired_difference": 0.02},
        {"dataset": "mmlu", "paired_difference": 0.04},
        {"dataset": "mmlu", "paired_difference": 0.06},
    ]
    result = aggregate_estimates(rows, ("dataset",), "paired_difference")
    by_dataset = {row["dataset"]: row for row in result}

    assert by_dataset["all"]["n_seeds"] == 3
    assert by_dataset["all"]["mean"] == 0.0
    assert by_dataset["all"]["sample_sd"] == 0.01
    assert by_dataset["mmlu"]["min"] == 0.02
    assert by_dataset["mmlu"]["max"] == 0.06


def test_aggregate_estimates_handles_one_seed() -> None:
    result = aggregate_estimates(
        [{"effect": "truth", "estimate": 0.25}],
        ("effect",),
        "estimate",
    )
    assert result[0]["n_seeds"] == 1
    assert result[0]["sample_sd"] == 0.0
