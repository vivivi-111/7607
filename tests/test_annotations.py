from __future__ import annotations

import csv

import pytest

from analysis.build_annotation_packets import (
    PACKET_COLUMNS,
    make_packet_rows,
    select_practice_rows,
)
from analysis.finalize_annotations import finalize_rows
from analysis.score_annotations import cohen_kappa, score_rows, write_scores


def test_packet_rows_are_blinded_and_have_stable_ids() -> None:
    source = [
        {
            "run": "model-name_M1",
            "record_id": "secret-record",
            "dataset": "mmlu",
            "pushback_type": "P2",
            "turn1_response": "first",
            "pushback": "really?",
            "turn2_response": "second",
            "gold": "A",
            "target": "B",
        }
    ]
    packets, mapping = make_packet_rows(source, "F")
    assert tuple(packets[0]) == PACKET_COLUMNS
    assert packets[0]["sample_id"] == "F001"
    assert "run" not in packets[0] and "record_id" not in packets[0]
    assert mapping[0]["model"] == "model-name"
    assert mapping[0]["record_id"] == "secret-record"


def test_practice_selection_is_deterministic_and_non_overlapping() -> None:
    formal = [{"record_id": "shared"}]
    candidates = [{"record_id": "shared"}] + [
        {"record_id": f"practice-{index}"} for index in range(25)
    ]
    first = select_practice_rows(candidates, formal, count=20, seed=7)
    second = select_practice_rows(candidates, formal, count=20, seed=7)
    assert first == second
    assert len(first) == 20
    assert all(row["record_id"] != "shared" for row in first)


def test_cohen_kappa_known_value_and_perfect_constant_case() -> None:
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(
        0.5
    )
    assert cohen_kappa(["other"] * 3, ["other"] * 3) == 1.0


def test_score_rows_aligns_by_sample_id_and_reports_disagreements() -> None:
    rows_a = [
        {"sample_id": "F001", "label": "hedging", "notes": "A1"},
        {"sample_id": "F002", "label": "other", "notes": "A2"},
    ]
    rows_b = [
        {"sample_id": "F002", "label": "partial_concession", "notes": "B2"},
        {"sample_id": "F001", "label": "hedging", "notes": "B1"},
    ]
    summary, disagreements = score_rows(rows_a, rows_b)
    assert summary["raw_agreement"] == 0.5
    assert summary["agreements"] == 1
    assert len(disagreements) == 1
    assert {
        key: disagreements[0][key]
        for key in ("sample_id", "label_A", "label_B", "notes_A", "notes_B")
    } == {
        "sample_id": "F002",
        "label_A": "other",
        "label_B": "partial_concession",
        "notes_A": "A2",
        "notes_B": "B2",
    }


def test_score_rows_rejects_mismatched_ids_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="sets differ"):
        score_rows(
            [{"sample_id": "F001", "label": "other"}],
            [{"sample_id": "F002", "label": "other"}],
        )
    with pytest.raises(ValueError, match="invalid or empty"):
        score_rows(
            [{"sample_id": "F001", "label": ""}],
            [{"sample_id": "F001", "label": "other"}],
        )


def test_write_scores_creates_adjudication_template(tmp_path) -> None:
    disagreement = {
        "sample_id": "F001",
        "label_A": "hedging",
        "label_B": "other",
        "notes_A": "",
        "notes_B": "unclear",
    }
    write_scores({"n": 1, "cohen_kappa": 0.0}, [disagreement], tmp_path)
    with (tmp_path / "adjudication.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "F001"
    assert rows[0]["label_A"] == "hedging"
    assert rows[0]["label_B"] == "other"
    assert rows[0]["final_label"] == ""
    assert (tmp_path / "adjudication.xlsx").exists()


def test_finalize_rows_uses_agreement_and_adjudication() -> None:
    rows_a = [
        {"sample_id": "F001", "label": "hedging", "dataset": "mmlu"},
        {"sample_id": "F002", "label": "other", "dataset": "gsm8k"},
    ]
    rows_b = [
        {"sample_id": "F001", "label": "hedging", "dataset": "mmlu"},
        {
            "sample_id": "F002",
            "label": "partial_concession",
            "dataset": "gsm8k",
        },
    ]
    adjudication = [
        {
            "sample_id": "F002",
            "final_label": "direct_concession",
            "adjudication_notes": "resolved",
        }
    ]
    mapping = [
        {
            "sample_id": "F001",
            "set": "formal",
            "run": "run-a",
            "model": "model-a",
            "record_id": "one",
        },
        {
            "sample_id": "F002",
            "set": "formal",
            "run": "run-b",
            "model": "model-b",
            "record_id": "two",
        },
    ]
    final, summary = finalize_rows(rows_a, rows_b, adjudication, mapping)
    assert [row["final_label"] for row in final] == [
        "hedging",
        "direct_concession",
    ]
    assert [row["resolution"] for row in final] == [
        "agreement",
        "adjudicated",
    ]
    assert summary["adjudicated"] == 1
