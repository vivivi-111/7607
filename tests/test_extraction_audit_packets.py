from __future__ import annotations

import json

from analysis.build_extraction_audit_packets import candidate_rows


def test_prior_audit_record_is_excluded_from_both_rounds(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "dialogues.jsonl"
    rows = [
        {
            "record_id": "q1",
            "round": round_number,
            "response": "Final answer: A",
            "extraction_status": "ANSWER",
            "extracted": "A",
            "checkpoint_key": f"q1|round{round_number}",
        }
        for round_number in (1, 2)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidates = candidate_rows(
        [path],
        {
            "q1": {
                "question": "Question?",
                "choices": ["A", "B"],
            }
        },
        {"q1"},
        set(),
    )
    assert candidates == []
