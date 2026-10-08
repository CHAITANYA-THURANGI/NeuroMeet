"""Unit tests for Speech, NLP, and Action evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    character_error_rate,
    compute_bleu,
    compute_rouge,
    diarization_error_rate,
    evaluate_action_items,
    word_error_rate,
)


def test_wer_and_cer() -> None:
    ref = "the meeting concluded on time"
    hyp = "the meeting concluded on time"
    assert word_error_rate(ref, hyp) == 0.0
    assert character_error_rate(ref, hyp) == 0.0

    hyp_err = "the meeting ended in time"
    wer = word_error_rate(ref, hyp_err)
    assert wer > 0.0


def test_rouge_and_bleu() -> None:
    ref = "maya will deliver the openapi schema by tomorrow"
    hyp = "maya will deliver the schema by tomorrow"

    rouge = compute_rouge(ref, hyp)
    assert rouge["rouge1"] > 70.0
    assert rouge["rougeL"] > 70.0

    bleu = compute_bleu(ref, hyp)
    assert bleu > 50.0


def test_diarization_error_rate() -> None:
    ref_turns = [{"start_sec": 0.0, "end_sec": 5.0, "speaker_id": "spk1"}]
    hyp_turns = [{"start_sec": 0.0, "end_sec": 5.0, "speaker_id": "spk1"}]
    der_res = diarization_error_rate(ref_turns, hyp_turns, total_duration_sec=5.0)
    assert der_res["der"] == 0.0


def test_evaluate_action_items() -> None:
    ref = [{"task": "deploy redis patch", "assignee": "Sarah"}]
    hyp = [{"task": "deploy redis patch with exponential backoff", "assignee": "Sarah"}]
    res = evaluate_action_items(ref, hyp)
    assert res["precision"] == 100.0
    assert res["recall"] == 100.0
    assert res["f1"] == 100.0
