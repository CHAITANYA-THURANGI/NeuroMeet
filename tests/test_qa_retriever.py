"""Unit tests for Dense Retriever and Meeting Q&A Engine."""

import torch
from src.models.dense_retriever import DenseRetriever
from src.qa.meeting_qa import MeetingQAEngine


def test_dense_retriever_forward_and_shapes() -> None:
    retriever = DenseRetriever(vocab_size=1000, emb_dim=64, hid_dim=64)
    q = torch.randint(1, 900, (2, 8))
    d = torch.randint(1, 900, (3, 12))

    sim = retriever(q, d)
    assert sim.shape == (2, 3)

    q_vec = retriever.encode_query(q)
    assert q_vec.shape == (2, 64)
    # Check L2 normalization
    diff = torch.abs(torch.norm(q_vec, dim=-1) - 1.0).max().item()
    assert diff < 1e-4


def test_meeting_qa_engine() -> None:
    qa = MeetingQAEngine()
    turns = [
        {"speaker": "Maya", "text": "I am fixing the database webhook error by tomorrow."},
        {"speaker": "Leo", "text": "The frontend components are ready."},
    ]
    res = qa.answer_question("Who is fixing the webhook?", turns)
    assert res.query == "Who is fixing the webhook?"
    assert "Maya" in res.answer or "Maya" in res.citations[0].speaker
    assert len(res.citations) >= 1
