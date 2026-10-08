"""Unit tests for Hierarchical Attention Network and TextRank summarization."""

import torch
from src.models.meeting_summarizer import HierarchicalAttentionSummarizer
from src.summarization.abstractive import MeetingMinutesGenerator
from src.summarization.extractive import TextRankExtractiveSummarizer


def test_han_forward_pass() -> None:
    model = HierarchicalAttentionSummarizer(
        vocab_size=1000,
        emb_dim=64,
        word_hid_dim=64,
        utt_hid_dim=96,
        dec_hid_dim=96,
    )
    # [B=2, U=4 utts, W=8 words per utt]
    meeting_tokens = torch.randint(1, 900, (2, 4, 8))
    target_tokens = torch.randint(1, 900, (2, 10))

    logits = model(meeting_tokens, target_tokens)
    # Target length is 10, so output steps are 9
    assert logits.shape == (2, 9, 1000)


def test_textrank_extractive_summarizer() -> None:
    summarizer = TextRankExtractiveSummarizer()
    dialogue = [
        "Welcome everyone to the sprint review.",
        "We completed the API migration on time with zero downtime.",
        "That is great news, good job team.",
        "Next sprint we will focus on the mobile app redesign.",
        "Thank you all for joining today.",
    ]
    summary = summarizer.summarize(dialogue, top_n=2)
    assert len(summary) == 2
    assert all(s in dialogue for s in summary)


def test_meeting_minutes_generator() -> None:
    gen = MeetingMinutesGenerator()
    lines = [
        "Alex: Let's review the sprint goals.",
        "Maya: We decided to deploy the new auth service by tomorrow.",
        "Leo: Regarding the UI, everything is green.",
    ]
    minutes = gen.generate(lines, meeting_title="Sprint Check")
    assert minutes.title == "Sprint Check"
    assert len(minutes.executive_summary) > 0
    assert len(minutes.key_decisions) >= 1
    assert len(minutes.discussion_topics) >= 1
