"""Unit tests for Meeting Dynamics, Sentiment, and Health Scoring."""

import torch
from src.action_items.extractor import ActionItem
from src.analytics.meeting_health import compute_meeting_health_score
from src.analytics.participation import compute_participation_metrics
from src.analytics.sentiment_flow import analyze_sentiment_arc
from src.models.dynamics_net import MeetingDynamicsNet
from src.summarization.abstractive import MeetingMinutes


def test_dynamics_net_forward() -> None:
    net = MeetingDynamicsNet(vocab_size=1000, emb_dim=64, hid_dim=64)
    tokens = torch.randint(1, 900, (2, 10))
    s_logits, a_logits, eng = net(tokens)
    assert s_logits.shape == (2, 3)
    assert a_logits.shape == (2, 3)
    assert eng.shape == (2, 1)


def test_participation_metrics_and_gini() -> None:
    turns = [
        {"speaker": "Alice", "duration_sec": 10.0, "text": "Hello world."},
        {"speaker": "Bob", "duration_sec": 10.0, "text": "Hi Alice."},
    ]
    part = compute_participation_metrics(turns)
    assert part.total_duration_sec == 20.0
    assert part.talk_time_percentages["Alice"] == 50.0
    assert part.talk_time_percentages["Bob"] == 50.0
    # Gini of equal times is 0.0
    assert part.dominance_index == 0.0


def test_sentiment_arc_analysis() -> None:
    turns = [
        {"speaker": "Alex", "text": "Awesome job, this is excellent!"},
        {"speaker": "Bob", "text": "I disagree, there is a serious problem."},
    ]
    arc = analyze_sentiment_arc(turns)
    assert len(arc.turns) == 2
    assert arc.turns[0].sentiment == "positive"
    assert arc.turns[1].sentiment == "negative"
    assert len(arc.friction_points) >= 1


def test_meeting_health_score() -> None:
    turns = [
        {"speaker": "Alice", "duration_sec": 10.0, "text": "We decided to proceed."},
        {"speaker": "Bob", "duration_sec": 10.0, "text": "I will deliver the task by Friday."},
    ]
    part = compute_participation_metrics(turns)
    sent = analyze_sentiment_arc(turns)
    action = [
        ActionItem(
            task="Deliver the task",
            assignee="Bob",
            deadline="Friday",
            priority="high",
            confidence=0.9,
            source_utterance="I will deliver the task by Friday.",
        )
    ]
    mins = MeetingMinutes(
        title="Test",
        executive_summary="Summary",
        key_decisions=["We decided to proceed."],
    )
    health = compute_meeting_health_score(part, sent, action, mins)
    assert health.overall_score >= 0 and health.overall_score <= 100
    assert health.grade in ["A+", "A", "B", "C", "D"]
    assert len(health.recommendations) >= 1
