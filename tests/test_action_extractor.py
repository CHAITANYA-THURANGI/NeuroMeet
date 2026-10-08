"""Unit tests for Action Item Extraction and Prioritization."""

import torch
from src.action_items.extractor import ActionItem, DeepActionExtractor
from src.action_items.formatter import format_action_items_jira, format_action_items_markdown
from src.models.action_extractor import ActionItemClassifier


def test_action_item_classifier_shapes() -> None:
    classifier = ActionItemClassifier(vocab_size=1000, emb_dim=64, hid_dim=64)
    input_ids = torch.randint(1, 900, (2, 12))
    tag_logits, prob, prio_logits = classifier(input_ids)
    assert tag_logits.shape == (2, 12, 9)
    assert prob.shape == (2, 1)
    assert prio_logits.shape == (2, 4)


def test_deep_action_extractor_syntactic_cues() -> None:
    extractor = DeepActionExtractor()
    turn = "I will fix the payment timeout bug by tomorrow EOD."
    action = extractor.extract_from_utterance(turn, current_speaker="Maya")

    assert action is not None
    assert "payment" in action.task.lower() or "fix" in action.task.lower()
    assert action.deadline.lower().startswith("tomorrow")
    assert action.assignee == "Maya"


def test_format_action_items() -> None:
    item = ActionItem(
        task="Deploy Redis cache patch",
        assignee="Sarah",
        deadline="Tonight",
        priority="urgent",
        confidence=0.95,
        source_utterance="I will deploy Redis patch tonight.",
    )
    md = format_action_items_markdown([item])
    assert "[URGENT]" in md
    assert "Sarah" in md

    jira = format_action_items_jira([item])
    assert len(jira) == 1
    assert jira[0]["fields"]["summary"] == "Deploy Redis cache patch"
    assert jira[0]["fields"]["priority"]["name"] == "Highest"
