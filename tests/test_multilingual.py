"""Unit tests for multilingual processing (Hindi, Telugu, English) across NeuroMeet AI."""

import pytest
from src.action_items.extractor import ActionItemClassifier, DeepActionExtractor
from src.datasets.generator import generate_multilingual_india_sync
from src.pipeline.omni_meeting import OmniMeetingPipeline, detect_turn_language
from src.qa.meeting_qa import MeetingQAEngine
from src.summarization.extractive import TextRankExtractiveSummarizer


def test_detect_turn_language() -> None:
    """Tests language detection across Unicode scripts and transliterated keywords."""
    # Devanagari Hindi
    assert detect_turn_language("मैं कल शाम तक काम पूरा कर दूँगा।") == "hi"
    # Telugu Script
    assert detect_turn_language("నేను శుక్రవారం లోగా Android SDK build ని పంపిస్తాను.") == "te"
    # Hinglish
    assert detect_turn_language("main kal tak API schema finalize karunga") == "hi"
    # Tenglish
    assert detect_turn_language("nenu repu morning kalla test chestanu") == "te"
    # Standard English
    assert detect_turn_language("Good morning team, let's review the sprint deliverables.") == "en"


def test_multilingual_action_item_extraction() -> None:
    """Verifies that Hindi and Telugu commitment patterns are extracted as action items."""
    extractor = DeepActionExtractor()
    turns = [
        {
            "turn_index": 0,
            "speaker": "Sneha",
            "language": "hi",
            "text": "मैं कल शाम तक database index optimization patch deploy करूँगी।",
        },
        {
            "turn_index": 1,
            "speaker": "Karthik",
            "language": "te",
            "text": "నేను శుక్రవారం లోగా Android SDK build ని QA టీమ్‌కి పంపిస్తాను.",
        },
        {
            "turn_index": 2,
            "speaker": "Rajesh",
            "language": "en",
            "text": "Great, let's keep the focus on performance.",
        },
    ]

    action_items = extractor.extract_all(turns)
    assert len(action_items) >= 2

    # Check Hindi action
    hi_items = [a for a in action_items if a.assignee == "Sneha"]
    assert len(hi_items) >= 1
    assert "कल" in hi_items[0].deadline or "शाम" in hi_items[0].deadline or "तक" in hi_items[0].deadline

    # Check Telugu action
    te_items = [a for a in action_items if a.assignee == "Karthik"]
    assert len(te_items) >= 1
    assert "శుక్రవారం" in te_items[0].deadline or "లోగా" in te_items[0].deadline


def test_multilingual_textrank_summarization() -> None:
    """Tests that TextRank extractive summarizer handles Devanagari and Telugu without tokenization errors."""
    summarizer = TextRankExtractiveSummarizer()
    sentences = [
        "Good morning everyone, let's start the sync.",
        "हाँ राजेश, Redis cache लगाने के बाद latency 40% कम हो गई है।",
        "మొబైల్ క్లయింట్‌లో offline speech recognition model చాలా వేగంగా పనిచేస్తోంది.",
        "We decided to roll out the multilingual pilot in Mumbai and Hyderabad next Tuesday.",
    ]

    summary_sents = summarizer.summarize(sentences, top_n=2)
    assert len(summary_sents) == 2
    assert any("Redis" in s or "మొబైల్" in s or "multilingual" in s for s in summary_sents)


def test_multilingual_qa_crosslingual() -> None:
    """Verifies cross-lingual retrieval matching across translations and native text."""
    qa = MeetingQAEngine()
    scenario = generate_multilingual_india_sync()
    turns = scenario.turns

    # Query in English about Telugu speaker's task
    res_te = qa.answer_question("Who is working on the Android SDK build?", turns)
    assert "Karthik" in res_te.answer or "Karthik" in str(res_te.citations)

    # Query in English about Redis cache
    res_hi = qa.answer_question("What is the status of Redis cache latency?", turns)
    assert "40%" in res_hi.answer or "latency" in res_hi.answer.lower() or len(res_hi.citations) >= 1


def test_omni_pipeline_multilingual_sync() -> None:
    """Tests full OmniMeetingPipeline execution on a multilingual meeting scenario."""
    pipeline = OmniMeetingPipeline()
    scenario = generate_multilingual_india_sync()

    result = pipeline.process_transcript(scenario.turns, title="Multilingual India Sync")

    assert result.title == "Multilingual India Sync"
    assert len(result.turns) == 7
    assert "hi" in result.detected_languages
    assert "te" in result.detected_languages
    assert "en" in result.detected_languages
    assert len(result.action_items) >= 2
    assert result.health.overall_score > 0

    # Test markdown output includes language info
    md = result.to_markdown()
    assert "HI" in md
    assert "TE" in md
    assert "EN" in md
