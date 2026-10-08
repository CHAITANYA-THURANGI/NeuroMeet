"""
Tests for long-duration meeting support, chaptering, vectorized VAD, and English corporate discourse.
"""

import numpy as np
import pytest
import torch

from src.audio.vad import EnergyZCRVAD
from src.datasets.generator import generate_long_executive_townhall
from src.diarization.clustering import SpectralSpeakerClusterer
from src.models.meeting_summarizer import HierarchicalAttentionSummarizer
from src.pipeline.omni_meeting import OmniMeetingPipeline
from src.summarization.abstractive import MeetingMinutesGenerator
from src.summarization.hierarchical import HierarchicalMeetingSummarizer


def test_hierarchical_summarizer_long_transcript_chunking():
    """Verify that hierarchical attention summarizer processes 120+ turns without truncation."""
    vocab_size = 1000
    inner_model = HierarchicalAttentionSummarizer(
        vocab_size=vocab_size,
        emb_dim=64,
        word_hid_dim=64,
        utt_hid_dim=96,
        dec_hid_dim=96,
    )
    model = HierarchicalMeetingSummarizer(model=inner_model, vocab_size=vocab_size, max_utts=64)

    # 135 utterances (exceeding standard max_utts of 64)
    utterances = [
        f"Speaker {i % 4}: We are discussing point {i} in the extended corporate roadmap and aligning timelines."
        for i in range(135)
    ]

    salience = model.rank_salience(utterances)
    assert len(salience) == 135
    # Verify weights are non-negative and normalized to approximately 1.0
    assert all(w >= 0.0 for w in salience)
    assert pytest.approx(sum(salience), rel=0.01) == 1.0


def test_meeting_minutes_chaptering_on_long_meetings():
    """Verify that meetings with >= 12 turns generate structured chapters."""
    generator = MeetingMinutesGenerator()
    townhall = generate_long_executive_townhall()
    transcript_lines = [f"{t['speaker']}: {t['text']}" for t in townhall.turns]

    minutes = generator.generate(transcript_lines, meeting_title=townhall.title)
    assert hasattr(minutes, "chapters")
    chapters = minutes.chapters

    # 24 turns should create between 2 and 5 chapters
    assert 2 <= len(chapters) <= 5
    for ch in chapters:
        assert "title" in ch
        assert "summary" in ch
        assert "start_turn" in ch
        assert "end_turn" in ch
        assert ch["start_turn"] <= ch["end_turn"]

    # Check key executive decisions and highlights in English corporate discourse
    assert len(minutes.key_decisions) >= 2
    assert len(minutes.key_highlights) >= 2


def test_energy_zcr_vad_long_vectorized_audio():
    """Test vectorized VAD computation over 60 seconds of synthetic audio."""
    sample_rate = 16000
    duration_sec = 60
    total_samples = sample_rate * duration_sec

    # Create synthetic audio: alternate 2s silence and 2s tone
    waveform = np.zeros(total_samples, dtype=np.float32)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    # Inject 440 Hz tone every other 2-second block
    for sec_start in range(0, duration_sec, 4):
        start_idx = sec_start * sample_rate
        end_idx = min(total_samples, (sec_start + 2) * sample_rate)
        waveform[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 440 * t[start_idx:end_idx])

    vad = EnergyZCRVAD(sample_rate=sample_rate, frame_length_ms=25.0, hop_length_ms=10.0)
    segments = vad.detect(waveform)

    assert len(segments) > 0
    for seg in segments:
        assert 0.0 <= seg.start_sec < seg.end_sec <= float(duration_sec)
        assert seg.duration_sec > 0.0
        assert seg.confidence >= 0.0


def test_landmark_spectral_clustering_scalability():
    """Verify that landmark-accelerated spectral clustering handles 700+ window features without OOM or failure."""
    clusterer = SpectralSpeakerClusterer(min_speakers=2, max_speakers=4)

    # 750 window vectors of dim 32, partitioned into 3 distinct speaker centers
    num_windows = 750
    dim = 32
    centers = np.random.randn(3, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels_true = np.random.randint(0, 3, size=num_windows)
    features = centers[labels_true] + 0.1 * np.random.randn(num_windows, dim)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    labels_pred = clusterer.cluster(features)
    assert len(labels_pred) == num_windows
    assert len(set(labels_pred)) in [2, 3, 4]


def test_omni_pipeline_executive_townhall_e2e():
    """Verify end-to-end processing of a long executive town hall meeting."""
    pipeline = OmniMeetingPipeline()
    townhall = generate_long_executive_townhall()

    result = pipeline.process_transcript(
        transcript_input=townhall.turns,
        title=townhall.title,
    )

    assert result.title == townhall.title
    assert result.total_duration_sec >= 120.0
    assert len(result.turns) == len(townhall.turns)
    assert len(result.minutes.chapters) >= 2
    assert len(result.action_items) >= 2
    assert len(result.minutes.key_decisions) >= 2

    # Verify English executive decisions
    joined_decisions = " ".join(result.minutes.key_decisions).lower()
    assert (
        "signed off" in joined_decisions
        or "deploy" in joined_decisions
        or "budget" in joined_decisions
        or "approved" in joined_decisions
        or "proceed" in joined_decisions
    )

    # Verify Markdown export contains Agenda Chapters
    md = result.to_markdown()
    assert "Structured Agenda Phases:" in md
    assert "Executive Summary" in md
    assert "Action Items & Commitments" in md
