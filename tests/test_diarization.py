"""Unit tests for Diarization Segmentation and Clustering."""

import numpy as np
import pytest
import torch
from src.audio.wav_io import generate_synthetic_audio
from src.diarization.clustering import (
    AgglomerativeSpeakerClusterer,
    SpectralSpeakerClusterer,
    cosine_affinity_matrix,
)
from src.diarization.diarizer import SpeakerDiarizer
from src.models.speaker_net import SpeakerNet


def test_cosine_affinity_matrix() -> None:
    embs = np.random.randn(4, 32).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    aff = cosine_affinity_matrix(embs)
    assert aff.shape == (4, 4)
    assert np.all(aff >= 0.0) and np.all(aff <= 1.0)
    assert np.allclose(np.diag(aff), 1.0)


def test_spectral_clusterer() -> None:
    clusterer = SpectralSpeakerClusterer(min_speakers=2, max_speakers=4)
    # 2 synthetic distinct clusters
    c1 = np.ones((5, 16), dtype=np.float32) + np.random.randn(5, 16) * 0.05
    c2 = -np.ones((5, 16), dtype=np.float32) + np.random.randn(5, 16) * 0.05
    embs = np.vstack([c1, c2])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    labels = clusterer.cluster(embs, num_speakers=2)
    assert len(labels) == 10
    # Cluster 1 members should share the same label
    assert len(set(labels[:5])) == 1
    # Cluster 2 members should share the same label
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_speaker_diarizer_pipeline() -> None:
    net = SpeakerNet(feat_dim=80, channels=64, emb_dim=128)
    diarizer = SpeakerDiarizer(speaker_net=net, sample_rate=16000)

    audio = generate_synthetic_audio(duration_sec=3.0, sample_rate=16000)
    turns = diarizer.diarize(audio)
    assert isinstance(turns, list)
    if turns:
        assert turns[0].start_sec >= 0.0
        assert turns[0].end_sec <= 3.1
        assert "Speaker" in turns[0].speaker_id
