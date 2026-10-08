"""Unit tests for ECAPA-TDNN SpeakerNet architecture."""

import pytest
import torch
from src.models.speaker_net import (
    AttentiveStatisticsPooling,
    SpeakerNet,
    SqueezeExcitationBlock,
    TDNNBlock,
)


def test_squeeze_excitation_forward() -> None:
    se = SqueezeExcitationBlock(channels=64, bottleneck_dim=16)
    x = torch.randn(2, 64, 50)
    out = se(x)
    assert out.shape == (2, 64, 50)


def test_tdnn_block_forward() -> None:
    block = TDNNBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=2)
    x = torch.randn(2, 64, 80)
    out = block(x)
    assert out.shape == (2, 64, 80)


def test_attentive_statistics_pooling() -> None:
    asp = AttentiveStatisticsPooling(in_dim=64, attn_dim=32)
    x = torch.randn(2, 64, 100)
    out = asp(x)
    assert out.shape == (2, 128)  # 64 mean + 64 std


def test_speaker_net_forward_and_norm() -> None:
    net = SpeakerNet(feat_dim=80, channels=96, emb_dim=192)
    net.eval()
    x = torch.randn(3, 80, 120)
    emb = net(x)
    assert emb.shape == (3, 192)

    # Check L2 normalization: norm should equal 1.0
    norms = torch.norm(emb, p=2, dim=-1)
    diff = torch.abs(norms - 1.0).max().item()
    assert diff < 1e-4


def test_speaker_similarity() -> None:
    net = SpeakerNet(feat_dim=80, channels=96, emb_dim=192)
    e1 = torch.randn(1, 192)
    e1 = torch.nn.functional.normalize(e1, p=2, dim=-1)
    sim_self = net.compute_similarity(e1, e1)
    assert pytest.approx(sim_self, abs=1e-3) == 1.0
