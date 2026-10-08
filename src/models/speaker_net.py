"""Neural Speaker Verification & Embedding Network (ECAPA-TDNN inspired).
Produces 192-dimensional L2-normalized d-vectors for voice identification and diarization.
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for acoustic modeling."""

    def __init__(self, channels: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, bottleneck_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bottleneck_dim, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global temporal average pooling: [B, C, T] -> [B, C, 1]
        mean = x.mean(dim=-1, keepdim=True)
        scale = self.sigmoid(self.conv2(self.relu(self.conv1(mean))))
        return x * scale


class TDNNBlock(nn.Module):
    """Time-Delay Neural Network block with dilation and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcitationBlock(out_channels, bottleneck_dim=out_channels // 4)
        self.dropout = nn.Dropout(dropout)
        self.res = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res(x)
        out = self.relu(self.bn(self.conv(x)))
        out = self.se(out)
        out = self.dropout(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP):
    Computes temporal attention-weighted mean and variance across acoustic frames.
    """

    def __init__(self, in_dim: int, attn_dim: int = 96) -> None:
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Conv1d(in_dim, attn_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            pooled: [B, 2*C]
        """
        # Temporal attention weights: [B, C, T]
        alpha = self.attn_net(x)

        # Attention-weighted mean: [B, C]
        mean = torch.sum(alpha * x, dim=-1)

        # Attention-weighted standard deviation: [B, C]
        var = torch.sum(alpha * ((x - mean.unsqueeze(-1)) ** 2), dim=-1)
        std = torch.sqrt(torch.clamp(var, min=1e-5))

        # Concatenate mean and std: [B, 2*C]
        return torch.cat([mean, std], dim=-1)


class SpeakerNet(nn.Module):
    """End-to-End Speaker Verification Network.
    Maps acoustic Log-Mel features [B, 80, T] to 192-dim L2-normalized speaker embeddings.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        channels: int = 192,
        emb_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim

        # Front-end TDNN feature projection
        self.input_layer = nn.Sequential(
            nn.Conv1d(feat_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # Multi-scale dilated TDNN layers with Squeeze-and-Excitation
        self.layer1 = TDNNBlock(channels, channels, kernel_size=3, dilation=2, dropout=dropout)
        self.layer2 = TDNNBlock(channels, channels, kernel_size=3, dilation=3, dropout=dropout)
        self.layer3 = TDNNBlock(channels, channels, kernel_size=3, dilation=4, dropout=dropout)

        # Multi-layer feature aggregation: concatenate multi-scale layers
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 2, kernel_size=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(in_dim=channels * 2)

        # Embedding projection & bottleneck
        self.fc = nn.Linear((channels * 2) * 2, emb_dim)
        self.bn_out = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-Mel spectrogram [batch, feat_dim, time_frames]
        Returns:
            embedding: L2-normalized vector [batch, emb_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # TDNN forward
        out0 = self.input_layer(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        # Aggregation of multi-scale representations
        cat_features = torch.cat([out1, out2, out3], dim=1)
        mfa_out = self.mfa(cat_features)

        # Statistics pooling over time dimension
        stats = self.asp(mfa_out)

        # Final embedding projection
        emb = self.bn_out(self.fc(stats))

        # L2-normalization for hyperspherical cosine distance matching
        norm_emb = F.normalize(emb, p=2, dim=-1)
        return norm_emb

    @torch.no_grad()
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Computes cosine similarity between two speaker embeddings."""
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        return float(sim.mean().item())
