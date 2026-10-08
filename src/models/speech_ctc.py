"""Conformer & BiGRU Acoustic Model with Connectionist Temporal Classification (CTC).
Processes Log-Mel spectrograms into character-level speech transcriptions.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


# Standard English character vocabulary for meeting transcription
CHAR_VOCAB = [
    "<blank>",  # 0: CTC Blank token
    " ",        # 1: Space
    "'", "-", ".", ",", "?", "!",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}


class ConformerFeedForward(nn.Module):
    """Feed-Forward module with Swish/SiLU activation and expansion factor."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Depthwise separable convolution module with GLU gating."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.glu(self.pointwise_conv1(x))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.silu(x)
        x = self.dropout(self.pointwise_conv2(x))
        x = x.transpose(1, 2)  # [B, T, D]
        return residual + x


class ConformerBlock(nn.Module):
    """Conformer Block: Macaron-style FFN + MHSA + Conv + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        kernel_size: int = 15,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Macaron half-step FFN 1
        x = x + 0.5 * self.ffn1(x)
        # Multi-Head Self-Attention
        attn_out, _ = self.self_attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x), mask=mask)
        x = x + attn_out
        # Depthwise Convolution
        x = self.conv_module(x)
        # Macaron half-step FFN 2
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class SpeechCTC(nn.Module):
    """Conformer-BiGRU Acoustic CTC Model for Speech-to-Text Transcription."""

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 144,
        n_conformer_blocks: int = 4,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        vocab_size: int = len(CHAR_VOCAB),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Subsampling front-end: Conv1D with stride 2
        self.subsampling = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        # Conformer stack
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                kernel_size=conv_kernel_size,
                ff_expansion=4,
                dropout=dropout,
            )
            for _ in range(n_conformer_blocks)
        ])

        # Temporal BiGRU layer for recurrent phonetic modeling
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Final CTC projection layer
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-Mel spectrogram [batch, feat_dim, time_frames]

        Returns:
            log_probs: [batch, subsampled_time, vocab_size] in log-softmax space
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Subsample time: [B, feat_dim, T] -> [B, d_model, T_sub]
        x_sub = self.subsampling(x)
        # Transpose to [B, T_sub, d_model]
        x_trans = x_sub.transpose(1, 2)

        # Conformer passes
        for block in self.conformer_blocks:
            x_trans = block(x_trans)

        # BiGRU pass
        gru_out, _ = self.bigru(x_trans)

        # CTC logits & log-probabilities
        logits = self.classifier(gru_out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def decode_greedy(self, log_probs: torch.Tensor) -> List[str]:
        """Greedy CTC collapse: selects argmax and collapses repeating chars and blanks."""
        # argmax along vocab dimension: [B, T]
        preds = torch.argmax(log_probs, dim=-1).cpu().numpy()
        transcripts: List[str] = []

        for row in preds:
            collapsed = []
            prev_token = -1
            for token_idx in row:
                if token_idx != prev_token:
                    if token_idx != 0:  # 0 is <blank>
                        char = IDX2CHAR.get(token_idx, "")
                        collapsed.append(char)
                    prev_token = token_idx
            transcripts.append("".join(collapsed).strip())

        return transcripts
