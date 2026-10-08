"""Meeting Dynamics, Sentiment, and Consensus Classification Network."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import BahdanauAttention


SENTIMENT_CLASSES = ["positive", "neutral", "negative"]
AGREEMENT_CLASSES = ["consensus", "neutral", "contention"]


class MeetingDynamicsNet(nn.Module):
    """Deep Multi-Task Network for Tracking Utterance Sentiment, Consensus, and Engagement.

    Predicts:
    1. Sentiment: Positive, Neutral, Negative
    2. Agreement/Consensus: Consensus (approval), Neutral, Contention (debate/disagreement)
    3. Engagement Intensity: Continuous scalar score [0.0, 1.0]
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        emb_dim: int = 128,
        hid_dim: int = 96,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bigru = nn.GRU(
            emb_dim,
            hid_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = BahdanauAttention(hid_dim, hid_dim, attn_dim=64)

        # Multi-task heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(SENTIMENT_CLASSES)),
        )

        self.agreement_head = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(AGREEMENT_CLASSES)),
        )

        self.engagement_head = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            sentiment_logits: [batch, 3]
            agreement_logits: [batch, 3]
            engagement_score: [batch, 1] in [0, 1]
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        embs = self.embedding(input_ids)
        hiddens, _ = self.bigru(embs)

        # Attention pooling
        query = hiddens.mean(dim=1)
        pooled, _ = self.attn(query, hiddens, mask=mask)

        sentiment_logits = self.sentiment_head(pooled)
        agreement_logits = self.agreement_head(pooled)
        engagement_score = self.engagement_head(pooled)

        return sentiment_logits, agreement_logits, engagement_score

    @torch.no_grad()
    def analyze_utterance(self, input_ids: torch.Tensor) -> Dict[str, any]:
        """Analyzes a single utterance and returns interpretable dynamics metrics."""
        self.eval()
        s_logits, a_logits, eng_score = self.forward(input_ids)

        s_probs = F.softmax(s_logits, dim=-1).squeeze(0).cpu().numpy()
        a_probs = F.softmax(a_logits, dim=-1).squeeze(0).cpu().numpy()
        eng = float(eng_score.squeeze().item())

        s_idx = int(s_probs.argmax())
        a_idx = int(a_probs.argmax())

        return {
            "sentiment": SENTIMENT_CLASSES[s_idx],
            "sentiment_confidence": round(float(s_probs[s_idx]), 3),
            "agreement": AGREEMENT_CLASSES[a_idx],
            "agreement_confidence": round(float(a_probs[a_idx]), 3),
            "engagement_score": round(eng, 3),
            "sentiment_probs": {cls: round(float(prob), 3) for cls, prob in zip(SENTIMENT_CLASSES, s_probs)},
        }
