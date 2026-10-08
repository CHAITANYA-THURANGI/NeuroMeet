"""Deep Action Item, Assignee, and Deadline Sequence Tagger.
Extracts structured action items from spoken meeting utterances.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


TAG_SET = ["O", "B-ACT", "I-ACT", "B-WHO", "I-WHO", "B-WHEN", "I-WHEN", "B-PRIO", "I-PRIO"]
TAG2IDX = {tag: i for i, tag in enumerate(TAG_SET)}
IDX2TAG = {i: tag for i, tag in enumerate(TAG_SET)}
PRIORITY_CLASSES = ["low", "medium", "high", "urgent"]


class ActionItemClassifier(nn.Module):
    """Multi-Task Neural Network for Meeting Action Item Extraction.

    Tasks:
    1. Token-level BIO tagging (Action verb, task description, assignee, deadline)
    2. Utterance actionability probability (Is there an action item?)
    3. Priority classification (Low, Medium, High, Urgent)
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        emb_dim: int = 128,
        hid_dim: int = 160,
        num_layers: int = 2,
        num_heads: int = 4,
        num_tags: int = len(TAG_SET),
        num_priorities: int = len(PRIORITY_CLASSES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM sequence backbone
        self.bilstm = nn.LSTM(
            emb_dim,
            hid_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-Head Self-Attention over tokens
        self.self_attn = MultiHeadAttention(hid_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hid_dim)

        # Head 1: Token-level BIO tag classifier
        self.tag_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim // 2, num_tags),
        )

        # Head 2: Utterance-level Actionability score (binary)
        self.action_head = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Head 3: Priority category classifier (4-class)
        self.priority_head = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_priorities),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            mask: Optional [batch, seq_len]
        Returns:
            tag_logits: [batch, seq_len, num_tags]
            action_prob: [batch, 1] in [0, 1]
            priority_logits: [batch, num_priorities]
        """
        embs = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.bilstm(embs)  # [B, T, hid_dim]

        # Multi-Head Attention enhancement
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out, mask=mask)
        rep = self.norm(lstm_out + attn_out)

        # Token-level tag predictions
        tag_logits = self.tag_head(rep)  # [B, T, num_tags]

        # Utterance-level pooling (mean over valid tokens)
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            pooled = (rep * mask_float).sum(dim=1) / torch.clamp(mask_float.sum(dim=1), min=1.0)
        else:
            pooled = rep.mean(dim=1)

        action_prob = torch.sigmoid(self.action_head(pooled))
        priority_logits = self.priority_head(pooled)

        return tag_logits, action_prob, priority_logits

    @torch.no_grad()
    def extract_action(
        self,
        words: List[str],
        input_ids: torch.Tensor,
        threshold: float = 0.45,
    ) -> Optional[Dict[str, str]]:
        """Parses words and model predictions into structured action item entity."""
        self.eval()
        tag_logits, action_prob, priority_logits = self.forward(input_ids.unsqueeze(0))

        prob = float(action_prob.squeeze().item())
        if prob < threshold:
            return None

        # Predict tags
        pred_tags = torch.argmax(tag_logits.squeeze(0), dim=-1).cpu().numpy()
        pred_priority = PRIORITY_CLASSES[int(torch.argmax(priority_logits.squeeze(0), dim=-1).item())]

        action_tokens = []
        who_tokens = []
        when_tokens = []

        for idx, word in enumerate(words):
            if idx >= len(pred_tags):
                break
            tag = IDX2TAG.get(int(pred_tags[idx]), "O")

            if "ACT" in tag:
                action_tokens.append(word)
            elif "WHO" in tag:
                who_tokens.append(word)
            elif "WHEN" in tag:
                when_tokens.append(word)

        task_desc = " ".join(action_tokens).strip() or " ".join(words)
        assignee = " ".join(who_tokens).strip() or "Unassigned"
        deadline = " ".join(when_tokens).strip() or "Next sync"

        return {
            "task": task_desc,
            "assignee": assignee,
            "deadline": deadline,
            "priority": pred_priority,
            "confidence": round(prob, 3),
        }
