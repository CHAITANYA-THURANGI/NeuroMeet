"""Hierarchical Attention Meeting Summarizer integration."""

from __future__ import annotations
import re
from typing import Dict, List, Optional
import torch
from ..models.meeting_summarizer import HierarchicalAttentionSummarizer


def simple_hash_vocab(token: str, vocab_size: int = 4000) -> int:
    """Hashes token into vocabulary ID 1..vocab_size-1 (0 is PAD)."""
    clean = token.lower().strip()
    if not clean:
        return 0
    return 1 + (abs(hash(clean)) % (vocab_size - 1))


class HierarchicalMeetingSummarizer:
    """Orchestrates neural hierarchical summarization over meeting turns."""

    def __init__(
        self,
        model: HierarchicalAttentionSummarizer,
        vocab_size: int = 4000,
        max_utts: int = 64,
        max_words_per_utt: int = 32,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab_size = vocab_size
        self.max_utts = max_utts
        self.max_words_per_utt = max_words_per_utt

    def _prepare_tensors(self, utterances: List[str]) -> torch.Tensor:
        """Pads and hashes text utterances into tensor [1, num_utts, words_per_utt]."""
        num_utts = min(len(utterances), self.max_utts)
        if num_utts == 0:
            return torch.zeros((1, 1, self.max_words_per_utt), dtype=torch.long, device=self.device)

        tensor = torch.zeros((1, num_utts, self.max_words_per_utt), dtype=torch.long, device=self.device)

        for u_idx in range(num_utts):
            words = utterances[u_idx].split()[: self.max_words_per_utt]
            for w_idx, w in enumerate(words):
                tensor[0, u_idx, w_idx] = simple_hash_vocab(w, self.vocab_size)

        return tensor

    @torch.no_grad()
    def rank_salience(self, utterances: List[str]) -> List[float]:
        """Calculates utterance-level attention weights reflecting meeting importance."""
        if not utterances:
            return []
        inputs = self._prepare_tensors(utterances)
        utt_context, _ = self.model.encode(inputs)

        # Global meeting context query
        q = utt_context.mean(dim=1, keepdim=True)
        _, weights = self.model.cross_attn(q, utt_context, utt_context)
        weights_np = weights.squeeze().cpu().numpy()

        if weights_np.ndim == 0:
            return [1.0]
        return [round(float(w), 4) for w in weights_np[: len(utterances)]]
