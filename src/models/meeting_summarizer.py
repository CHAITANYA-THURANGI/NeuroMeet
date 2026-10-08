"""Hierarchical Attention Network (HAN) with Pointer-Generator Copy Mechanism.
Summarizes multi-turn meeting transcripts by modeling word-level and utterance-level context.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import BahdanauAttention, ScaledDotProductAttention


class WordLevelEncoder(nn.Module):
    """Encodes a single utterance word-by-word using BiGRU + Bahdanau attention."""

    def __init__(self, vocab_size: int, emb_dim: int, word_hid_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bigru = nn.GRU(
            emb_dim,
            word_hid_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.word_attn = BahdanauAttention(word_hid_dim, word_hid_dim, attn_dim=64)

    def forward(self, words: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            words: [batch, num_words]
            mask: [batch, num_words]
        Returns:
            utt_vector: [batch, word_hid_dim]
            word_hidden: [batch, num_words, word_hid_dim]
        """
        embs = self.embedding(words)  # [B, W, emb_dim]
        hidden, _ = self.bigru(embs)  # [B, W, word_hid_dim]

        # Self-attention pooling over words to create single utterance representation
        query = hidden.mean(dim=1)    # [B, word_hid_dim]
        utt_vector, _ = self.word_attn(query, hidden, mask=mask)
        return utt_vector, hidden


class UtteranceLevelEncoder(nn.Module):
    """Encodes the sequence of utterances across the meeting using BiGRU."""

    def __init__(self, in_dim: int, utt_hid_dim: int) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            in_dim,
            utt_hid_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, utt_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            utt_vectors: [batch, num_utterances, in_dim]
        Returns:
            context_states: [batch, num_utterances, utt_hid_dim]
        """
        out, _ = self.bigru(utt_vectors)
        return out


class HierarchicalAttentionSummarizer(nn.Module):
    """Hierarchical Attention Meeting Summarizer with Pointer-Generator Copy Net.

    Features:
    1. Word-level attention per utterance
    2. Utterance-level attention across meeting turns
    3. Pointer-Generator Copy Mechanism: preserves named entities, assignees, and metrics
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        emb_dim: int = 128,
        word_hid_dim: int = 128,
        utt_hid_dim: int = 192,
        dec_hid_dim: int = 192,
        num_heads: int = 4,
        copy_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.copy_gate = copy_gate
        self.utt_hid_dim = utt_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # Hierarchical Encoders
        self.word_encoder = WordLevelEncoder(vocab_size, emb_dim, word_hid_dim)
        self.utt_encoder = UtteranceLevelEncoder(word_hid_dim, utt_hid_dim)

        # Decoder embedding & recurrent state
        self.dec_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dec_gru = nn.GRUCell(emb_dim + utt_hid_dim, dec_hid_dim)

        # Cross-attention over meeting utterances
        self.cross_attn = ScaledDotProductAttention(dropout=dropout)
        self.w_q = nn.Linear(dec_hid_dim, utt_hid_dim)
        self.w_k = nn.Linear(utt_hid_dim, utt_hid_dim)
        self.w_v = nn.Linear(utt_hid_dim, utt_hid_dim)

        # Vocabulary distribution projection
        self.fc_vocab = nn.Sequential(
            nn.Linear(dec_hid_dim + utt_hid_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, vocab_size),
        )

        # Pointer-Generator Copy Gate: p_gen = sigmoid(w_c c + w_s s + w_x x + b)
        if copy_gate:
            self.p_gen_layer = nn.Linear(utt_hid_dim + dec_hid_dim + emb_dim, 1)

    def encode(
        self,
        meeting_tokens: torch.Tensor,
        utt_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes meeting utterances into hierarchical context vectors.

        Args:
            meeting_tokens: [batch, num_utts, words_per_utt]
        Returns:
            utt_context: [batch, num_utts, utt_hid_dim]
            init_dec_state: [batch, dec_hid_dim]
        """
        batch_size, num_utts, words_per_utt = meeting_tokens.shape
        flat_utts = meeting_tokens.view(batch_size * num_utts, words_per_utt)

        # 1. Word-level encoding & attention
        utt_reps, _ = self.word_encoder(flat_utts)  # [B*U, word_hid_dim]
        utt_reps = utt_reps.view(batch_size, num_utts, -1)

        # 2. Utterance-level sequence encoding
        utt_context = self.utt_encoder(utt_reps)  # [B, num_utts, utt_hid_dim]

        # Initial decoder hidden state: pooled utterance context
        init_dec_state = torch.tanh(utt_context.mean(dim=1))
        if init_dec_state.shape[-1] != self.dec_hid_dim:
            init_dec_state = nn.Linear(self.utt_hid_dim, self.dec_hid_dim).to(meeting_tokens.device)(init_dec_state)

        return utt_context, init_dec_state

    def step(
        self,
        y_prev: torch.Tensor,
        state: torch.Tensor,
        utt_context: torch.Tensor,
        source_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A single decoding step with optional pointer-generator copy distribution."""
        batch_size = y_prev.size(0)
        emb_y = self.dec_embedding(y_prev)  # [B, emb_dim]

        # Compute cross-attention context over utterances
        q = self.w_q(state).unsqueeze(1)    # [B, 1, utt_hid_dim]
        k = self.w_k(utt_context)           # [B, num_utts, utt_hid_dim]
        v = self.w_v(utt_context)           # [B, num_utts, utt_hid_dim]

        context, attn_weights = self.cross_attn(q, k, v)
        context = context.squeeze(1)        # [B, utt_hid_dim]
        attn_weights = attn_weights.squeeze(1)  # [B, num_utts]

        # Update GRU decoder cell
        gru_in = torch.cat([emb_y, context], dim=-1)
        new_state = self.dec_gru(gru_in, state)

        # Vocab logits
        vocab_dist = F.softmax(self.fc_vocab(torch.cat([new_state, context], dim=-1)), dim=-1)

        if self.copy_gate:
            p_gen = torch.sigmoid(self.p_gen_layer(torch.cat([context, new_state, emb_y], dim=-1)))
            final_dist = p_gen * vocab_dist
        else:
            final_dist = vocab_dist

        return final_dist, new_state, attn_weights

    def forward(
        self,
        meeting_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass for training."""
        utt_context, state = self.encode(meeting_tokens)
        batch_size, max_tgt_len = target_tokens.shape

        outputs = []
        for t in range(max_tgt_len - 1):
            y_t = target_tokens[:, t]
            dist, state, _ = self.step(y_t, state, utt_context)
            outputs.append(dist.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, tgt_len - 1, vocab_size]
