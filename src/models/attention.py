"""Deep Learning Attention Mechanisms for NeuroMeet.
Implements Scaled Dot-Product, Bahdanau Additive, Luong General, and Multi-Head Attention.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) * V."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [..., seq_q, d_k]
            key:   [..., seq_k, d_k]
            value: [..., seq_k, d_v]
            mask:  Optional boolean or float mask broadcastable to [..., seq_q, seq_k]

        Returns:
            context: [..., seq_q, d_v]
            attn_weights: [..., seq_q, seq_k]
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        return context, attn_weights


class BahdanauAttention(nn.Module):
    """Additive Attention (Bahdanau et al., 2014):
    score(s, h) = v^T tanh(W_s s + W_h h)
    """

    def __init__(self, query_dim: int, key_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, query_dim] or [batch, 1, query_dim]
            keys:  [batch, seq_len, key_dim]
            mask:  Optional [batch, seq_len]

        Returns:
            context: [batch, key_dim]
            attn_weights: [batch, seq_len]
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, query_dim]

        proj_q = self.w_q(query)        # [batch, 1, attn_dim]
        proj_k = self.w_k(keys)         # [batch, seq_len, attn_dim]

        scores = self.v(torch.tanh(proj_q + proj_k)).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)  # [batch, key_dim]

        return context, attn_weights


class LuongGeneralAttention(nn.Module):
    """Multiplicative Attention (Luong et al., 2015):
    score(s, h) = s^T W h
    """

    def __init__(self, query_dim: int, key_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(key_dim, query_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, query_dim]
            keys:  [batch, seq_len, key_dim]
        """
        proj_k = self.w(keys)  # [batch, seq_len, query_dim]
        scores = torch.bmm(proj_k, query.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention module with projection layers."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections & split into heads: [B, num_heads, seq_len, d_k]
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_k]

        context, weights = self.attention(q, k, v, mask=mask)

        # Concatenate heads and project output: [B, seq_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_out(context)
        return out, weights
