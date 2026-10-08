"""Unit tests for Attention modules: Scaled Dot, Bahdanau, Luong, and Multi-Head."""

import pytest
import torch
from src.models.attention import (
    BahdanauAttention,
    LuongGeneralAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
)


def test_scaled_dot_product_attention() -> None:
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(2, 4, 32)
    k = torch.randn(2, 6, 32)
    v = torch.randn(2, 6, 32)
    out, weights = attn(q, k, v)
    assert out.shape == (2, 4, 32)
    assert weights.shape == (2, 4, 6)
    # Check row-sums of weights equal 1.0
    row_sums = weights.sum(dim=-1)
    diff = torch.abs(row_sums - 1.0).max().item()
    assert diff < 1e-4


def test_bahdanau_attention() -> None:
    attn = BahdanauAttention(query_dim=64, key_dim=64, attn_dim=32)
    q = torch.randn(2, 64)
    keys = torch.randn(2, 10, 64)
    ctx, weights = attn(q, keys)
    assert ctx.shape == (2, 64)
    assert weights.shape == (2, 10)


def test_luong_general_attention() -> None:
    attn = LuongGeneralAttention(query_dim=64, key_dim=64)
    q = torch.randn(2, 64)
    keys = torch.randn(2, 8, 64)
    ctx, weights = attn(q, keys)
    assert ctx.shape == (2, 64)
    assert weights.shape == (2, 8)


def test_multi_head_attention() -> None:
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 12, 64)
    out, weights = mha(x, x, x)
    assert out.shape == (2, 12, 64)
