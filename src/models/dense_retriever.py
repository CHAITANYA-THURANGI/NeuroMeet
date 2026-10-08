"""Dual-Encoder Dense Retriever for Meeting Q&A and Conversational Memory."""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ScaledDotProductAttention


class DenseRetriever(nn.Module):
    """Dual-Encoder Neural Semantic Retriever for indexing and querying meeting transcripts.
    Computes dense semantic embeddings for both queries and transcript segments.
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        emb_dim: int = 128,
        hid_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Query encoder
        self.q_gru = nn.GRU(emb_dim, hid_dim // 2, batch_first=True, bidirectional=True)
        self.q_proj = nn.Linear(hid_dim, hid_dim)

        # Context utterance encoder
        self.doc_gru = nn.GRU(emb_dim, hid_dim // 2, batch_first=True, bidirectional=True)
        self.doc_proj = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        """Encodes query into L2-normalized dense vector [B, hid_dim]."""
        if query_ids.dim() == 1:
            query_ids = query_ids.unsqueeze(0)
        embs = self.dropout(self.embedding(query_ids))
        out, _ = self.q_gru(embs)
        pooled = out.mean(dim=1)
        proj = self.q_proj(pooled)
        return F.normalize(proj, p=2, dim=-1)

    def encode_doc(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """Encodes meeting document/utterance into L2-normalized dense vector [B, hid_dim]."""
        if doc_ids.dim() == 1:
            doc_ids = doc_ids.unsqueeze(0)
        embs = self.dropout(self.embedding(doc_ids))
        out, _ = self.doc_gru(embs)
        pooled = out.mean(dim=1)
        proj = self.doc_proj(pooled)
        return F.normalize(proj, p=2, dim=-1)

    def forward(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity matrix between queries and documents."""
        q_vec = self.encode_query(query_ids)
        d_vec = self.encode_doc(doc_ids)
        # Dot product of normalized vectors = cosine similarity
        sim = torch.matmul(q_vec, d_vec.transpose(0, 1))
        return sim

    @torch.no_grad()
    def search(
        self,
        query_ids: torch.Tensor,
        doc_embeddings: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds top-k closest documents for the given query."""
        self.eval()
        q_vec = self.encode_query(query_ids)
        # doc_embeddings: [N, hid_dim]
        scores = torch.matmul(q_vec, doc_embeddings.transpose(0, 1)).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
        return top_scores, top_indices
