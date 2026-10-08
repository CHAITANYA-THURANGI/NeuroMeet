"""Neural & Lexical Semantic Q&A over Meeting Transcripts."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional
import torch
from ..models.dense_retriever import DenseRetriever


@dataclass
class QACitation:
    speaker: str
    text: str
    turn_index: int
    score: float
    timestamp_sec: Optional[float] = None


@dataclass
class QAResult:
    query: str
    answer: str
    confidence: float
    citations: List[QACitation] = field(default_factory=list)


def simple_hash_tokens(text: str, vocab_size: int = 4000) -> torch.Tensor:
    words = text.split()
    ids = []
    for w in words:
        clean = w.lower().strip()
        val = 1 + (abs(hash(clean)) % (vocab_size - 1)) if clean else 0
        ids.append(val)
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)


class MeetingQAEngine:
    """Answers arbitrary factual, procedural, or assignment questions about the meeting."""

    def __init__(
        self,
        retriever: Optional[DenseRetriever] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.retriever = retriever.to(self.device).eval() if retriever else None

    def answer_question(
        self,
        query: str,
        turns: List[Dict[str, any]],
        top_k: int = 3,
    ) -> QAResult:
        """Finds most relevant turns and formulates grounded response with exact citations."""
        if not turns:
            return QAResult(
                query=query,
                answer="No meeting content is loaded to answer this question.",
                confidence=0.0,
                citations=[],
            )

        q_clean = query.strip().lower()
        q_tokens = set(re.findall(r"\b\w+\b", q_clean, flags=re.UNICODE))

        scored_turns = []

        for idx, turn in enumerate(turns):
            text = turn.get("text", "")
            translation = turn.get("translation", "")
            spk = turn.get("speaker", f"Speaker {idx + 1}")
            start_t = turn.get("start_sec", None)

            # 1. Lexical overlap score (matches both native Hindi/Telugu words and English translation)
            t_tokens = set(re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE))
            if translation:
                t_tokens |= set(re.findall(r"\b\w+\b", translation.lower(), flags=re.UNICODE))
            overlap = len(q_tokens.intersection(t_tokens))
            lex_score = overlap / math.sqrt(len(q_tokens) * max(1, len(t_tokens)))

            # 2. Neural similarity score (if retriever present)
            neu_score = 0.0
            if self.retriever is not None:
                q_tensor = simple_hash_tokens(query).to(self.device)
                d_tensor = simple_hash_tokens(text).to(self.device)
                with torch.no_grad():
                    q_emb = self.retriever.encode_query(q_tensor)
                    d_emb = self.retriever.encode_doc(d_tensor)
                    neu_score = float(torch.matmul(q_emb, d_emb.T).item())

            # Fused score
            final_score = (lex_score * 0.6) + (neu_score * 0.4) if self.retriever else lex_score

            scored_turns.append({
                "turn_index": idx,
                "speaker": spk,
                "text": text,
                "score": final_score,
                "timestamp_sec": start_t,
            })

        # Rank turns
        scored_turns.sort(key=lambda x: x["score"], reverse=True)
        top_matches = scored_turns[:top_k]

        citations = [
            QACitation(
                speaker=m["speaker"],
                text=m["text"],
                turn_index=m["turn_index"],
                score=round(float(m["score"]), 3),
                timestamp_sec=m["timestamp_sec"],
            )
            for m in top_matches if m["score"] > 0.01
        ]

        if not citations:
            return QAResult(
                query=query,
                answer=f"I couldn't find any direct discussion about '{query}' in the meeting transcript.",
                confidence=0.2,
                citations=[],
            )

        # Synthesize clear answer
        best = citations[0]
        if "who" in q_clean:
            answer = f"According to {best.speaker}, \"{best.text}\""
        elif "when" in q_clean or "date" in q_clean or "deadline" in q_clean:
            answer = f"In discussion, {best.speaker} noted: \"{best.text}\""
        elif "what" in q_clean or "why" in q_clean:
            answer = f"{best.speaker} stated: \"{best.text}\""
        else:
            answer = f"Based on the transcript, {best.speaker} said: \"{best.text}\""

        top_conf = min(0.98, max(0.45, best.score * 1.5))

        return QAResult(
            query=query,
            answer=answer,
            confidence=round(top_conf, 2),
            citations=citations,
        )
