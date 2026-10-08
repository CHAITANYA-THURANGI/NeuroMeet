"""Meeting Sentiment Arc, Consensus, and Friction Analysis."""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
import torch
from ..models.dynamics_net import MeetingDynamicsNet


POSITIVE_WORDS = {
    "great", "awesome", "excellent", "agree", "perfect", "good", "love", "impressive",
    "sounds good", "absolutely", "definitely", "on track", "approved", "excited"
}
NEGATIVE_WORDS = {
    "disagree", "issue", "concern", "problem", "broken", "delay", "blocker", "bad",
    "worry", "risk", "failed", "conflict", "behind", "friction", "skeptical"
}


@dataclass
class TurnDynamics:
    turn_index: int
    speaker: str
    text: str
    sentiment: str  # positive, neutral, negative
    sentiment_score: float  # -1.0 to +1.0
    agreement: str  # consensus, neutral, contention
    engagement: float  # 0.0 to 1.0


@dataclass
class SentimentArc:
    turns: List[TurnDynamics]
    overall_sentiment: str
    sentiment_balance: Dict[str, float]  # % positive, neutral, negative
    consensus_score: float  # 0.0 to 100.0
    friction_points: List[Dict[str, Any]]


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


def analyze_sentiment_arc(
    turns: List[Dict[str, any]],
    dynamics_model: Optional[MeetingDynamicsNet] = None,
    device: str = "cpu",
) -> SentimentArc:
    """Analyzes chronological sentiment and consensus dynamics across dialogue turns."""
    if not turns:
        return SentimentArc(
            turns=[],
            overall_sentiment="neutral",
            sentiment_balance={"positive": 0.0, "neutral": 100.0, "negative": 0.0},
            consensus_score=50.0,
            friction_points=[],
        )

    analyzed_turns: List[TurnDynamics] = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    friction_list = []

    for idx, turn in enumerate(turns):
        spk = turn.get("speaker", f"Speaker {idx + 1}")
        text = turn.get("text", "")
        lower = text.lower()

        # Neural prediction if model provided
        neural_data = None
        if dynamics_model is not None:
            input_ids = simple_hash_tokens(text).unsqueeze(0).to(device)
            neural_data = dynamics_model.analyze_utterance(input_ids)

        # Lexical keyword evaluation
        pos_hits = sum(1 for w in POSITIVE_WORDS if w in lower)
        neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lower)

        if pos_hits > neg_hits:
            sent = "positive"
            score = 0.5 + min(0.5, pos_hits * 0.25)
        elif neg_hits > pos_hits:
            sent = "negative"
            score = -0.5 - min(0.5, neg_hits * 0.25)
        else:
            if neural_data:
                sent = neural_data["sentiment"]
                score = 0.6 if sent == "positive" else (-0.6 if sent == "negative" else 0.0)
            else:
                sent = "neutral"
                score = 0.0

        if sent == "positive":
            pos_count += 1
        elif sent == "negative":
            neg_count += 1
        else:
            neu_count += 1

        # Agreement / Contention
        if "disagree" in lower or "not sure" in lower or "risk" in lower or (neural_data and neural_data["agreement"] == "contention"):
            agreement = "contention"
            friction_list.append({
                "turn_index": idx,
                "speaker": spk,
                "text": text,
                "reason": "Diverging viewpoint or flagged risk",
            })
        elif "agree" in lower or "sounds good" in lower or "approved" in lower or (neural_data and neural_data["agreement"] == "consensus"):
            agreement = "consensus"
        else:
            agreement = "neutral"

        engagement = neural_data["engagement_score"] if neural_data else round(min(1.0, len(text.split()) / 25.0), 2)

        analyzed_turns.append(
            TurnDynamics(
                turn_index=idx,
                speaker=spk,
                text=text,
                sentiment=sent,
                sentiment_score=round(score, 2),
                agreement=agreement,
                engagement=engagement,
            )
        )

    total = len(turns)
    pos_pct = round((pos_count / total) * 100.0, 1)
    neg_pct = round((neg_count / total) * 100.0, 1)
    neu_pct = round((neu_count / total) * 100.0, 1)

    overall = "positive" if pos_count > neg_count and pos_count >= total * 0.25 else (
        "negative" if neg_count > pos_count and neg_count >= total * 0.25 else "neutral"
    )

    # Consensus score: Higher positive/consensus turns vs contention
    consensus_score = round(max(10.0, min(100.0, 50.0 + (pos_pct - neg_pct) * 0.6 - (len(friction_list) * 4))), 1)

    return SentimentArc(
        turns=analyzed_turns,
        overall_sentiment=overall,
        sentiment_balance={"positive": pos_pct, "neutral": neu_pct, "negative": neg_pct},
        consensus_score=consensus_score,
        friction_points=friction_list,
    )
