"""Speaker participation, talk-time distribution, and dominance scoring."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class ParticipationAnalytics:
    """Metrics assessing conversational equity and speaker contributions."""
    total_duration_sec: float
    total_turns: int
    talk_time_by_speaker: Dict[str, float]
    talk_time_percentages: Dict[str, float]
    turns_by_speaker: Dict[str, int]
    words_by_speaker: Dict[str, int]
    dominance_index: float  # Gini coefficient of talk-time: 0.0 (equal) to 1.0 (monopoly)
    dominant_speaker: str


def compute_gini(values: np.ndarray) -> float:
    """Computes Gini coefficient of inequality for an array of values."""
    if len(values) <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_participation_metrics(
    turns: List[Dict[str, any]],
) -> ParticipationAnalytics:
    """Calculates granular participation statistics from meeting turns.

    Args:
        turns: List of dicts, each with "speaker", "duration_sec" (or estimated from text), "text".
    """
    if not turns:
        return ParticipationAnalytics(
            total_duration_sec=0.0,
            total_turns=0,
            talk_time_by_speaker={},
            talk_time_percentages={},
            turns_by_speaker={},
            words_by_speaker={},
            dominance_index=0.0,
            dominant_speaker="None",
        )

    talk_time: Dict[str, float] = {}
    turn_counts: Dict[str, int] = {}
    word_counts: Dict[str, int] = {}
    total_sec = 0.0

    for turn in turns:
        spk = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        words = len(text.split())

        # If explicit duration is missing, estimate duration assuming ~2.5 words per second
        dur = float(turn.get("duration_sec", max(1.0, words / 2.5)))

        talk_time[spk] = round(talk_time.get(spk, 0.0) + dur, 2)
        turn_counts[spk] = turn_counts.get(spk, 0) + 1
        word_counts[spk] = word_counts.get(spk, 0) + words
        total_sec += dur

    total_sec = max(0.1, total_sec)
    percentages = {spk: round((t / total_sec) * 100.0, 1) for spk, t in talk_time.items()}

    # Compute Gini coefficient
    times = np.array(list(talk_time.values()), dtype=np.float32)
    gini = compute_gini(times)

    dominant = max(talk_time.items(), key=lambda x: x[1])[0] if talk_time else "None"

    return ParticipationAnalytics(
        total_duration_sec=round(total_sec, 2),
        total_turns=len(turns),
        talk_time_by_speaker=talk_time,
        talk_time_percentages=percentages,
        turns_by_speaker=turn_counts,
        words_by_speaker=word_counts,
        dominance_index=round(gini, 3),
        dominant_speaker=dominant,
    )
