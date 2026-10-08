"""Holistic Meeting Health Score and AI Meeting Coach recommendations."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .participation import ParticipationAnalytics
from .sentiment_flow import SentimentArc
from ..action_items.extractor import ActionItem
from ..summarization.abstractive import MeetingMinutes


@dataclass
class MeetingHealthReport:
    """Consolidated assessment of meeting effectiveness."""
    overall_score: int  # 0 to 100
    grade: str          # A+, A, B, C, D
    category: str       # e.g., "Highly Productive & Collaborative"
    participation_subscore: int  # 0 - 25
    actionability_subscore: int  # 0 - 35
    sentiment_subscore: int      # 0 - 25
    efficiency_subscore: int     # 0 - 15
    recommendations: List[str] = field(default_factory=list)


def compute_meeting_health_score(
    participation: ParticipationAnalytics,
    sentiment: SentimentArc,
    action_items: List[ActionItem],
    minutes: MeetingMinutes,
) -> MeetingHealthReport:
    """Computes a 0-100 meeting health score with automated coaching tips."""
    # 1. Participation subscore (Max 25): Gini 0.0 -> 25pts, Gini 0.8 -> 5pts
    gini = participation.dominance_index
    part_score = int(round(max(5.0, 25.0 * (1.0 - gini))))

    # 2. Actionability subscore (Max 35): Based on presence of action items & decisions
    n_actions = len(action_items)
    n_decisions = len(minutes.key_decisions)
    act_raw = min(20.0, n_actions * 6.0) + min(15.0, n_decisions * 5.0)
    act_score = int(round(act_raw))

    # 3. Sentiment subscore (Max 25): Based on consensus score
    sent_score = int(round((sentiment.consensus_score / 100.0) * 25.0))

    # 4. Efficiency subscore (Max 15): Summary compression and turn volume
    eff_score = 12 if participation.total_turns > 3 else 8

    total = int(round(part_score + act_score + sent_score + eff_score))
    total = max(10, min(100, total))

    if total >= 90:
        grade = "A+"
        category = "Outstanding & High-Impact"
    elif total >= 80:
        grade = "A"
        category = "Highly Productive & Collaborative"
    elif total >= 70:
        grade = "B"
        category = "Effective with Minor Action Gaps"
    elif total >= 60:
        grade = "C"
        category = "Average — Needs Clearer Ownership"
    else:
        grade = "D"
        category = "Low Engagement / Unclear Next Steps"

    recs: List[str] = []
    if gini > 0.45:
        recs.append(f"Speaker '{participation.dominant_speaker}' accounted for a majority of talk-time. Consider encouraging other participants to speak.")
    if n_actions == 0:
        recs.append("No explicit action items were detected. Ensure every agenda topic concludes with clear owners and due dates.")
    if len(sentiment.friction_points) > 2:
        recs.append("Multiple points of contention were raised. Schedule a targeted follow-up to address outstanding risks.")
    if n_actions >= 3 and gini <= 0.35:
        recs.append("Excellent meeting hygiene: balanced conversation with concrete actionable commitments!")

    if not recs:
        recs.append("Meeting was concise, well-moderated, and achieved clear decision outcomes.")

    return MeetingHealthReport(
        overall_score=total,
        grade=grade,
        category=category,
        participation_subscore=part_score,
        actionability_subscore=act_score,
        sentiment_subscore=sent_score,
        efficiency_subscore=eff_score,
        recommendations=recs,
    )
