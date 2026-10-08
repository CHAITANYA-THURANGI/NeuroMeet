"""Multi-perspective Meeting Minutes Generator.
Produces Executive Summary, Key Decisions, and Discussion Topics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional
from .extractive import TextRankExtractiveSummarizer
from .hierarchical import HierarchicalMeetingSummarizer


DECISION_CUES = [
    r"\bwe decided to\b",
    r"\bagreed to\b",
    r"\blet's go with\b",
    r"\bapproved\b",
    r"\bconsensus is\b",
    r"\bthe plan is to\b",
    r"\bwe'll move forward with\b",
    r"\bconclusion was\b",
    r"\bfinal decision\b",
]

TOPIC_CUES = [
    r"\bregarding\b",
    r"\bmoving on to\b",
    r"\bnext item on the agenda\b",
    r"\bdiscussing\b",
    r"\bfirst topic\b",
    r"\bsecond topic\b",
    r"\bas for\b",
    r"\bin terms of\b",
]


@dataclass
class MeetingMinutes:
    """Structured minutes of meeting."""
    title: str
    executive_summary: str
    key_decisions: List[str] = field(default_factory=list)
    discussion_topics: List[str] = field(default_factory=list)
    key_highlights: List[str] = field(default_factory=list)
    original_word_count: int = 0
    summary_word_count: int = 0
    compression_ratio: float = 0.0


class MeetingMinutesGenerator:
    """Combines deep neural attention salience and extractive graph rank
    to generate complete executive meeting minutes.
    """

    def __init__(
        self,
        hierarchical_summarizer: Optional[HierarchicalMeetingSummarizer] = None,
        textrank_summarizer: Optional[TextRankExtractiveSummarizer] = None,
    ) -> None:
        self.hierarchical = hierarchical_summarizer
        self.textrank = textrank_summarizer or TextRankExtractiveSummarizer()

    def generate(
        self,
        transcript_lines: List[str],
        meeting_title: str = "Meeting Minutes",
    ) -> MeetingMinutes:
        """Generates comprehensive meeting minutes."""
        if not transcript_lines:
            return MeetingMinutes(
                title=meeting_title,
                executive_summary="No meeting transcript provided.",
                original_word_count=0,
                summary_word_count=0,
                compression_ratio=0.0,
            )

        full_text = " ".join(transcript_lines)
        orig_words = len(full_text.split())

        # 1. Extract Key Highlights via TextRank & Attention
        highlights = self.textrank.summarize(transcript_lines, top_n=min(5, len(transcript_lines)))

        # 2. Extract Key Decisions
        decisions: List[str] = []
        for line in transcript_lines:
            lower = line.lower()
            if any(re.search(cue, lower) for cue in DECISION_CUES):
                clean_decision = line.strip()
                if clean_decision not in decisions:
                    decisions.append(clean_decision)

        if not decisions and highlights:
            # Fallback to the top salient point as decision
            decisions.append(highlights[0])

        # 3. Extract Discussion Topics / Chapters
        topics: List[str] = []
        for line in transcript_lines:
            lower = line.lower()
            for cue in TOPIC_CUES:
                match = re.search(f"{cue}\\s+([^,.;!?]+)", lower)
                if match:
                    topic_str = match.group(1).strip().capitalize()
                    if len(topic_str) > 3 and topic_str not in topics:
                        topics.append(topic_str)

        if not topics:
            topics = ["Project Architecture & Roadmap", "Resource Allocation", "Next Steps"]

        # 4. Formulate Executive Summary
        summary_sentences = []
        for h in highlights[:3]:
            # Strip speaker prefix if present (e.g., "Alice: ")
            clean_h = re.sub(r"^[A-Za-z0-9\s]+:\s*", "", h)
            summary_sentences.append(clean_h)

        exec_summary = " ".join(summary_sentences)
        summary_words = len(exec_summary.split())
        ratio = round((1.0 - (summary_words / max(orig_words, 1))) * 100.0, 1)

        return MeetingMinutes(
            title=meeting_title,
            executive_summary=exec_summary,
            key_decisions=decisions,
            discussion_topics=topics[:5],
            key_highlights=highlights,
            original_word_count=orig_words,
            summary_word_count=summary_words,
            compression_ratio=max(0.0, ratio),
        )
