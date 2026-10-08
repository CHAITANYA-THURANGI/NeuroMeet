"""Pydantic request and response schemas for NeuroMeet REST API."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TurnSchema(BaseModel):
    turn_index: int = Field(default=0, description="Sequential index of dialogue turn")
    speaker: str = Field(..., description="Speaker identifier or name")
    text: str = Field(..., description="Transcribed spoken utterance")
    language: Optional[str] = Field(default="en", description="Detected language code (en, hi, te, etc.)")
    translation: Optional[str] = Field(default=None, description="English translation if turn is non-English")
    start_sec: Optional[float] = Field(default=None, description="Start timestamp in seconds")
    end_sec: Optional[float] = Field(default=None, description="End timestamp in seconds")
    duration_sec: Optional[float] = Field(default=None, description="Turn duration in seconds")


class ActionItemSchema(BaseModel):
    task: str = Field(..., description="Actionable task commitment")
    assignee: str = Field(default="Unassigned", description="Assigned individual or team")
    deadline: str = Field(default="Next sprint", description="Target completion deadline")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, urgent")
    confidence: float = Field(default=0.85, description="Model confidence in extraction")
    source_utterance: Optional[str] = Field(default=None, description="Original speaker quote")


class MinutesSchema(BaseModel):
    title: str = Field(..., description="Meeting title")
    executive_summary: str = Field(..., description="High-level executive overview")
    key_decisions: List[str] = Field(default_factory=list, description="Decisions agreed upon")
    discussion_topics: List[str] = Field(default_factory=list, description="Agenda topics & chapters")
    key_highlights: List[str] = Field(default_factory=list, description="Top salient takeaways")
    chapters: List[Dict[str, Any]] = Field(default_factory=list, description="Long-meeting structured chapter breakdowns")
    original_word_count: int = Field(default=0)
    summary_word_count: int = Field(default=0)
    compression_ratio: float = Field(default=0.0)


class ParticipationSchema(BaseModel):
    total_duration_sec: float
    total_turns: int
    talk_time_by_speaker: Dict[str, float]
    talk_time_percentages: Dict[str, float]
    turns_by_speaker: Dict[str, int]
    words_by_speaker: Dict[str, int]
    dominance_index: float
    dominant_speaker: str


class TurnDynamicsSchema(BaseModel):
    turn_index: int
    speaker: str
    text: str
    sentiment: str
    sentiment_score: float
    agreement: str
    engagement: float


class SentimentSchema(BaseModel):
    overall_sentiment: str
    sentiment_balance: Dict[str, float]
    consensus_score: float
    friction_points: List[Dict[str, Any]]
    turns: List[TurnDynamicsSchema]


class HealthSchema(BaseModel):
    overall_score: int
    grade: str
    category: str
    participation_subscore: int
    actionability_subscore: int
    sentiment_subscore: int
    efficiency_subscore: int
    recommendations: List[str]


class MeetingAnalysisResponse(BaseModel):
    title: str
    total_duration_sec: float
    turns: List[TurnSchema]
    minutes: MinutesSchema
    action_items: List[ActionItemSchema]
    participation: ParticipationSchema
    sentiment: SentimentSchema
    health: HealthSchema
    detected_languages: List[str] = Field(default_factory=lambda: ["en"], description="Languages spoken across the meeting")


class ProcessTranscriptRequest(BaseModel):
    title: Optional[str] = Field(default="Executive Meeting", description="Meeting Title")
    transcript: Optional[str] = Field(default=None, description="Raw multiline transcript text")
    turns: Optional[List[TurnSchema]] = Field(default=None, description="Pre-structured list of turns")


class SummarizeRequest(BaseModel):
    transcript: str = Field(..., description="Raw text transcript to summarize")
    title: Optional[str] = Field(default="Meeting Summary")


class SummarizeResponse(BaseModel):
    title: str
    executive_summary: str
    key_decisions: List[str]
    discussion_topics: List[str]
    chapters: List[Dict[str, Any]] = Field(default_factory=list)
    compression_ratio: float


class ActionItemExtractRequest(BaseModel):
    transcript: Optional[str] = Field(default=None)
    turns: Optional[List[TurnSchema]] = Field(default=None)


class ActionItemExtractResponse(BaseModel):
    count: int
    action_items: List[ActionItemSchema]
    markdown_table: str


class QARequest(BaseModel):
    query: str = Field(..., description="Natural language question about the meeting")
    turns: Optional[List[TurnSchema]] = Field(default=None, description="Dialogue turns context")
    scenario_id: Optional[str] = Field(default=None, description="Pre-loaded scenario ID to query")


class CitationSchema(BaseModel):
    speaker: str
    text: str
    turn_index: int
    score: float
    timestamp_sec: Optional[float] = None


class QAResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    citations: List[CitationSchema]


class ExportRequest(BaseModel):
    meeting_data: Dict[str, Any]
    format: str = Field(default="markdown", description="Export format: markdown | html | json")


class ExportResponse(BaseModel):
    format: str
    content: str
    filename: str


class SystemHealthResponse(BaseModel):
    status: str
    version: str
    device: str
    cuda_available: bool
    models_loaded: bool
