"""Main FastAPI application for NeuroMeet AI."""

from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from api.schemas import (
    ActionItemExtractRequest,
    ActionItemExtractResponse,
    ActionItemSchema,
    CitationSchema,
    ExportRequest,
    ExportResponse,
    MeetingAnalysisResponse,
    ProcessTranscriptRequest,
    QARequest,
    QAResponse,
    SummarizeRequest,
    SummarizeResponse,
    SystemHealthResponse,
    TurnSchema,
)
from api.services.meeting_service import MeetingService
from api.services.model_registry import ModelRegistry, get_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("neuromeet.api")

cfg = load_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuroMeet AI server...")
    # Pre-warm pipeline
    _ = get_pipeline()
    logger.info("NeuroMeet pipeline ready.")
    yield


app = FastAPI(
    title="NeuroMeet API",
    description="Enterprise Deep Learning Framework for Meeting Speech Diarization, Summarization, and Action Items.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for local web studio and browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

meeting_service = MeetingService()


@app.get("/health", response_model=SystemHealthResponse, tags=["Health"])
@app.get("/api/v1/health", response_model=SystemHealthResponse, tags=["Health"])
async def check_health() -> SystemHealthResponse:
    """Returns runtime health, PyTorch CUDA status, and model states."""
    import torch
    dev = ModelRegistry.get_device()
    return SystemHealthResponse(
        status="healthy",
        version="1.0.0",
        device=dev,
        cuda_available=torch.cuda.is_available(),
        models_loaded=True,
    )


@app.post("/api/v1/meetings/process-transcript", response_model=MeetingAnalysisResponse, tags=["Meeting Analysis"])
async def process_transcript(req: ProcessTranscriptRequest) -> MeetingAnalysisResponse:
    """Analyzes a multi-turn meeting transcript (text or turn objects) through the full neural suite."""
    pipeline = get_pipeline()

    if req.turns:
        turns_data = [t.model_dump() for t in req.turns]
        result = pipeline.process_transcript(turns_data, title=req.title or "Executive Meeting")
    elif req.transcript:
        result = pipeline.process_transcript(req.transcript, title=req.title or "Executive Meeting")
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'transcript' or 'turns'.")

    return MeetingAnalysisResponse(**result.to_dict())


@app.post("/api/v1/meetings/process-audio", response_model=MeetingAnalysisResponse, tags=["Meeting Analysis"])
async def process_audio(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
) -> MeetingAnalysisResponse:
    """Processes uploaded audio (WAV, MP3, WebM) through VAD, Diarization, Multilingual ASR, and Meeting Intelligence."""
    pipeline = get_pipeline()
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        effective_title = title or file.filename or "Audio Meeting Analysis"
        result = pipeline.process_audio(contents, title=effective_title, language=language)
        return MeetingAnalysisResponse(**result.to_dict())
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failure: {str(e)}")


@app.post("/api/v1/meetings/summarize", response_model=SummarizeResponse, tags=["NLP Engines"])
async def summarize_transcript(req: SummarizeRequest) -> SummarizeResponse:
    """Extracts executive overview, decisions, and topics from a raw meeting transcript."""
    pipeline = get_pipeline()
    result = pipeline.process_transcript(req.transcript, title=req.title or "Meeting Summary")
    return SummarizeResponse(
        title=result.title,
        executive_summary=result.minutes.executive_summary,
        key_decisions=result.minutes.key_decisions,
        discussion_topics=result.minutes.discussion_topics,
        compression_ratio=result.minutes.compression_ratio,
    )


@app.post("/api/v1/meetings/action-items", response_model=ActionItemExtractResponse, tags=["NLP Engines"])
async def extract_action_items(req: ActionItemExtractRequest) -> ActionItemExtractResponse:
    """Identifies tasks, assignees, deadlines, and priorities from meeting text."""
    pipeline = get_pipeline()
    if req.turns:
        turns_data = [t.model_dump() for t in req.turns]
        result = pipeline.process_transcript(turns_data)
    elif req.transcript:
        result = pipeline.process_transcript(req.transcript)
    else:
        raise HTTPException(status_code=400, detail="Provide transcript or turns.")

    from src.action_items.formatter import format_action_items_markdown
    md_table = format_action_items_markdown(result.action_items)

    action_schemas = [ActionItemSchema(**a.__dict__) for a in result.action_items]
    return ActionItemExtractResponse(
        count=len(result.action_items),
        action_items=action_schemas,
        markdown_table=md_table,
    )


@app.post("/api/v1/meetings/qa", response_model=QAResponse, tags=["Meeting Q&A"])
async def answer_question(req: QARequest) -> QAResponse:
    """Answers factual and assignment questions grounded in the meeting transcript."""
    pipeline = get_pipeline()

    if req.turns:
        turns = [t.model_dump() for t in req.turns]
    elif req.scenario_id:
        sc = meeting_service.get_scenario_by_id(req.scenario_id)
        if not sc:
            raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found.")
        turns = sc.turns
    else:
        raise HTTPException(status_code=400, detail="Must provide either turns or scenario_id.")

    qa_res = pipeline.qa_engine.answer_question(req.query, turns)
    citations = [
        CitationSchema(
            speaker=c.speaker,
            text=c.text,
            turn_index=c.turn_index,
            score=c.score,
            timestamp_sec=c.timestamp_sec,
        )
        for c in qa_res.citations
    ]

    return QAResponse(
        query=qa_res.query,
        answer=qa_res.answer,
        confidence=qa_res.confidence,
        citations=citations,
    )


@app.get("/api/v1/meetings/samples", tags=["Samples & Demos"])
async def get_sample_scenarios() -> List[Dict[str, Any]]:
    """Returns list of pre-packaged corporate meeting scenarios."""
    return meeting_service.get_sample_scenarios_list()


@app.get("/api/v1/meetings/sample/{scenario_id}", response_model=MeetingAnalysisResponse, tags=["Samples & Demos"])
async def load_sample_scenario(scenario_id: str) -> MeetingAnalysisResponse:
    """Executes end-to-end analysis on one of the pre-packaged corporate meeting scenarios."""
    result = meeting_service.process_scenario_by_id(scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")
    return MeetingAnalysisResponse(**result.to_dict())


@app.post("/api/v1/meetings/export", response_model=ExportResponse, tags=["Export"])
async def export_meeting(req: ExportRequest) -> ExportResponse:
    """Exports processed meeting intelligence to Markdown, Printable HTML, or JSON."""
    title = req.meeting_data.get("title", "Meeting_Minutes").replace(" ", "_")
    fmt = req.format.lower()

    if fmt == "html":
        content = meeting_service.generate_html_export(req.meeting_data)
        filename = f"{title}.html"
    elif fmt == "json":
        content = json.dumps(req.meeting_data, indent=2)
        filename = f"{title}.json"
    else:
        # Default Markdown
        pipeline = get_pipeline()
        turns = req.meeting_data.get("turns", [])
        res = pipeline.process_transcript(turns, title=req.meeting_data.get("title", "Meeting Minutes"))
        content = res.to_markdown()
        filename = f"{title}.md"

    return ExportResponse(format=fmt, content=content, filename=filename)


# Mount static web UI if directory exists
web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index() -> HTMLResponse:
        index_file = web_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>NeuroMeet Web Studio Loading...</h1>")
