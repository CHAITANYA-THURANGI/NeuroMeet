# NeuroMeet REST API Specification

Base URL: `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`  
ReDoc: `http://127.0.0.1:8000/redoc`

---

## 1. Health & System Status

### `GET /health` or `GET /api/v1/health`
Checks runtime health, PyTorch CUDA support, and model statuses.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "cuda_available": false,
  "models_loaded": true
}
```

---

## 2. Meeting Analysis

### `POST /api/v1/meetings/process-transcript`
Analyzes a multi-turn transcript through the complete deep intelligence suite.

**Request Body:**
```json
{
  "title": "Sprint 42 Review",
  "transcript": "Alex: Let's discuss payment integration.\nMaya: I will finalize the OpenAPI contracts by tomorrow EOD.\nLeo: Sounds good, I will connect the UI components by Thursday."
}
```

**Response (200 OK):**
```json
{
  "title": "Sprint 42 Review",
  "total_duration_sec": 24.5,
  "turns": [ ... ],
  "minutes": {
    "title": "Sprint 42 Review",
    "executive_summary": "Alex reviewed payment integration milestones...",
    "key_decisions": [ ... ],
    "discussion_topics": [ "Payment Integration", "API Synchronization" ],
    "compression_ratio": 72.4
  },
  "action_items": [
    {
      "task": "finalize the OpenAPI contracts",
      "assignee": "Maya",
      "deadline": "Tomorrow EOD",
      "priority": "high",
      "confidence": 0.94
    }
  ],
  "participation": {
    "total_duration_sec": 24.5,
    "total_turns": 3,
    "talk_time_by_speaker": { "Alex": 6.2, "Maya": 10.4, "Leo": 7.9 },
    "talk_time_percentages": { "Alex": 25.3, "Maya": 42.4, "Leo": 32.3 },
    "dominance_index": 0.12,
    "dominant_speaker": "Maya"
  },
  "sentiment": {
    "overall_sentiment": "positive",
    "consensus_score": 92.0,
    "friction_points": []
  },
  "health": {
    "overall_score": 91,
    "grade": "A+",
    "category": "Outstanding & High-Impact",
    "recommendations": [ ... ]
  }
}
```

---

### `POST /api/v1/meetings/process-audio`
Uploads a WAV or MP3 audio file. Performs VAD, SpeakerNet diarization, SpeechCTC transcription, and complete meeting intelligence.

**Request:** `multipart/form-data`
- `file`: binary WAV audio file
- `title`: string (optional)

---

### `POST /api/v1/meetings/qa`
Answers natural language questions about the meeting with cited transcript turns.

**Request Body:**
```json
{
  "query": "Who is delivering the OpenAPI schema?",
  "scenario_id": "sprint_planning"
}
```

**Response (200 OK):**
```json
{
  "query": "Who is delivering the OpenAPI schema?",
  "answer": "According to Maya (Backend Lead), \"Yes, I will finalize the OpenAPI schema and fix the webhook retry logic by tomorrow EOD.\"",
  "confidence": 0.95,
  "citations": [
    {
      "speaker": "Maya (Backend Lead)",
      "text": "Yes, I will finalize the OpenAPI schema and fix the webhook retry logic by tomorrow EOD.",
      "turn_index": 4,
      "score": 0.82
    }
  ]
}
```

---

### `POST /api/v1/meetings/export`
Exports meeting minutes to Markdown, Printable HTML, or JSON.

**Request Body:**
```json
{
  "meeting_data": { ... },
  "format": "html"
}
```
