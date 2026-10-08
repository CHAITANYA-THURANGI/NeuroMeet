"""Unit tests for FastAPI REST endpoints using TestClient."""

import io
from fastapi.testclient import TestClient
from api.main import app
from src.audio.wav_io import generate_synthetic_audio, write_wav


client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


def test_samples_endpoint() -> None:
    resp = client.get("/api/v1/meetings/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) >= 4
    sample_ids = [s["id"] for s in samples]
    assert "sprint_planning_42" in sample_ids or "sprint_planning" in sample_ids


def test_sample_scenario_execution() -> None:
    resp = client.get("/api/v1/meetings/sample/sprint_planning")
    assert resp.status_code == 200
    data = resp.json()
    assert "Sprint 42" in data["title"]
    assert len(data["turns"]) == 8
    assert len(data["action_items"]) >= 1


def test_process_transcript_endpoint() -> None:
    payload = {
        "title": "API Review",
        "transcript": "Alex: We decided to deploy the service.\nBob: I will configure the monitoring by Friday."
    }
    resp = client.post("/api/v1/meetings/process-transcript", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "API Review"
    assert len(data["turns"]) == 2
    assert len(data["action_items"]) >= 1


def test_qa_endpoint() -> None:
    payload = {
        "query": "Who will configure monitoring?",
        "turns": [
            {"speaker": "Bob", "text": "I will configure the monitoring by Friday.", "turn_index": 0}
        ]
    }
    resp = client.post("/api/v1/meetings/qa", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Bob" in data["answer"] or (len(data["citations"]) > 0 and data["citations"][0]["speaker"] == "Bob")


def test_export_endpoint() -> None:
    meeting_data = {
        "title": "Strategy Sync",
        "turns": [{"speaker": "Alice", "text": "We approved the plan."}],
        "minutes": {"executive_summary": "Approved plan.", "key_decisions": ["Approved plan."], "discussion_topics": ["Strategy"]},
        "action_items": [],
        "health": {"overall_score": 85, "grade": "A", "recommendations": []}
    }
    resp = client.post("/api/v1/meetings/export", json={"meeting_data": meeting_data, "format": "markdown"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "markdown"
    assert "Strategy Sync" in data["content"]


def test_process_audio_endpoint_wav() -> None:
    wav = generate_synthetic_audio(duration_sec=1.0, sample_rate=16000)
    buf = io.BytesIO()
    write_wav(buf, wav, sample_rate=16000)
    buf.seek(0)

    files = {"file": ("test_meeting.wav", buf.getvalue(), "audio/wav")}
    data = {"title": "Synthetic Audio Meeting"}
    resp = client.post("/api/v1/meetings/process-audio", files=files, data=data)
    assert resp.status_code == 200
    res_json = resp.json()
    assert res_json["title"] == "Synthetic Audio Meeting"
    assert len(res_json["turns"]) >= 1


def test_process_audio_endpoint_webm() -> None:
    import subprocess
    wav = generate_synthetic_audio(duration_sec=1.0, sample_rate=16000)
    buf = io.BytesIO()
    write_wav(buf, wav, sample_rate=16000)

    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "webm", "-c:a", "libopus", "pipe:1"],
        input=buf.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    files = {"file": ("recording.webm", proc.stdout, "audio/webm")}
    data = {"title": "Live WebM Recording"}
    resp = client.post("/api/v1/meetings/process-audio", files=files, data=data)
    assert resp.status_code == 200
    res_json = resp.json()
    assert res_json["title"] == "Live WebM Recording"
    assert len(res_json["turns"]) >= 1
