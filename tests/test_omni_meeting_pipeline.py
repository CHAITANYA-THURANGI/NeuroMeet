"""Unit tests for OmniMeeting Pipeline end-to-end execution."""

from src.audio.wav_io import generate_synthetic_audio
from src.pipeline.omni_meeting import OmniMeetingPipeline


def test_omni_pipeline_transcript_run() -> None:
    pipeline = OmniMeetingPipeline()
    transcript = """Alex: Let's discuss Sprint 42 planning today.
Maya: I will fix the webhook retry issue by tomorrow EOD.
Leo: I agree, let's target Tuesday for the launch."""

    result = pipeline.process_transcript(transcript, title="Sprint Review")
    assert result.title == "Sprint Review"
    assert len(result.turns) == 3
    assert len(result.action_items) >= 1
    assert result.health.overall_score > 0
    assert "Sprint Review" in result.to_markdown()


def test_omni_pipeline_audio_run() -> None:
    pipeline = OmniMeetingPipeline()
    audio = generate_synthetic_audio(duration_sec=2.5, sample_rate=16000)
    result = pipeline.process_audio(audio, title="Audio Demo")
    assert result.title == "Audio Demo"
    assert len(result.turns) >= 1
    assert result.total_duration_sec > 0
