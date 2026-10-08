"""Unit tests for audio parsing, feature extraction, and VAD."""

import io
import pytest
import torch
from src.audio.wav_io import generate_synthetic_audio, read_wav, resample_waveform, write_wav
from src.audio.features import LogMelExtractor, compute_mfcc, create_mel_filterbank
from src.audio.vad import EnergyZCRVAD


def test_wav_read_write_roundtrip() -> None:
    # Generate 1 sec audio at 16kHz
    orig_wav = generate_synthetic_audio(duration_sec=1.0, sample_rate=16000, speaker_id=1)
    buf = io.BytesIO()
    write_wav(buf, orig_wav, sample_rate=16000)

    # Read back
    read_tensor, sr = read_wav(buf, target_sample_rate=16000)
    assert sr == 16000
    assert read_tensor.shape == orig_wav.shape
    # Check numerical proximity within 16-bit quantization
    diff = torch.abs(orig_wav - read_tensor).max().item()
    assert diff < 0.01


def test_wav_read_write_filepath(tmp_path) -> None:
    orig_wav = generate_synthetic_audio(duration_sec=1.0, sample_rate=16000, speaker_id=2)
    wav_path = tmp_path / "test_sample.wav"
    write_wav(wav_path, orig_wav, sample_rate=16000)

    # Read using string and Path
    read_str, sr_str = read_wav(str(wav_path), target_sample_rate=16000)
    read_path, sr_path = read_wav(wav_path, target_sample_rate=16000)

    assert sr_str == 16000 and sr_path == 16000
    assert read_str.shape == orig_wav.shape
    assert read_path.shape == orig_wav.shape


def test_read_wav_webm_format() -> None:
    import subprocess
    orig_wav = generate_synthetic_audio(duration_sec=1.0, sample_rate=16000, speaker_id=1)
    buf = io.BytesIO()
    write_wav(buf, orig_wav, sample_rate=16000)

    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "webm", "-c:a", "libopus", "pipe:1"],
        input=buf.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    webm_bytes = proc.stdout
    recovered, sr = read_wav(webm_bytes, target_sample_rate=16000)
    assert sr == 16000
    assert recovered.shape == (1, 16000)


def test_resample_waveform() -> None:
    wav_8k = torch.randn(1, 8000)
    wav_16k = resample_waveform(wav_8k, orig_sr=8000, target_sr=16000)
    assert wav_16k.shape == (1, 16000)


def test_log_mel_extractor_shapes() -> None:
    extractor = LogMelExtractor(sample_rate=16000, n_mels=80)
    wav = torch.randn(2, 16000 * 2)  # 2 seconds batch of 2
    mel = extractor(wav)
    assert mel.dim() == 3
    assert mel.size(0) == 2
    assert mel.size(1) == 80
    assert mel.size(2) > 0


def test_mfcc_computation() -> None:
    log_mel = torch.randn(2, 80, 100)
    mfcc = compute_mfcc(log_mel, n_mfcc=13)
    assert mfcc.shape == (2, 13, 100)


def test_vad_speech_detection() -> None:
    vad = EnergyZCRVAD(sample_rate=16000)
    # 2 seconds speech + 1 second silence
    speech = generate_synthetic_audio(duration_sec=2.0, sample_rate=16000, amplitude=0.8)
    silence = torch.zeros((1, 16000 * 1))
    audio = torch.cat([speech, silence], dim=-1)

    segments = vad.detect(audio)
    assert len(segments) >= 1
    assert segments[0].start_sec >= 0.0
    assert segments[0].end_sec <= 3.1
