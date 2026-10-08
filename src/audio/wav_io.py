"""WAV audio file parser, writer, resampler, and synthetic generator."""

from __future__ import annotations
import io
import math
import struct
import wave
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch


def read_wav(
    source: Union[str, Path, bytes, io.BytesIO],
    target_sample_rate: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """Reads a WAV audio file or byte buffer into a normalized float32 tensor [-1, 1].
    Converts multi-channel to mono (average) and resamples to target_sample_rate if needed.

    Returns:
        waveform: Shape [1, num_samples], float32 in [-1.0, 1.0]
        sample_rate: int
    """
    if isinstance(source, (str, Path)):
        source_path = str(source)
    elif isinstance(source, bytes):
        wav_file = wave.open(io.BytesIO(source), "rb")
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        wav_file = wave.open(source, "rb")
    elif isinstance(source, torch.Tensor):
        if source.dim() == 1:
            source = source.unsqueeze(0)
        return source, target_sample_rate
    else:
        raise ValueError(f"Unsupported source type for WAV reading: {type(source)}")

    with wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        src_sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        frames_bytes = wav_file.readframes(n_frames)

    # Convert bytes based on bit-depth
    if sampwidth == 2:  # 16-bit PCM
        samples = np.frombuffer(frames_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:  # 8-bit unsigned PCM
        samples = (np.frombuffer(frames_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:  # 32-bit int or float
        try:
            samples = np.frombuffer(frames_bytes, dtype=np.float32)
        except Exception:
            samples = np.frombuffer(frames_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        # Fallback manual unpack 24-bit PCM
        raw_bytes = list(frames_bytes)
        n_samples = len(raw_bytes) // 3
        samples = []
        for i in range(n_samples):
            b = raw_bytes[i * 3 : (i + 1) * 3]
            val = int.from_bytes(b, byteorder="little", signed=True)
            samples.append(val / 8388608.0)
        samples = np.array(samples, dtype=np.float32)

    # Reshape channels: [num_frames, n_channels]
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        # Downmix to mono: average across channels
        samples = samples.mean(axis=1)

    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)  # [1, N]

    if src_sr != target_sample_rate and target_sample_rate > 0:
        waveform = resample_waveform(waveform, src_sr, target_sample_rate)
        src_sr = target_sample_rate

    return waveform, src_sr


def write_wav(
    destination: Union[str, Path, io.BytesIO],
    waveform: torch.Tensor,
    sample_rate: int = 16000,
) -> None:
    """Writes a float32 waveform tensor [1, N] or [N] into a 16-bit PCM WAV file."""
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    # Clamp and convert to 16-bit integer PCM
    samples = waveform.detach().cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    int16_samples = (samples * 32767.0).astype(np.int16)

    if isinstance(destination, (str, Path)):
        dest_path = str(destination)
        wav_file = wave.open(dest_path, "wb")
    else:
        wav_file = wave.open(destination, "wb")

    with wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_samples.tobytes())


def resample_waveform(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resamples a 1D or 2D waveform tensor using linear/polyphase interpolation.
    Pure PyTorch implementation with zero external audio dependencies.
    """
    if orig_sr == target_sr:
        return waveform

    orig_dim = waveform.dim()
    if orig_dim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    elif orig_dim == 2:
        waveform = waveform.unsqueeze(1)  # [B, 1, N]

    batch_size, channels, orig_len = waveform.shape
    target_len = int(round(orig_len * (target_sr / orig_sr)))

    if target_len <= 0:
        target_len = 1

    # Linear interpolation along time axis
    resampled = torch.nn.functional.interpolate(
        waveform,
        size=target_len,
        mode="linear",
        align_corners=False,
    )

    if orig_dim == 1:
        return resampled.squeeze(0).squeeze(0)
    elif orig_dim == 2:
        return resampled.squeeze(1)
    return resampled


def generate_synthetic_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    speaker_id: int = 1,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Generates synthetic multi-harmonic acoustic signals mimicking human speech
    formants and fundamental frequency for test benches and demos.
    Different speaker_id produces distinct formant and pitch contours.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

    # Speaker-specific fundamental pitch (F0) & formant frequencies
    pitch_base = 120.0 if speaker_id % 2 == 1 else 210.0  # Male vs Female pitch
    f1 = 500.0 + (speaker_id * 35.0)
    f2 = 1500.0 + (speaker_id * 70.0)
    f3 = 2500.0 + (speaker_id * 50.0)

    # Modulating speech envelope (syllable bursts of ~4 Hz speech rhythm)
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))
    # Periodic silence pauses between utterances
    pause_mask = (np.sin(2.0 * np.pi * 0.5 * t) > -0.7).astype(np.float32)

    # Harmonic acoustic stack
    harmonics = (
        np.sin(2.0 * np.pi * pitch_base * t) * 0.5
        + np.sin(2.0 * np.pi * f1 * t) * 0.3
        + np.sin(2.0 * np.pi * f2 * t) * 0.15
        + np.sin(2.0 * np.pi * f3 * t) * 0.05
    )

    # Add gentle pink/gaussian background noise
    noise = np.random.normal(0, 0.008, num_samples).astype(np.float32)

    signal = (harmonics * envelope * pause_mask + noise) * amplitude
    signal = np.clip(signal, -1.0, 1.0)

    return torch.from_numpy(signal).unsqueeze(0)  # [1, N]
