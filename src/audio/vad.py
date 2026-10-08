"""Dual-threshold Voice Activity Detection (Energy & Zero-Crossing Rate)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch


@dataclass
class SpeechSegment:
    """Represents a detected contiguous segment of speech."""
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float


class EnergyZCRVAD:
    """Voice Activity Detector combining Short-Time Energy (STE) and
    Zero-Crossing Rate (ZCR) with temporal smoothing and hang-over hysteresis.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        energy_threshold: float = 0.015,
        zcr_threshold: float = 0.12,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 300.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(round(sample_rate * frame_length_ms / 1000.0))
        self.hop_len = int(round(sample_rate * hop_length_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.min_speech_frames = int(round(min_speech_duration_ms / hop_length_ms))
        self.min_silence_frames = int(round(min_silence_duration_ms / hop_length_ms))

    def detect(self, waveform: torch.Tensor | np.ndarray) -> List[SpeechSegment]:
        """Detects speech segments in the provided waveform.

        Args:
            waveform: 1D or 2D audio tensor/array

        Returns:
            List of SpeechSegment with start, end, duration, and confidence.
        """
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0)
            signal = waveform.detach().cpu().numpy()
        else:
            signal = np.squeeze(waveform)

        if len(signal) < self.frame_len:
            return []

        # Frame extraction
        num_frames = 1 + (len(signal) - self.frame_len) // self.hop_len
        energies = np.zeros(num_frames, dtype=np.float32)
        zcrs = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_len
            frame = signal[start : start + self.frame_len]

            # Short-Time Root Mean Square (RMS) Energy
            energies[i] = np.sqrt(np.mean(frame ** 2) + 1e-9)

            # Zero-Crossing Rate
            signs = np.sign(frame)
            # Replace 0 with 1 to avoid false zeros
            signs[signs == 0] = 1
            zcr = 0.5 * np.mean(np.abs(signs[1:] - signs[:-1]))
            zcrs[i] = zcr

        # Adaptive background noise estimation (10th percentile energy floor)
        ambient_energy = float(np.percentile(energies, 10)) if num_frames > 0 else 0.005
        effective_energy_thresh = max(self.energy_threshold, ambient_energy * 1.8)

        # Raw frame-level speech decision
        is_speech = (energies > effective_energy_thresh) | (
            (energies > effective_energy_thresh * 0.6) & (zcrs > self.zcr_threshold)
        )

        # Temporal smoothing: fill short silence gaps
        smoothed = np.copy(is_speech)
        silence_count = 0
        gap_start = -1

        for i in range(num_frames):
            if not smoothed[i]:
                if silence_count == 0:
                    gap_start = i
                silence_count += 1
            else:
                if 0 < silence_count < self.min_silence_frames and gap_start != -1:
                    smoothed[gap_start:i] = True  # bridge gap
                silence_count = 0
                gap_start = -1

        # Extract contiguous speech runs
        segments: List[SpeechSegment] = []
        in_speech = False
        speech_start_idx = 0

        for i in range(num_frames):
            if smoothed[i] and not in_speech:
                in_speech = True
                speech_start_idx = i
            elif not smoothed[i] and in_speech:
                in_speech = False
                speech_len = i - speech_start_idx
                if speech_len >= self.min_speech_frames:
                    start_sec = (speech_start_idx * self.hop_len) / self.sample_rate
                    end_sec = ((i * self.hop_len) + self.frame_len) / self.sample_rate
                    conf = float(np.mean(energies[speech_start_idx:i]) / (effective_energy_thresh + 1e-6))
                    conf = min(1.0, max(0.5, conf * 0.5))
                    segments.append(
                        SpeechSegment(
                            start_sec=round(start_sec, 3),
                            end_sec=round(end_sec, 3),
                            duration_sec=round(end_sec - start_sec, 3),
                            confidence=round(conf, 3),
                        )
                    )

        # Trailing segment if signal ends mid-speech
        if in_speech:
            speech_len = num_frames - speech_start_idx
            if speech_len >= self.min_speech_frames:
                start_sec = (speech_start_idx * self.hop_len) / self.sample_rate
                end_sec = len(signal) / self.sample_rate
                segments.append(
                    SpeechSegment(
                        start_sec=round(start_sec, 3),
                        end_sec=round(end_sec, 3),
                        duration_sec=round(end_sec - start_sec, 3),
                        confidence=0.85,
                    )
                )

        return segments
