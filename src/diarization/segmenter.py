"""Temporal audio window segmenter for extracting speech slices."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from ..audio.vad import SpeechSegment


@dataclass
class AudioWindow:
    """A sliced window of speech audio for speaker embedding extraction."""
    start_sec: float
    end_sec: float
    waveform: torch.Tensor


class AudioSegmenter:
    """Slices detected speech regions into overlapping fixed-duration windows."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_sec: float = 1.5,
        step_size_sec: float = 0.75,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.win_samples = int(round(window_size_sec * sample_rate))
        self.step_samples = int(round(step_size_sec * sample_rate))

    def segment_speech(
        self,
        waveform: torch.Tensor,
        speech_segments: List[SpeechSegment],
    ) -> List[AudioWindow]:
        """Slices full waveform into uniform windows based on VAD speech segments.

        Args:
            waveform: [1, N] or [N] tensor
            speech_segments: list of detected speech boundaries
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        total_samples = len(waveform)
        windows: List[AudioWindow] = []

        for seg in speech_segments:
            seg_start_sample = max(0, int(round(seg.start_sec * self.sample_rate)))
            seg_end_sample = min(total_samples, int(round(seg.end_sec * self.sample_rate)))
            seg_duration_samples = seg_end_sample - seg_start_sample

            if seg_duration_samples < self.win_samples // 2:
                # If too short, pad symmetrically to min length
                slice_audio = waveform[seg_start_sample:seg_end_sample]
                pad_len = self.win_samples - len(slice_audio)
                if pad_len > 0:
                    slice_audio = torch.nn.functional.pad(slice_audio, (0, pad_len))
                windows.append(
                    AudioWindow(
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        waveform=slice_audio.unsqueeze(0),
                    )
                )
                continue

            curr_start = seg_start_sample
            while curr_start + self.win_samples <= seg_end_sample:
                curr_end = curr_start + self.win_samples
                slice_audio = waveform[curr_start:curr_end]

                windows.append(
                    AudioWindow(
                        start_sec=round(curr_start / self.sample_rate, 3),
                        end_sec=round(curr_end / self.sample_rate, 3),
                        waveform=slice_audio.unsqueeze(0),
                    )
                )
                curr_start += self.step_samples

            # Remaining tail window
            if curr_start < seg_end_sample and (seg_end_sample - curr_start) >= (self.win_samples // 3):
                tail_start = max(0, seg_end_sample - self.win_samples)
                slice_audio = waveform[tail_start:seg_end_sample]
                if len(slice_audio) < self.win_samples:
                    slice_audio = torch.nn.functional.pad(slice_audio, (0, self.win_samples - len(slice_audio)))
                windows.append(
                    AudioWindow(
                        start_sec=round(tail_start / self.sample_rate, 3),
                        end_sec=round(seg_end_sample / self.sample_rate, 3),
                        waveform=slice_audio.unsqueeze(0),
                    )
                )

        return windows
