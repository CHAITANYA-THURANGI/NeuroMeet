"""End-to-End Speaker Diarization Pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
from ..audio.features import LogMelExtractor
from ..audio.vad import EnergyZCRVAD, SpeechSegment
from ..models.speaker_net import SpeakerNet
from .clustering import AgglomerativeSpeakerClusterer, SpectralSpeakerClusterer
from .segmenter import AudioSegmenter, AudioWindow


@dataclass
class SpeakerTurn:
    """Represents an attributed speech turn in the meeting."""
    speaker_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float


class SpeakerDiarizer:
    """Full-pipeline neural speaker diarization:
    Acoustic Signal -> VAD -> Window Slicing -> SpeakerNet -> Spectral Clustering -> Turn Fusion
    """

    def __init__(
        self,
        speaker_net: SpeakerNet,
        sample_rate: int = 16000,
        min_speakers: int = 1,
        max_speakers: int = 6,
        window_size_sec: float = 1.5,
        step_size_sec: float = 0.75,
        clustering_method: str = "spectral",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.speaker_net = speaker_net.to(self.device).eval()
        self.sample_rate = sample_rate

        self.vad = EnergyZCRVAD(sample_rate=sample_rate)
        self.segmenter = AudioSegmenter(
            sample_rate=sample_rate,
            window_size_sec=window_size_sec,
            step_size_sec=step_size_sec,
        )
        self.feature_extractor = LogMelExtractor(sample_rate=sample_rate).to(self.device)

        if clustering_method == "spectral":
            self.clusterer = SpectralSpeakerClusterer(min_speakers=min_speakers, max_speakers=max_speakers)
        else:
            self.clusterer = AgglomerativeSpeakerClusterer(threshold=0.65)

    @torch.no_grad()
    def diarize(
        self,
        waveform: torch.Tensor,
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerTurn]:
        """Performs speaker diarization on raw audio.

        Args:
            waveform: [1, N] or [N] tensor
            num_speakers: Optional known number of speakers
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # 1. Voice Activity Detection
        speech_segs = self.vad.detect(waveform)
        if not speech_segs:
            return []

        # 2. Window segmentation
        windows = self.segmenter.segment_speech(waveform, speech_segs)
        if not windows:
            return []

        # 3. Extract speaker embeddings per window (batched for high throughput on long audio)
        embeddings_list = []
        batch_size = 32
        for i in range(0, len(windows), batch_size):
            batch_wins = windows[i : i + batch_size]
            waveforms = [w.waveform.squeeze() for w in batch_wins]
            max_len = max(w.shape[-1] for w in waveforms)
            padded = [
                torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) if w.shape[-1] < max_len else w
                for w in waveforms
            ]
            batch_t = torch.stack(padded).to(self.device)
            features = self.feature_extractor(batch_t)
            embs = self.speaker_net(features)
            embeddings_list.extend(embs.cpu().numpy())

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # 4. Cluster embeddings into discrete speaker labels
        if isinstance(self.clusterer, SpectralSpeakerClusterer):
            labels = self.clusterer.cluster(embeddings, num_speakers=num_speakers)
        else:
            labels = self.clusterer.cluster(embeddings)

        # 5. Build raw window turns
        raw_turns: List[SpeakerTurn] = []
        for win, label in zip(windows, labels):
            raw_turns.append(
                SpeakerTurn(
                    speaker_id=f"Speaker {label + 1}",
                    start_sec=win.start_sec,
                    end_sec=win.end_sec,
                    duration_sec=round(win.end_sec - win.start_sec, 3),
                    confidence=0.92,
                )
            )

        # 6. Fuse adjacent overlapping or consecutive turns of the same speaker
        fused_turns = self._merge_consecutive_turns(raw_turns)
        return fused_turns

    def _merge_consecutive_turns(self, turns: List[SpeakerTurn], max_gap_sec: float = 0.5) -> List[SpeakerTurn]:
        """Merges consecutive same-speaker windows into cohesive conversational turns."""
        if not turns:
            return []

        merged: List[SpeakerTurn] = []
        current = turns[0]

        for nxt in turns[1:]:
            if nxt.speaker_id == current.speaker_id and (nxt.start_sec - current.end_sec) <= max_gap_sec:
                # Merge current with nxt
                new_end = max(current.end_sec, nxt.end_sec)
                current = SpeakerTurn(
                    speaker_id=current.speaker_id,
                    start_sec=current.start_sec,
                    end_sec=new_end,
                    duration_sec=round(new_end - current.start_sec, 3),
                    confidence=round((current.confidence + nxt.confidence) / 2.0, 3),
                )
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        return merged
