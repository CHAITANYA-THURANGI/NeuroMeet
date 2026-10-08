"""Acoustic feature extraction: Log-Mel Filterbanks, Spectrograms, and MFCCs."""

from __future__ import annotations
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def hz_to_mel(hz: float) -> float:
    """Converts frequency in Hz to Mel scale (O'Shaughnessy formula)."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Converts Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Constructs a triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1]."""
    num_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Uniformly spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    # Convert Hz to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)

    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return torch.from_numpy(filterbank)


def create_dct_matrix(n_mfcc: int = 13, n_mels: int = 80) -> torch.Tensor:
    """Generates Type-II Discrete Cosine Transform (DCT) matrix for MFCC extraction."""
    dct = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    for i in range(n_mfcc):
        for j in range(n_mels):
            dct[i, j] = math.cos(math.pi * i * (j + 0.5) / n_mels)
    # Orthogonal normalization factor
    dct[0, :] *= math.sqrt(1.0 / n_mels)
    dct[1:, :] *= math.sqrt(2.0 / n_mels)
    return torch.from_numpy(dct)


class LogMelExtractor(nn.Module):
    """Deep learning ready, differentiable Log-Mel Spectrogram feature extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        # Precompute window and mel filterbank
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_filterbank", mel_fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes Log-Mel Spectrogram features.

        Args:
            waveform: Tensor of shape [batch, num_samples] or [num_samples]

        Returns:
            log_mel: Tensor of shape [batch, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device
        window = self.window.to(device)
        mel_fb = self.mel_filterbank.to(device)

        # STFT: [batch, n_fft // 2 + 1, time_frames, 2] (or complex)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Power spectrogram |STFT|^2
        power_spec = stft.abs().pow(2.0)  # [batch, freq_bins, time_frames]

        # Mel filterbank projection: [batch, n_mels, time_frames]
        # mel_fb is [n_mels, freq_bins]
        mel_spec = torch.matmul(mel_fb, power_spec)

        # Numerical stabilization and log compression
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Per-instance standardization (Mean and variance normalization along time)
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-6
        norm_log_mel = (log_mel - mean) / std

        return norm_log_mel


def compute_mfcc(
    log_mel: torch.Tensor,
    n_mfcc: int = 13,
) -> torch.Tensor:
    """Computes Mel-Frequency Cepstral Coefficients (MFCC) from log-mel spectrogram.

    Args:
        log_mel: Tensor of shape [batch, n_mels, time_frames]
        n_mfcc: Number of cepstral coefficients (typically 13)

    Returns:
        mfcc: Tensor of shape [batch, n_mfcc, time_frames]
    """
    n_mels = log_mel.size(1)
    dct = create_dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels).to(log_mel.device)
    # Matrix multiplication along mels axis
    mfcc = torch.matmul(dct, log_mel)
    return mfcc
