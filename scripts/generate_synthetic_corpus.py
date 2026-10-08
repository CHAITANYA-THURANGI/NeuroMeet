"""Generates pre-packaged JSON scenarios and multi-speaker synthetic WAV audio preview."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import torch

from src.audio.wav_io import generate_synthetic_audio, write_wav
from src.datasets.generator import generate_meeting_scenarios


def main() -> None:
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    audio_dir = data_dir / "audio_samples"
    processed_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Generating meeting scenarios in data/processed/...")
    scenarios = generate_meeting_scenarios()

    for sc_id, sc in scenarios.items():
        out_file = processed_dir / f"sample_{sc_id}.json"
        data = asdict(sc)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  -> Saved {out_file} ({len(sc.turns)} turns)")

    print("\nGenerating synthetic multi-speaker audio benchmark preview in data/audio_samples/...")
    # Generate 3 distinct turns: Speaker 1 (2.5s), Speaker 2 (3.0s), Speaker 3 (2.8s)
    spk1_audio = generate_synthetic_audio(duration_sec=2.5, speaker_id=1, amplitude=0.6)
    silence = torch.zeros((1, int(16000 * 0.4)), dtype=torch.float32)
    spk2_audio = generate_synthetic_audio(duration_sec=3.0, speaker_id=2, amplitude=0.6)
    spk3_audio = generate_synthetic_audio(duration_sec=2.8, speaker_id=3, amplitude=0.6)

    combined_meeting_audio = torch.cat([spk1_audio, silence, spk2_audio, silence, spk3_audio], dim=-1)
    wav_path = audio_dir / "synthetic_meeting_preview.wav"
    write_wav(wav_path, combined_meeting_audio, sample_rate=16000)
    print(f"  -> Generated {wav_path} (Duration: {combined_meeting_audio.shape[-1] / 16000:.1f}s)")
    print("Done!")


if __name__ == "__main__":
    main()
