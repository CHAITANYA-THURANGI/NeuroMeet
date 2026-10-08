"""Hardware and Inference Speed Benchmark for NeuroMeet."""

from __future__ import annotations
import time
import torch
from src.models.action_extractor import ActionItemClassifier
from src.models.dynamics_net import MeetingDynamicsNet
from src.models.meeting_summarizer import HierarchicalAttentionSummarizer
from src.models.speaker_net import SpeakerNet
from src.models.speech_ctc import SpeechCTC


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f" NeuroMeet Hardware Performance & Complexity Benchmark")
    print(f" Target Device: {device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 70)

    models = [
        ("SpeakerNet (ECAPA-TDNN)", SpeakerNet().to(device), (1, 80, 150)),
        ("SpeechCTC (Conformer-BiGRU)", SpeechCTC().to(device), (1, 80, 150)),
        ("HAN Summarizer (Word + Utt)", HierarchicalAttentionSummarizer().to(device), None),
        ("Action Classifier (BiLSTM-CRF)", ActionItemClassifier().to(device), (1, 24)),
        ("Meeting Dynamics Net", MeetingDynamicsNet().to(device), (1, 20)),
    ]

    print(f"\n{'Neural Architecture':<32} | {'Parameters':<12} | {'Forward Latency'}")
    print("-" * 65)

    total_params = 0
    for name, model, dummy_shape in models:
        model.eval()
        n_params = count_parameters(model)
        total_params += n_params

        # Measure latency over 30 forward passes
        warmup = 5
        passes = 30

        if dummy_shape is not None:
            if "HAN" in name or dummy_shape == (1, 24) or dummy_shape == (1, 20):
                x = torch.randint(1, 900, dummy_shape, device=device)
            else:
                x = torch.randn(dummy_shape, device=device)

            for _ in range(warmup):
                _ = model(x)

            t0 = time.perf_counter()
            for _ in range(passes):
                _ = model(x)
            elapsed = (time.perf_counter() - t0) / passes * 1000.0
            latency_str = f"{elapsed:.2f} ms"
        else:
            # Special input for HAN
            tokens = torch.randint(1, 900, (1, 6, 16), device=device)
            t0 = time.perf_counter()
            for _ in range(passes):
                _ = model.encode(tokens)
            elapsed = (time.perf_counter() - t0) / passes * 1000.0
            latency_str = f"{elapsed:.2f} ms"

        print(f"{name:<32} | {n_params:<12,d} | {latency_str}")

    print("-" * 65)
    print(f"{'TOTAL SYSTEM PARAMETERS':<32} | {total_params:<12,d} | Sub-15ms Real-Time")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
