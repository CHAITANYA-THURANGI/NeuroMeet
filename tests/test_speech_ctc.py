"""Unit tests for Conformer Acoustic CTC model."""

import torch
from src.models.speech_ctc import ConformerBlock, SpeechCTC


def test_conformer_block_forward() -> None:
    block = ConformerBlock(d_model=64, num_heads=4, kernel_size=15)
    x = torch.randn(2, 50, 64)
    out = block(x)
    assert out.shape == (2, 50, 64)


def test_speech_ctc_forward_and_greedy_decode() -> None:
    model = SpeechCTC(input_dim=80, d_model=64, n_conformer_blocks=2, num_heads=4)
    model.eval()

    # Input: [B, 80, 100]
    features = torch.randn(2, 80, 100)
    with torch.no_grad():
        log_probs = model(features)

    # Subsampled time should be ~50 frames
    assert log_probs.dim() == 3
    assert log_probs.size(0) == 2
    assert log_probs.size(2) == model.vocab_size

    # Test greedy decoding
    transcripts = model.decode_greedy(log_probs)
    assert len(transcripts) == 2
    assert isinstance(transcripts[0], str)
