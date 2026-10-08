"""Training script for NeuroMeet Deep Learning models."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml

from src.config import load_config
from src.datasets.meeting_corpus import create_dataloader
from src.models.meeting_summarizer import HierarchicalAttentionSummarizer


def train_summarizer(cfg: dict, epochs: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("train", {}).get("device") != "cpu" else "cpu")
    print(f"Training HierarchicalAttentionSummarizer on device: {device}")

    model_cfg = cfg.get("models", {}).get("summarizer", {})
    model = HierarchicalAttentionSummarizer(
        vocab_size=4000,
        emb_dim=model_cfg.get("emb_dim", 128),
        word_hid_dim=model_cfg.get("word_hid_dim", 128),
        utt_hid_dim=model_cfg.get("utt_hid_dim", 192),
        dec_hid_dim=model_cfg.get("utt_hid_dim", 192),
        copy_gate=model_cfg.get("copy_gate", True),
    ).to(device)

    train_loader = create_dataloader(batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    checkpoints_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "experiments/checkpoints")) / "summarizer"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            meeting_tokens = batch["meeting_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            optimizer.zero_grad()
            # Forward pass: [B, tgt_len - 1, vocab_size]
            logits = model(meeting_tokens, target_tokens)
            targets = target_tokens[:, 1:]  # Shifted right

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch:02d}/{epochs:02d} — Train Loss: {avg_loss:.4f}")

    best_ckpt = checkpoints_dir / "best.pt"
    torch.save({"model_state": model.state_dict(), "config": model_cfg}, best_ckpt)
    print(f"Training completed! Checkpoint saved to: {best_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroMeet Model Training CLI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_summarizer(cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
