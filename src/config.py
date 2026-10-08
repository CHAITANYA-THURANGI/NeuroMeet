"""Configuration loader and manager for NeuroMeet."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "deterministic": True,
    "audio": {
        "sample_rate": 16000,
        "n_fft": 512,
        "hop_length": 160,
        "win_length": 400,
        "n_mels": 80,
        "f_min": 50.0,
        "f_max": 8000.0,
        "vad": {
            "energy_threshold": 0.015,
            "zcr_threshold": 0.12,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 300,
        },
    },
    "models": {
        "speaker_net": {
            "feat_dim": 80,
            "channels": 192,
            "bottleneck_dim": 96,
            "emb_dim": 192,
            "pooling": "attentive_stats",
            "dropout": 0.1,
        },
        "speech_ctc": {
            "input_dim": 80,
            "d_model": 144,
            "n_conformer_blocks": 4,
            "num_heads": 4,
            "conv_kernel_size": 15,
            "ff_expansion_factor": 4,
            "dropout": 0.1,
            "vocab_size": 40,
        },
        "summarizer": {
            "type": "hierarchical",
            "emb_dim": 128,
            "word_hid_dim": 128,
            "utt_hid_dim": 192,
            "attention": "scaled_dot",
            "copy_gate": True,
            "num_heads": 4,
            "max_summary_len": 256,
            "beam_size": 3,
            "coverage_loss": True,
        },
        "action_extractor": {
            "emb_dim": 128,
            "hid_dim": 160,
            "num_layers": 2,
            "dropout": 0.2,
            "tag_set": ["O", "B-ACT", "I-ACT", "B-WHO", "I-WHO", "B-WHEN", "I-WHEN", "B-PRIO", "I-PRIO"],
            "priority_classes": ["low", "medium", "high", "urgent"],
        },
        "dynamics_net": {
            "emb_dim": 128,
            "hid_dim": 96,
            "sentiment_classes": ["positive", "neutral", "negative"],
            "agreement_classes": ["consensus", "neutral", "contention"],
            "dropout": 0.15,
        },
        "dense_retriever": {
            "emb_dim": 128,
            "query_hid_dim": 128,
            "doc_hid_dim": 128,
            "top_k": 3,
        },
    },
    "diarization": {
        "min_speakers": 1,
        "max_speakers": 8,
        "clustering_method": "spectral",
        "affinity_metric": "cosine",
        "similarity_threshold": 0.65,
        "window_size_sec": 1.5,
        "step_size_sec": 0.75,
    },
    "train": {
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "epochs": 30,
        "patience": 6,
        "device": "auto",
        "fp16": False,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False,
        "cors_origins": ["*"],
        "max_audio_upload_mb": 50,
        "rate_limit_per_minute": 120,
    },
    "paths": {
        "data_dir": "data",
        "processed_dir": "data/processed",
        "checkpoints_dir": "experiments/checkpoints",
        "logs_dir": "experiments/logs",
        "web_dir": "web",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base dictionary."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads configuration from yaml file, merging with default config."""
    cfg = dict(DEFAULT_CONFIG)
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")

    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
            if user_cfg and isinstance(user_cfg, dict):
                cfg = _deep_merge(cfg, user_cfg)
    return cfg
