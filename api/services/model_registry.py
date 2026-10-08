"""Singleton Model Registry for NeuroMeet."""

from __future__ import annotations
import logging
from typing import Optional
import torch
from src.config import load_config
from src.pipeline.omni_meeting import OmniMeetingPipeline


logger = logging.getLogger("neuromeet.registry")


class ModelRegistry:
    """Manages neural network lifecycle, singleton pipeline instance, and device binding."""
    _instance: Optional[ModelRegistry] = None
    _pipeline: Optional[OmniMeetingPipeline] = None
    _device: Optional[str] = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_device(cls) -> str:
        if cls._device is None:
            cfg = load_config()
            configured_dev = cfg.get("train", {}).get("device", "auto")
            if configured_dev == "auto":
                cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                cls._device = configured_dev
        return cls._device

    @classmethod
    def get_pipeline(cls) -> OmniMeetingPipeline:
        """Returns initialized singleton OmniMeetingPipeline."""
        if cls._pipeline is None:
            dev = cls.get_device()
            logger.info(f"Initializing OmniMeetingPipeline on device: {dev}")
            cls._pipeline = OmniMeetingPipeline(device=dev)
        return cls._pipeline


def get_pipeline() -> OmniMeetingPipeline:
    return ModelRegistry.get_pipeline()
