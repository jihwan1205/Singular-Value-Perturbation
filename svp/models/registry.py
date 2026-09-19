"""Model registry: name -> (adapter class, default checkpoint, backbone)."""

from __future__ import annotations

from .base import LatentModelAdapter
from .coconut import CoconutAdapter
from .codi import CodiAdapter

MODELS: dict[str, dict] = {
    "coconut": {"cls": CoconutAdapter, "checkpoint": "ModalityDance/latent-tts-coconut", "arch": "gpt2"},
    "codi-gpt2": {"cls": CodiAdapter, "checkpoint": "checkpoints/codi-gpt2", "arch": "gpt2"},
    "codi-llama1b": {"cls": CodiAdapter, "checkpoint": "checkpoints/codi-llama1b", "arch": "llama"},
}


def load_adapter(name: str, checkpoint: str | None = None, device: str = "cuda") -> LatentModelAdapter:
    """Load a registered model family from its default or an overriding checkpoint.

    Args:
        name: One of :data:`MODELS`.
        checkpoint: HF hub id or local directory; ``None`` uses the default.
        device: Torch device string.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    spec = MODELS[name]
    ckpt = checkpoint or spec["checkpoint"]
    if spec["cls"] is CodiAdapter:
        return CodiAdapter(ckpt, arch=spec["arch"], device=device)
    return CoconutAdapter(ckpt, device=device)
