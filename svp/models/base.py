"""Adapter contract between a latent-reasoning checkpoint and the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class LatentModelAdapter(ABC):
    """A model plus the family-specific pieces the generation loop needs.

    Attributes:
        arch: Backbone key used to resolve perturbation targets
            (``"gpt2"`` or ``"llama"``).
        model: The underlying causal LM, in eval mode, on device.
        tokenizer: Tokenizer with a pad token set.
        latent_token_id: Placeholder id written at latent positions of the
            returned sequences.
        default_latent_steps: Number of continuous thoughts the checkpoint
            was trained with.
        logits_vocab_limit: If set, next-token logits are truncated to this
            many entries before decoding.
    """

    arch: str
    model: nn.Module
    tokenizer: object
    latent_token_id: int
    default_latent_steps: int
    default_max_new_tokens: int = 64
    logits_vocab_limit: int | None = None

    @abstractmethod
    def build_prompt(self, question: str) -> str:
        """Format a question so the next position is the first latent step."""

    def extra_prompt_ids(self) -> list[int]:
        """Token ids appended after the tokenized prompt (before latents)."""
        return []

    @abstractmethod
    def latent_feedback(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Map a ``(rows, hidden)`` last-layer state to the next input embedding."""

    @abstractmethod
    def post_latent_token_ids(self) -> list[int]:
        """Token ids fed after the last latent step, before answer decoding."""

    @abstractmethod
    def extract_answer(self, text: str) -> float | None:
        """Parse the numeric answer out of the generated text."""

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
