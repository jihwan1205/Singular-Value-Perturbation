"""Coconut adapter (GPT-2 backbone; ModalityDance/latent-tts-coconut layout)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LatentModelAdapter


class CoconutAdapter(LatentModelAdapter):
    """Continuous thought = raw last-layer hidden state fed back as input.

    Prompts end with ``\\n<|start-latent|>``; the latent phase is closed by
    ``<|end-latent|>``; the answer follows ``#`` in the generated text.
    """

    arch = "gpt2"
    default_latent_steps = 6

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float32, pad_token_id=self.tokenizer.pad_token_id
        )
        self.model.to(device).eval()
        tok = self.tokenizer.convert_tokens_to_ids
        self.latent_token_id = tok("<|latent|>")
        self.start_token_id = tok("<|start-latent|>")
        self.end_token_id = tok("<|end-latent|>")
        assert len({self.latent_token_id, self.start_token_id, self.end_token_id}) == 3

    def build_prompt(self, question: str) -> str:
        return question + "\n<|start-latent|>"

    def latent_feedback(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return last_hidden

    def post_latent_token_ids(self) -> list[int]:
        return [self.end_token_id]

    def extract_answer(self, text: str) -> float | None:
        try:
            return float(text.split("#")[-1].strip().replace(",", ""))
        except ValueError:
            return None
