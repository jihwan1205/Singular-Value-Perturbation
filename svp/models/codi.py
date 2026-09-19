"""CODI adapter (GPT-2 / LLaMA-3.2-1B; converted zen-E checkpoints).

Official inference semantics: prompt = the raw question followed by the
``bot`` token, six continuous thoughts each produced by the projector MLP from
the last hidden state, the ``eot`` token, then greedy decoding of
``The answer is: N`` with the ``eot`` logit masked out.

Checkpoints must be converted with ``scripts/convert_codi.py`` first (LoRA
merged into the base weights, projector saved as ``projector.pt``).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LatentModelAdapter

_NUMBER = re.compile(r"-?\d+\.?\d*")


class Projector(nn.Module):
    """CODI's ``prj``: Linear -> GELU -> Linear -> LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.0), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)

    def load_official(self, sd: dict[str, torch.Tensor]) -> None:
        self.net[1].weight.data.copy_(sd["1.weight"])
        self.net[1].bias.data.copy_(sd["1.bias"])
        self.net[3].weight.data.copy_(sd["3.weight"])
        self.net[3].bias.data.copy_(sd["3.bias"])
        self.ln.weight.data.copy_(sd["ln.weight"])
        self.ln.bias.data.copy_(sd["ln.bias"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x))


class CodiAdapter(LatentModelAdapter):
    default_latent_steps = 6

    def __init__(self, checkpoint: str, arch: str, device: str = "cuda"):
        self.arch = arch
        path = Path(checkpoint)
        if not (path / "projector.pt").exists():
            raise FileNotFoundError(
                f"{checkpoint} is not a converted CODI checkpoint; run scripts/convert_codi.py"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        self.model.to(device).eval()

        vocab = self.model.config.vocab_size  # original + 3 (pad, bot, eot)
        self.bot_token_id = vocab - 2
        self.eot_token_id = vocab - 1
        self.latent_token_id = self.bot_token_id
        self.logits_vocab_limit = vocab - 1

        self.projector = Projector(self.model.config.hidden_size)
        self.projector.load_official(torch.load(path / "projector.pt", map_location="cpu"))
        self.projector.to(device).eval()

    def build_prompt(self, question: str) -> str:
        return question.strip().replace("  ", " ")

    def extra_prompt_ids(self) -> list[int]:
        return [self.bot_token_id]

    def latent_feedback(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.projector(last_hidden)

    def post_latent_token_ids(self) -> list[int]:
        return [self.eot_token_id]

    def extract_answer(self, text: str) -> float | None:
        numbers = _NUMBER.findall(text.replace(",", ""))
        if not numbers:
            return None
        try:
            return float(numbers[-1])
        except ValueError:
            return None
