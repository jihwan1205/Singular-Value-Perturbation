"""Group rollouts: G perturbed samples per prompt with their codes captured."""

from __future__ import annotations

import torch

from ..engine import GenerationSettings, generate_batch
from ..metrics import is_correct
from ..models.base import LatentModelAdapter
from ..perturb import SVPController
from .config import GrpoConfig


def group_seed(base_seed: int, iteration: int, question_idx: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    z = (base_seed * 0x9E3779B97F4A7C15 + iteration * 0xBF58476D1CE4E5B9
         + question_idx * 0x94D049BB133111EB + 1) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & 0x7FFFFFFF


def cut_at_eos(tokens: list[int], eos_id: int) -> list[int]:
    if eos_id in tokens:
        return tokens[: tokens.index(eos_id) + 1]
    return tokens


@torch.no_grad()
def rollout_groups(adapter: LatentModelAdapter, controller: SVPController,
                   items: list[dict], cfg: GrpoConfig, iteration: int) -> list[dict]:
    """Roll out ``len(items)`` prompts x ``G`` samples in one batch.

    Returns one record per prompt: the ``(G, layers, rank)`` codes the
    samples were drawn with, the answer token ids (through the first EOS),
    correctness flags, and the decode-time token log-probs.
    """
    torch.manual_seed(group_seed(cfg.seed, iteration, int(items[0]["idx"])))
    g = cfg.group_size
    settings = GenerationSettings(
        latent_steps=cfg.latent_steps, max_new_tokens=64 + cfg.extra_decode, n_samples=g,
        batch_size=len(items), temperature=cfg.temperature, top_p=cfg.top_p,
        sample_seed=group_seed(cfg.seed + 1, iteration, int(items[0]["idx"])),
        return_token_logps=True,
    )
    out = generate_batch(adapter, [it["question"] for it in items], settings, controller)
    codes = controller.codes.to("cpu", torch.float32)
    steps = settings.max_new_tokens - cfg.latent_steps - len(adapter.post_latent_token_ids())
    generated = out.sequences[:, -steps:]
    records = []
    for p, item in enumerate(items):
        rows = slice(p * g, (p + 1) * g)
        answers = [adapter.extract_answer(t) for t in out.generated_texts[rows]]
        records.append({
            "idx": int(item["idx"]), "question": item["question"], "gold": item["gold"],
            "codes": codes[rows].clone(),
            "answer_ids": [cut_at_eos(r.tolist(), adapter.eos_token_id) for r in generated[rows]],
            "corrects": [bool(is_correct(a, item["gold"])) for a in answers],
            "rollout_logp": out.token_logps[rows].clone(),
        })
    return records
