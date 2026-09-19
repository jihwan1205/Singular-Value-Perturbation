"""Two-phase (latent -> explicit) batched generation over ``model.forward``.

1. Prefill the prompt (perturbed when a controller is given).
2. ``latent_steps`` continuous thoughts: the adapter maps the last hidden
   state to the next input embedding (perturbed).
3. Feed the adapter's post-latent marker tokens (clean).
4. Decode the answer greedily, or by temperature / nucleus sampling (clean),
   until EOS or the budget ``max_new_tokens`` (which counts latent and marker
   positions) is exhausted. Finished rows are pruned from the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from transformers import DynamicCache

from .models.base import LatentModelAdapter


class NoiseController(Protocol):
    def begin_batch(self, rows: int) -> None: ...
    def set_active(self, active: bool) -> None: ...
    def end_batch(self) -> None: ...


@dataclass
class GenerationSettings:
    """Attributes:
        latent_steps: Number of continuous thoughts.
        max_new_tokens: Budget after the prompt, including latent and marker positions.
        n_samples: Samples per question (rows are expanded in-batch).
        batch_size: Questions per forward chunk (rows = batch_size * n_samples).
        temperature: Answer-token softmax temperature; ``0`` = greedy.
        top_p: Nucleus mass for answer sampling; ``1`` = no truncation.
        sample_seed: Seed of the answer-sampling stream (independent of the
            noise RNG; a row's draws depend only on its global row id).
        return_token_logps: Also return the log-prob of each generated token.
    """

    latent_steps: int = 6
    max_new_tokens: int = 64
    n_samples: int = 1
    batch_size: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    sample_seed: int = 0
    return_token_logps: bool = False


@dataclass
class GenerationOutput:
    """``sequences``: ``(rows, L)`` prompt + latent placeholders + markers + answer.
    ``generated_texts``: decoded answer part per row.
    ``token_logps``: ``(rows, answer_budget)`` fp32 log-probs (zeros past EOS) or ``None``.
    """

    sequences: torch.Tensor
    generated_texts: list[str] = field(default_factory=list)
    token_logps: torch.Tensor | None = None


def _step_seed(base_seed: int, step: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    z = (base_seed * 0x9E3779B97F4A7C15 + (step + 1) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


def select_next_token(logits: torch.Tensor, temperature: float, top_p: float,
                      uniforms: torch.Tensor | None) -> torch.Tensor:
    """Argmax, or inverse-CDF nucleus sampling driven by one uniform per row."""
    if temperature == 0.0 or uniforms is None:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    u = uniforms.to(torch.float32).unsqueeze(-1)
    if top_p >= 1.0:
        cdf = scaled.softmax(dim=-1).cumsum(dim=-1)
        idx = torch.searchsorted(cdf.contiguous(), u)
        return idx.clamp(max=cdf.shape[-1] - 1).squeeze(-1)
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    kept = probs * ((cum - probs) < top_p)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    pos = torch.searchsorted(kept.cumsum(dim=-1).contiguous(), u)
    pos = pos.clamp(max=kept.shape[-1] - 1)
    return sorted_idx.gather(-1, pos).squeeze(-1)


def encode_prompts(adapter: LatentModelAdapter, questions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded ``(input_ids, attention_mask)`` including the adapter's extra ids."""
    tokenizer = adapter.tokenizer
    tokenizer.padding_side = "left"
    enc = tokenizer([adapter.build_prompt(q) for q in questions], return_tensors="pt", padding=True)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    extra = adapter.extra_prompt_ids()
    if extra:
        b = input_ids.shape[0]
        input_ids = torch.cat([input_ids, torch.tensor(extra, dtype=torch.long).expand(b, len(extra))], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, len(extra))], dim=-1)
    return input_ids, attention_mask


@torch.no_grad()
def generate_batch(
    adapter: LatentModelAdapter,
    questions: list[str],
    settings: GenerationSettings,
    controller: NoiseController | None = None,
    row_offset: int = 0,
    total_rows: int | None = None,
) -> GenerationOutput:
    """Generate ``n_samples`` completions per question in one batched pass.

    Args:
        adapter: Model adapter.
        questions: One chunk of raw questions.
        settings: Generation settings.
        controller: Optional weight-noise controller (active for prefill and
            latent steps only).
        row_offset: Global index of this chunk's first row (answer sampling).
        total_rows: Rows in the whole run; with ``row_offset`` it makes a
            row's sampled tokens independent of batch composition.
    """
    model, device, n = adapter.model, adapter.device, settings.n_samples
    input_ids, attention_mask = encode_prompts(adapter, questions)
    input_ids = input_ids.repeat_interleave(n, dim=0).to(device)
    attention_mask = attention_mask.repeat_interleave(n, dim=0).to(device)
    rows = input_ids.shape[0]
    embed_dtype = model.get_input_embeddings().weight.dtype

    if controller is not None:
        controller.begin_batch(rows)
        controller.set_active(True)

    positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                    past_key_values=DynamicCache(), use_cache=True,
                    output_hidden_states=True, logits_to_keep=1)
    last_hidden = outputs.hidden_states[-1][:, -1, :]
    past = outputs.past_key_values
    next_pos = attention_mask.long().sum(-1)

    sequences = [input_ids]
    for _ in range(settings.latent_steps):
        feedback = adapter.latent_feedback(last_hidden)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(rows, 1)], dim=-1)
        outputs = model(inputs_embeds=feedback.unsqueeze(1).to(embed_dtype),
                        attention_mask=attention_mask, position_ids=next_pos.unsqueeze(-1),
                        past_key_values=past, use_cache=True,
                        output_hidden_states=True, logits_to_keep=1)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        past = outputs.past_key_values
        next_pos = next_pos + 1
        sequences.append(torch.full((rows, 1), adapter.latent_token_id, dtype=torch.long, device=device))

    if controller is not None:
        controller.set_active(False)

    post_ids = adapter.post_latent_token_ids()
    for token_id in post_ids:
        col = torch.full((rows, 1), token_id, dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(rows, 1)], dim=-1)
        outputs = model(input_ids=col, attention_mask=attention_mask, position_ids=next_pos.unsqueeze(-1),
                        past_key_values=past, use_cache=True, logits_to_keep=1)
        past = outputs.past_key_values
        next_pos = next_pos + 1
        sequences.append(col)

    steps = max(settings.max_new_tokens - settings.latent_steps - len(post_ids), 0)
    generated = torch.full((rows, steps), adapter.pad_token_id, dtype=torch.long, device=device)
    token_logps = torch.zeros(rows, steps, dtype=torch.float32, device=device) if settings.return_token_logps else None
    active = torch.arange(rows, device=device)
    sampling = settings.temperature > 0.0
    draw_rows = rows if total_rows is None else total_rows
    if sampling and row_offset + rows > draw_rows:
        raise ValueError(f"row_offset {row_offset} + rows {rows} exceeds total_rows {draw_rows}")
    sample_gen = torch.Generator(device=device) if sampling else None
    vocab_limit = adapter.logits_vocab_limit

    for step in range(steps):
        logits = outputs.logits[:, -1, :]
        if vocab_limit is not None:
            logits = logits[:, :vocab_limit]
        uniforms = None
        if sampling:
            sample_gen.manual_seed(_step_seed(settings.sample_seed, step))
            uniforms = torch.rand(draw_rows, generator=sample_gen, device=device)[row_offset + active]
        next_tokens = select_next_token(logits, settings.temperature, settings.top_p, uniforms)
        generated[active, step] = next_tokens
        if token_logps is not None:
            token_logps[active, step] = logits.float().log_softmax(-1).gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        keep = next_tokens != adapter.eos_token_id
        if not keep.any():
            break
        if not keep.all():
            active, next_tokens = active[keep], next_tokens[keep]
            attention_mask, next_pos = attention_mask[keep], next_pos[keep]
            past.batch_select_indices(keep.nonzero().squeeze(-1))
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(active.shape[0], 1)], dim=-1)
        outputs = model(input_ids=next_tokens.unsqueeze(-1), attention_mask=attention_mask,
                        position_ids=next_pos.unsqueeze(-1), past_key_values=past,
                        use_cache=True, logits_to_keep=1)
        past = outputs.past_key_values
        next_pos = next_pos + 1

    if controller is not None:
        controller.end_batch()
    texts = [adapter.decode(row.tolist()) for row in generated]
    return GenerationOutput(
        sequences=torch.cat(sequences + [generated], dim=-1).cpu(),
        generated_texts=texts,
        token_logps=token_logps.cpu() if token_logps is not None else None,
    )
