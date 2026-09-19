"""Grad-enabled replay of rollouts under their stored codes (full BPTT).

The engine runs without gradients, so the update re-runs each rollout's
computation: prefill -> latent steps (feedback threaded, not detached) -> one
teacher-forced forward over the marker tokens and the sampled answer. The
controller is active for prefill + latent steps and off for the answer,
exactly as at rollout time. ``K`` groups are fused into one left-padded batch.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers import DynamicCache

from ..engine import encode_prompts
from ..models.base import LatentModelAdapter
from ..perturb import SVPController


def replay(adapter: LatentModelAdapter, controller: SVPController, groups: list[dict],
           latent_steps: int) -> dict:
    """Replay ``K`` groups of ``G`` rows each.

    Args:
        groups: Records with ``question``, ``codes`` ``(G, layers, rank)`` and
            ``answer_ids`` (per row, first EOS included).

    Returns:
        ``token_logp`` ``(K*G, L)`` grad-carrying answer log-probs, ``mask``
        ``(K*G, L)``, and ``entropy`` per group (mean answer-token entropy).
    """
    model, device = adapter.model, adapter.device
    assert not model.training
    K = len(groups)
    G = groups[0]["codes"].shape[0]
    rows = K * G

    ids, mask = encode_prompts(adapter, [g["question"] for g in groups])
    input_ids = ids.repeat_interleave(G, dim=0).to(device)
    mask = mask.repeat_interleave(G, dim=0).to(device)
    controller.begin_batch(rows, torch.cat([g["codes"] for g in groups]))
    controller.set_active(True)

    positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                past_key_values=DynamicCache(), use_cache=True, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1][:, -1, :]
    cache = out.past_key_values
    next_pos = mask.long().sum(-1)
    embed_dtype = model.get_input_embeddings().weight.dtype
    for _ in range(latent_steps):
        feedback = adapter.latent_feedback(h)
        mask = torch.cat([mask, mask.new_ones(rows, 1)], dim=-1)
        out = model(inputs_embeds=feedback.unsqueeze(1).to(embed_dtype), attention_mask=mask,
                    position_ids=next_pos.unsqueeze(-1), past_key_values=cache, use_cache=True,
                    output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1][:, -1, :]
        cache = out.past_key_values
        next_pos = next_pos + 1
    controller.set_active(False)

    post = adapter.post_latent_token_ids()
    P = len(post)
    answer_ids = [a for g in groups for a in g["answer_ids"]]
    lengths = [len(a) for a in answer_ids]
    L = max(lengths)
    assert min(lengths) >= 1
    pad = adapter.pad_token_id
    tail_in = torch.full((rows, P + L - 1), pad, dtype=torch.long)
    targets = torch.full((rows, L), pad, dtype=torch.long)
    for i, a in enumerate(answer_ids):
        row = post + a[:-1]
        tail_in[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        targets[i, : len(a)] = torch.tensor(a, dtype=torch.long)
    tail_in, targets = tail_in.to(device), targets.to(device)
    tgt_mask = torch.arange(L, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)

    mask = torch.cat([mask, mask.new_ones(rows, tail_in.shape[1])], dim=-1)
    pos = next_pos.unsqueeze(-1) + torch.arange(tail_in.shape[1], device=device).unsqueeze(0)
    out = model(input_ids=tail_in, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
    logits = out.logits[:, P - 1: P - 1 + L, :]
    if adapter.logits_vocab_limit is not None:
        logits = logits[:, :, : adapter.logits_vocab_limit]
    logp_all = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ent = -(logp_all.exp() * logp_all).sum(-1)
        entropy = [float((ent[k * G:(k + 1) * G] * tgt_mask[k * G:(k + 1) * G]).sum()
                         / tgt_mask[k * G:(k + 1) * G].sum()) for k in range(K)]
    controller.end_batch()
    return {"token_logp": token_logp, "mask": tgt_mask, "entropy": entropy}
