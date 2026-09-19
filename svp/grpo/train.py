"""SVP-V-GRPO trainer.

    python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json [--field value ...]
    torchrun --nproc_per_node 4 -m svp.grpo.train --config ...   # data-parallel
    python -m svp.grpo.train --config ... --resume results/grpo/<run>/ckpt_last.pt

Per iteration: roll out ``B * overprovision`` prompts x ``G`` samples under
SVP (one code per sample, held over prefill + latent steps; answer tokens
sampled at ``temperature``), keep the first ``B`` mixed-outcome groups, and
take ``mu`` clipped-surrogate steps on the target weights with a k3 KL to the
frozen base. Ratios are formed under the same codes the rollouts used, so
they measure the parameter change only. Under ``torchrun`` every rank rolls
out and replays its own share of the prompts; gradients are summed so the
update equals the single-process one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ..data import benchmark_path, load_examples, parse_gold, BENCHMARKS
from ..engine import GenerationSettings
from ..evaluate import evaluate
from ..models import load_adapter
from ..perturb import SVPController
from .config import GrpoConfig, add_config_args, build_config
from .replay import replay
from .rollout import rollout_groups
from .runlog import RunLogger

logger = logging.getLogger(__name__)


class StreamSampler:
    """Shuffled, no-repeat pass over the training file; reshuffled per pass."""

    def __init__(self, cfg: GrpoConfig):
        data = json.loads(Path(cfg.train_data).read_text())
        self.items = []
        for i in range(cfg.stream_skip, len(data)):
            gold = parse_gold(data[i]["answer"])
            if gold is not None:
                self.items.append({"idx": i, "question": data[i]["question"], "gold": gold})
        self.seed = cfg.seed * 77 + 13
        self.pass_index = 0
        self.cursor = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self.items.sort(key=lambda it: it["idx"])
        random.Random(self.seed + 1000 * self.pass_index).shuffle(self.items)

    def next(self, n: int) -> list[dict]:
        out = self.items[self.cursor: self.cursor + n]
        self.cursor += n
        if len(out) < n:
            self.pass_index += 1
            self._shuffle()
            self.cursor = n - len(out)
            out = out + self.items[: self.cursor]
        return out

    def state(self) -> dict:
        return {"pass_index": self.pass_index, "cursor": self.cursor}

    def load_state(self, state: dict) -> None:
        self.pass_index, self.cursor = state["pass_index"], state["cursor"]
        self._shuffle()


def setup_trainable(controller: SVPController) -> list[torch.nn.Parameter]:
    """Enable gradients on the target weights only (fused QKV: target columns only)."""
    params: dict[torch.nn.Parameter, None] = {}
    for w in controller.wrappers:
        p = w.base.weight
        if p in params:
            continue
        p.requires_grad_(True)
        if w.out_slice is not None:
            def make_hook(sl):
                def hook(grad):
                    masked = torch.zeros_like(grad)
                    masked[:, sl] = grad[:, sl]
                    return masked
                return hook
            p.register_hook(make_hook(w.out_slice))
        params[p] = None
    return list(params)


def surrogate(token_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
              mask: torch.Tensor, clip_eps: float) -> tuple[torch.Tensor, float]:
    """Token-level clipped surrogate with sequence advantage, summed over
    tokens and averaged over the group (no length normalization)."""
    ratio = (token_logp - old_logp).exp()
    a = adv.unsqueeze(1)
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * a
    loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask.shape[0]
    with torch.no_grad():
        clip_frac = float(((clipped < unclipped) & mask.bool()).float().sum() / mask.sum().clamp_min(1))
    return loss, clip_frac


def k3_kl(ref_logp: torch.Tensor, token_logp: torch.Tensor, mask: torch.Tensor, clamp: float) -> torch.Tensor:
    diff = (ref_logp - token_logp).clamp(-clamp, clamp)
    return ((diff.exp() - diff - 1.0) * mask).sum() / mask.sum()


def all_reduce_sum(values: list[float], device) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def run_eval(adapter, controller: SVPController, cfg: GrpoConfig, rank: int, world: int) -> dict:
    """Clean greedy pass@1 and SVP pass@k on the eval set, sharded over ranks."""
    examples = load_examples(benchmark_path(cfg.eval_data))[rank::world]
    torch.manual_seed(cfg.seed * 1009 + rank)
    greedy = GenerationSettings(latent_steps=cfg.latent_steps, max_new_tokens=64, n_samples=1, batch_size=32)
    _, rec1 = evaluate(adapter, examples, greedy, None)
    svp = GenerationSettings(latent_steps=cfg.latent_steps, max_new_tokens=64,
                             n_samples=cfg.eval_samples, batch_size=cfg.eval_batch)
    _, reck = evaluate(adapter, examples, svp, controller)
    n1, nk, n = all_reduce_sum([sum(r["corrects"][0] for r in rec1),
                                sum(any(r["corrects"]) for r in reck), len(examples)], adapter.device)
    return {"eval/pass@1": n1 / n, f"eval/pass@{cfg.eval_samples}": nk / n}


def run_bench(adapter, cfg: GrpoConfig, rank: int, world: int) -> dict:
    """Clean greedy accuracy on every benchmark, sharded over ranks."""
    settings = GenerationSettings(latent_steps=cfg.latent_steps, max_new_tokens=64, n_samples=1, batch_size=32)
    out = {}
    for name in BENCHMARKS:
        examples = load_examples(benchmark_path(name))[rank::world]
        _, recs = evaluate(adapter, examples, settings, None)
        n_correct, n = all_reduce_sum([sum(r["corrects"][0] for r in recs), len(examples)], adapter.device)
        out[f"bench/{name}"] = n_correct / n
    return out


def save_checkpoint(path: Path, k: int, cfg: GrpoConfig, controller: SVPController,
                    optimizer, sampler: StreamSampler) -> None:
    torch.save({"iter": k, "target": cfg.target, "weights": controller.export_weights(),
                "opt": optimizer.state_dict(), "sampler": sampler.state(), "config": cfg.to_dict()}, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="JSON file of GrpoConfig fields")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    add_config_args(p)
    args = p.parse_args()
    cfg = build_config(args.config, args)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(device))
    is_main = rank == 0
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.temperature <= 0:
        raise ValueError("GRPO needs temperature > 0 (the ratio is undefined for greedy rollouts)")

    run_dir = Path(cfg.out_dir) / cfg.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    sampler = StreamSampler(cfg)
    epoch_len = len(sampler.items) // (cfg.prompts_per_iter * cfg.overprovision)
    max_iters = int(round(cfg.epochs * epoch_len)) if cfg.epochs > 0 else cfg.max_iters
    logger.info("stream: %d prompts, %d iters/epoch, %d iters total", len(sampler.items), epoch_len, max_iters)

    torch.manual_seed(cfg.seed)
    adapter = load_adapter(cfg.model, cfg.checkpoint or None, device)
    ref = load_adapter(cfg.model, cfg.checkpoint or None, device)
    for m in (adapter.model, ref.model):
        for prm in m.parameters():
            prm.requires_grad_(False)
    controller = SVPController(adapter.model, adapter.arch, cfg.target, cfg.alpha)
    ref_controller = SVPController(ref.model, ref.arch, cfg.target, cfg.alpha)
    trainable = setup_trainable(controller)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        controller.import_weights(ck["weights"])
        controller.refresh_basis()
        optimizer.load_state_dict(ck["opt"])
        sampler.load_state(ck["sampler"])
        start = int(ck["iter"])
        logger.info("resumed from %s at iter %d", args.resume, start)
    ref_controller.import_factors(controller.export_factors())

    run_logger = RunLogger(run_dir, cfg.to_dict(), is_main, cfg.wandb_project, cfg.wandb_entity)
    B, G = cfg.prompts_per_iter, cfg.group_size

    def maybe_eval(k: int, final: bool = False) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False  # evals stay fp32
        if k % cfg.k_eval == 0 or final:
            m = run_eval(adapter, controller, cfg, rank, world)
            run_logger.log(k, m)
            logger.info("eval @%d: %s", k, json.dumps(m))
        if cfg.bench_at_epoch and ((k > 0 and k % epoch_len == 0) or final):
            m = run_bench(adapter, cfg, rank, world)
            run_logger.log(k, m)
            logger.info("bench @%d: %s", k, json.dumps(m))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)

    k = start
    try:
        for k in range(start, max_iters):
            maybe_eval(k)
            if is_main and k > 0 and k % epoch_len == 0:
                save_checkpoint(run_dir / f"ckpt_epoch{k // epoch_len}.pt", k, cfg, controller, optimizer, sampler)
            if k > 0 and k % cfg.k_svd == 0:
                controller.refresh_basis()
                ref_controller.import_factors(controller.export_factors())

            t0 = time.time()
            picks = sampler.next(B * cfg.overprovision)
            my_indices = list(range(rank, len(picks), world))
            records = []
            for s in range(0, len(my_indices), cfg.rollout_batch):
                idxs = my_indices[s: s + cfg.rollout_batch]
                records.extend(rollout_groups(adapter, controller, [picks[i] for i in idxs], cfg, k))
            flags = torch.zeros(len(picks), device=device)
            for i, rec in zip(my_indices, records, strict=True):
                flags[i] = float(0 < sum(rec["corrects"]) < G)
            if world > 1:
                dist.all_reduce(flags, op=dist.ReduceOp.SUM)
            kept_idx = [i for i in range(len(picks)) if flags[i] > 0][:B]
            kept_set = set(kept_idx)
            my_kept = [rec for i, rec in zip(my_indices, records, strict=True) if i in kept_set]
            n_kept = len(kept_idx)
            n_correct_all = [sum(r["corrects"]) for r in records]
            rollout_s = time.time() - t0
            if n_kept == 0:
                run_logger.log(k, {"group/kept": 0, "time/rollout_s": rollout_s})
                continue

            t1 = time.time()
            chunks = [my_kept[i: i + cfg.groups_per_pass] for i in range(0, len(my_kept), cfg.groups_per_pass)]
            with torch.no_grad():
                ref_logps = [rep for ch in chunks for rep in _split(replay(ref, ref_controller, ch, cfg.latent_steps), G)]
            old_logps: list[torch.Tensor | None] = [None] * len(my_kept)
            sums = {"pg": 0.0, "kl": 0.0, "ent": 0.0, "clip": 0.0, "gap": 0.0}
            grad_norm = 0.0
            for _ in range(cfg.mu):
                optimizer.zero_grad(set_to_none=True)
                gi = 0
                for ch in chunks:
                    reps = _split(replay(adapter, controller, ch, cfg.latent_steps), G)
                    chunk_loss = torch.zeros((), device=device)
                    for rec, rep in zip(ch, reps, strict=True):
                        if old_logps[gi] is None:
                            old_logps[gi] = rep["token_logp"].detach()
                            rl = rec["rollout_logp"].to(device)[:, : rep["token_logp"].shape[1]]
                            sums["gap"] += float(((old_logps[gi] - rl).abs() * rep["mask"]).sum() / rep["mask"].sum())
                        rewards = torch.tensor(rec["corrects"], dtype=torch.float32, device=device)
                        adv = rewards - rewards.mean()
                        pg, clip_frac = surrogate(rep["token_logp"], old_logps[gi], adv, rep["mask"], cfg.clip_eps)
                        kl = k3_kl(ref_logps[gi]["token_logp"], rep["token_logp"], rep["mask"], cfg.kl_clamp)
                        chunk_loss = chunk_loss + pg + cfg.beta_kl * kl
                        sums["pg"] += float(pg.detach())
                        sums["kl"] += float(kl.detach())
                        sums["ent"] += rep["entropy"]
                        sums["clip"] += clip_frac
                        gi += 1
                    (chunk_loss / n_kept).backward()
                if world > 1:
                    for prm in trainable:
                        if prm.grad is None:
                            prm.grad = torch.zeros_like(prm)
                        dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    trainable, cfg.grad_clip if cfg.grad_clip > 0 else float("inf")))
                optimizer.step()
            update_s = time.time() - t1

            keys = ("pg", "kl", "ent", "clip", "gap")
            reduced = all_reduce_sum([sums[x] for x in keys] + [
                sum(sum(r["corrects"]) for r in my_kept), len(records),
                sum(1 for c in n_correct_all if c == G), sum(1 for c in n_correct_all if c == 0)], device)
            denom = n_kept * cfg.mu
            if is_main:
                run_logger.log(k, {
                    "loss/pg": reduced[0] / denom, "loss/kl": reduced[1] / denom,
                    "policy/entropy": reduced[2] / denom, "policy/clip_frac": reduced[3] / denom,
                    "diag/replay_gap": reduced[4] / n_kept,
                    "reward/outcome_rate": reduced[5] / (n_kept * G),
                    "group/kept": n_kept, "group/mixed": int(flags.sum()),
                    "group/all_correct": reduced[7], "group/all_wrong": reduced[8],
                    "grad_norm": grad_norm, "lr": cfg.lr,
                    "time/rollout_s": rollout_s, "time/update_s": update_s,
                })
                if k % 20 == 0:
                    logger.info("iter %d/%d pg %.4f kl %.5f acc %.3f kept %d rollout %.1fs update %.1fs",
                                k, max_iters, reduced[0] / denom, reduced[1] / denom,
                                reduced[5] / (n_kept * G), n_kept, rollout_s, update_s)
                if (k + 1) % cfg.k_ckpt == 0:
                    save_checkpoint(run_dir / "ckpt_last.pt", k + 1, cfg, controller, optimizer, sampler)
            if world > 1:
                dist.barrier()
        k = max_iters
        maybe_eval(k, final=True)
    finally:
        if is_main:
            save_checkpoint(run_dir / "ckpt_final.pt", k, cfg, controller, optimizer, sampler)
            run_logger.finish()
        if world > 1:
            dist.destroy_process_group()


def _split(rep: dict, G: int) -> list[dict]:
    K = rep["token_logp"].shape[0] // G
    return [{"token_logp": rep["token_logp"][j * G:(j + 1) * G], "mask": rep["mask"][j * G:(j + 1) * G],
             "entropy": rep["entropy"][j]} for j in range(K)]


if __name__ == "__main__":
    main()
