"""SVP-V-GRPO configuration."""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class GrpoConfig:
    # model / rollout
    model: str = "coconut"
    checkpoint: str = ""            # base checkpoint override ("" = registry default)
    latent_steps: int = 6
    target: str = "attn_v"          # perturbed AND trained weight
    alpha: float = 0.6              # SVP scale for rollouts and for the pass@k eval
    group_size: int = 32            # G rollouts per prompt
    prompts_per_iter: int = 8       # B kept (mixed-outcome) groups per update
    overprovision: int = 2          # prompts rolled out per iter = B * overprovision
    rollout_batch: int = 16         # prompts per generate_batch call
    temperature: float = 1.0        # answer-token sampling temperature
    top_p: float = 1.0
    extra_decode: int = 8           # rollout budget = 64 + extra_decode tokens
    # objective
    beta_kl: float = 0.02           # k3 KL to the frozen base policy
    clip_eps: float = 0.2
    kl_clamp: float = 10.0
    mu: int = 2                     # inner update epochs per rollout batch
    groups_per_pass: int = 1        # kept groups fused into one replay forward
    # optimization
    lr: float = 3e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    tf32: bool = True
    epochs: float = 10.0            # passes over the training stream (derives max_iters)
    max_iters: int = 0              # used only when epochs == 0
    k_svd: int = 50                 # re-SVD period (iterations)
    k_eval: int = 500               # GSM8K pass@1 / pass@16 eval period
    k_ckpt: int = 2000              # rolling checkpoint period
    bench_at_epoch: bool = True     # six-benchmark greedy eval at every epoch boundary
    seed: int = 1
    # data
    train_data: str = "data/gsm_train.json"
    stream_skip: int = 1536         # leading training rows held out
    eval_data: str = "gsm8k"
    eval_samples: int = 16
    eval_batch: int = 16            # questions per forward in the pass@k eval
    # output
    out_dir: str = "results/grpo"
    run_name: str = "svp-v-coconut-gpt2"
    wandb_project: str = ""         # "" disables wandb
    wandb_entity: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse(field_type: type, value: str):
    if field_type is bool:
        return value.lower() in ("1", "true", "yes")
    return field_type(value)


def add_config_args(parser) -> None:
    types = typing.get_type_hints(GrpoConfig)
    for f in fields(GrpoConfig):
        parser.add_argument(f"--{f.name}", default=None, type=lambda v, t=types[f.name]: _parse(t, v))


def build_config(config_path: str | None, args) -> GrpoConfig:
    """Defaults < JSON config file < CLI flags."""
    values: dict = {}
    if config_path:
        file_values = json.loads(Path(config_path).read_text())
        unknown = set(file_values) - {f.name for f in fields(GrpoConfig)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        values.update(file_values)
    for f in fields(GrpoConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    return GrpoConfig(**values)
