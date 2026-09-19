"""Singular Value Perturbation (SVP) on a named weight target.

For a target weight ``W = U diag(s) V^T`` (thin SVD, computed once per layer),
each sample row ``i`` and each perturbed layer ``l`` draw an independent code
``g_{i,l} ~ N(0, I_r)`` and use the weight ``s_j -> s_j (1 + alpha g_{i,l,j})``.
The perturbed output is computed without materializing the weight:
``y = x W + alpha * ((x U) * (s * g)) V^T``, so one batched forward serves all
rows with distinct perturbations.

One draw per sample is held for the whole trajectory (prompt prefill and every
latent step); the controller is switched off before the answer is decoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

TARGETS: dict[str, dict[str, tuple[str, int | None]]] = {
    "attn_q": {"gpt2": ("attn.c_attn", 0), "llama": ("self_attn.q_proj", None)},
    "attn_k": {"gpt2": ("attn.c_attn", 1), "llama": ("self_attn.k_proj", None)},
    "attn_v": {"gpt2": ("attn.c_attn", 2), "llama": ("self_attn.v_proj", None)},
    "attn_o": {"gpt2": ("attn.c_proj", None), "llama": ("self_attn.o_proj", None)},
    "mlp_up": {"gpt2": ("mlp.c_fc", None), "llama": ("mlp.up_proj", None)},
    "mlp_down": {"gpt2": ("mlp.c_proj", None), "llama": ("mlp.down_proj", None)},
}


@dataclass
class ResolvedTarget:
    parent: nn.Module
    attr: str
    module: nn.Module
    out_slice: slice | None


def transformer_blocks(model: nn.Module, arch: str) -> list[nn.Module]:
    if arch == "gpt2":
        return list(model.transformer.h)
    if arch == "llama":
        return list(model.model.layers)
    raise ValueError(f"unknown arch {arch!r}")


def resolve_targets(model: nn.Module, arch: str, target: str) -> list[ResolvedTarget]:
    """One entry per transformer block for the named target."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGETS)}")
    path, component = TARGETS[target][arch]
    resolved = []
    for block in transformer_blocks(model, arch):
        parent = block
        *parents, attr = path.split(".")
        for p in parents:
            parent = getattr(parent, p)
        module = getattr(parent, attr)
        out_slice = None
        if component is not None:  # GPT-2 fuses Q/K/V into one Conv1D (in, 3*out)
            d = module.weight.shape[-1] // 3
            out_slice = slice(component * d, (component + 1) * d)
        resolved.append(ResolvedTarget(parent, attr, module, out_slice))
    return resolved


def effective_weight(module: nn.Module, out_slice: slice | None) -> torch.Tensor:
    """The ``(in, out)`` matrix the target names (``nn.Linear`` stores ``(out, in)``)."""
    if isinstance(module, nn.Linear):
        w = module.weight.t()
    elif hasattr(module, "nf"):  # transformers Conv1D stores (in, out)
        w = module.weight
    else:
        raise ValueError(f"unsupported module {type(module).__name__}")
    return w if out_slice is None else w[:, out_slice]


class PerturbedModule(nn.Module):
    """Wraps one target module and adds the per-row SVP delta to its output."""

    def __init__(self, target: ResolvedTarget, alpha: float, controller: SVPController, index: int):
        super().__init__()
        self.base = target.module
        self.out_slice = target.out_slice
        self.alpha = alpha
        self.controller = controller
        self.index = index
        self.register_buffer("U", torch.empty(0), persistent=False)
        self.register_buffer("V", torch.empty(0), persistent=False)
        self.register_buffer("S", torch.empty(0), persistent=False)
        self.refresh_basis()

    @property
    def weight(self) -> torch.Tensor:
        return effective_weight(self.base, self.out_slice)

    @property
    def rank(self) -> int:
        return self.S.shape[0]

    @torch.no_grad()
    def refresh_basis(self) -> None:
        """Recompute the SVD of the current weight (needed after weight updates)."""
        w = self.weight.detach()
        u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
        self.U = u.to(w.dtype)
        self.V = vh.t().to(w.dtype)
        self.S = s.to(w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.controller.active:
            return y
        codes = self.controller.codes
        if codes is None or codes.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"controller has codes for {None if codes is None else codes.shape[0]} rows, "
                f"input has {x.shape[0]}; call begin_batch(rows) first"
            )
        code = (self.alpha * self.S * codes[:, self.index].to(self.S.dtype)).to(x.dtype)
        x3 = x if x.dim() == 3 else x.unsqueeze(1)
        delta = ((x3 @ self.U.to(x.dtype)) * code.unsqueeze(1)) @ self.V.t().to(x.dtype)
        if x.dim() == 2:
            delta = delta.squeeze(1)
        if self.out_slice is None:
            return y + delta.to(y.dtype)
        y = y.clone()
        y[..., self.out_slice] += delta.to(y.dtype)
        return y


class SVPController:
    """Installs :class:`PerturbedModule` wrappers and owns the per-batch codes.

    Args:
        model: Backbone to wrap in place (see :meth:`remove`).
        arch: ``"gpt2"`` or ``"llama"``.
        target: A key of :data:`TARGETS`.
        alpha: Perturbation scale.
    """

    def __init__(self, model: nn.Module, arch: str, target: str, alpha: float):
        self.target = target
        self.alpha = float(alpha)
        self.active = False
        self.codes: torch.Tensor | None = None
        self.targets = resolve_targets(model, arch, target)
        self.wrappers: list[PerturbedModule] = []
        for i, t in enumerate(self.targets):
            wrapper = PerturbedModule(t, self.alpha, self, i)
            setattr(t.parent, t.attr, wrapper)
            self.wrappers.append(wrapper)
        ranks = {w.rank for w in self.wrappers}
        assert len(ranks) == 1, f"layers of target {target} differ in rank: {ranks}"
        self.rank = ranks.pop()
        self.n_layers = len(self.wrappers)
        self.device = self.wrappers[0].base.weight.device

    def begin_batch(self, rows: int, codes: torch.Tensor | None = None) -> None:
        """Draw fresh ``(rows, n_layers, rank)`` codes, or install the given ones."""
        if codes is None:
            codes = torch.randn(rows, self.n_layers, self.rank, device=self.device)
        elif tuple(codes.shape) != (rows, self.n_layers, self.rank):
            raise ValueError(f"codes shape {tuple(codes.shape)} != {(rows, self.n_layers, self.rank)}")
        self.codes = codes.to(self.device, torch.float32)

    def set_active(self, active: bool) -> None:
        self.active = active

    def end_batch(self) -> None:
        self.active = False

    def refresh_basis(self) -> None:
        for w in self.wrappers:
            w.refresh_basis()

    def export_factors(self) -> list[dict[str, torch.Tensor]]:
        return [{"U": w.U.clone(), "V": w.V.clone(), "S": w.S.clone()} for w in self.wrappers]

    def import_factors(self, factors: list[dict[str, torch.Tensor]]) -> None:
        for w, f in zip(self.wrappers, factors, strict=True):
            w.U = f["U"].to(w.U.device, w.U.dtype)
            w.V = f["V"].to(w.V.device, w.V.dtype)
            w.S = f["S"].to(w.S.device, w.S.dtype)

    def export_weights(self) -> list[torch.Tensor]:
        """The target ``(in, out)`` matrices, one per layer (cpu, fp32)."""
        return [w.weight.detach().to("cpu", torch.float32).clone() for w in self.wrappers]

    @torch.no_grad()
    def import_weights(self, weights: list[torch.Tensor]) -> None:
        for w, new in zip(self.wrappers, weights, strict=True):
            new = new.to(w.base.weight.device, w.base.weight.dtype)
            if isinstance(w.base, nn.Linear):
                w.base.weight.copy_(new.t())
            elif w.out_slice is None:
                w.base.weight.copy_(new)
            else:
                w.base.weight[:, w.out_slice].copy_(new)

    def remove(self) -> None:
        for t, w in zip(self.targets, self.wrappers, strict=True):
            setattr(t.parent, t.attr, w.base)
        self.wrappers = []
