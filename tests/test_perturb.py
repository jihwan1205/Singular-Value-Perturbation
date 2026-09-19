import pytest
import torch

from svp.perturb import SVPController, TARGETS, effective_weight, resolve_targets


def _codes(controller, rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, controller.n_layers, controller.rank, generator=g)


@pytest.mark.parametrize("target", list(TARGETS))
def test_delta_matches_materialized_weight_gpt2(gpt2_adapter, target):
    model = gpt2_adapter.model
    targets = resolve_targets(model, "gpt2", target)
    bases = [t.module for t in targets]
    weights = [effective_weight(t.module, t.out_slice).detach().clone() for t in targets]
    ctrl = SVPController(model, "gpt2", target, alpha=0.3)
    rows, T = 3, 5
    codes = _codes(ctrl, rows)
    ctrl.begin_batch(rows, codes)
    ctrl.set_active(True)
    for layer, (w, base, t) in enumerate(zip(ctrl.wrappers, bases, targets, strict=True)):
        x = torch.randn(rows, T, base.weight.shape[0] if hasattr(base, "nf") else base.in_features)
        y = w(x)
        expected = base(x).clone()
        u, s, vh = torch.linalg.svd(weights[layer], full_matrices=False)
        for i in range(rows):
            w_i = u @ torch.diag(s * (1 + 0.3 * codes[i, layer])) @ vh
            ref = x[i] @ w_i + (base.bias[t.out_slice] if t.out_slice is not None else base.bias)
            if t.out_slice is None:
                expected[i] = ref
            else:
                expected[i, :, t.out_slice] = ref
        assert torch.allclose(y, expected, atol=1e-4), f"layer {layer}"
    ctrl.remove()
    for t, base in zip(targets, bases, strict=True):
        assert getattr(t.parent, t.attr) is base


def test_delta_matches_materialized_weight_llama(llama_adapter):
    model = llama_adapter.model
    targets = resolve_targets(model, "llama", "attn_v")
    ctrl = SVPController(model, "llama", "attn_v", alpha=0.5)
    rows = 2
    codes = _codes(ctrl, rows)
    ctrl.begin_batch(rows, codes)
    ctrl.set_active(True)
    for layer, (w, t) in enumerate(zip(ctrl.wrappers, targets, strict=True)):
        x = torch.randn(rows, t.module.in_features)
        u, s, vh = torch.linalg.svd(t.module.weight.detach().t(), full_matrices=False)
        expected = torch.stack([x[i] @ (u @ torch.diag(s * (1 + 0.5 * codes[i, layer])) @ vh) for i in range(rows)])
        assert torch.allclose(w(x), expected, atol=1e-4)


def test_inactive_is_identity_and_layers_are_independent(gpt2_adapter):
    model = gpt2_adapter.model
    x = torch.randn(4, 3, 32)
    base_out = model.transformer.h[0].attn.c_attn(x).clone()
    ctrl = SVPController(model, "gpt2", "attn_v", alpha=1.0)
    torch.manual_seed(1)
    ctrl.begin_batch(4)
    assert ctrl.codes.shape == (4, 2, 32)
    assert not torch.allclose(ctrl.codes[:, 0], ctrl.codes[:, 1])
    assert torch.equal(model.transformer.h[0].attn.c_attn(x), base_out)
    ctrl.set_active(True)
    out = model.transformer.h[0].attn.c_attn(x)
    d = 32
    assert torch.equal(out[..., : 2 * d], base_out[..., : 2 * d])  # Q, K untouched
    assert not torch.allclose(out[..., 2 * d:], base_out[..., 2 * d:])
    ctrl.end_batch()
    assert not ctrl.active
    with pytest.raises(RuntimeError):
        ctrl.set_active(True)
        model.transformer.h[0].attn.c_attn(torch.randn(5, 3, 32))


def test_weight_roundtrip_and_refresh(gpt2_adapter):
    ctrl = SVPController(gpt2_adapter.model, "gpt2", "attn_v", alpha=0.1)
    weights = ctrl.export_weights()
    assert len(weights) == 2 and weights[0].shape == (32, 32)
    new = [w + 0.5 for w in weights]
    ctrl.import_weights(new)
    assert torch.allclose(ctrl.export_weights()[1], new[1])
    old_s = ctrl.wrappers[0].S.clone()
    ctrl.refresh_basis()
    assert not torch.allclose(ctrl.wrappers[0].S, old_s)
    factors = ctrl.export_factors()
    ctrl.import_factors(factors)
    assert torch.equal(ctrl.wrappers[0].S, factors[0]["S"])
