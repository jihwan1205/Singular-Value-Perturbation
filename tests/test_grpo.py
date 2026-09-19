import json

import torch

from svp.grpo.config import GrpoConfig
from svp.grpo.replay import replay
from svp.grpo.rollout import rollout_groups
from svp.grpo.train import StreamSampler, setup_trainable, surrogate, k3_kl
from svp.perturb import SVPController
from tests.conftest import QUESTIONS


def _cfg(**kw):
    base = dict(latent_steps=3, extra_decode=-52, group_size=4, temperature=1.0, seed=0)  # budget 12
    base.update(kw)
    return GrpoConfig(**base)


def _items():
    return [{"idx": i, "question": q, "gold": float(i)} for i, q in enumerate(QUESTIONS[:2])]


def test_replay_reproduces_rollout_logps(gpt2_adapter):
    cfg = _cfg()
    ctrl = SVPController(gpt2_adapter.model, "gpt2", "attn_v", alpha=0.8)
    records = rollout_groups(gpt2_adapter, ctrl, _items(), cfg, iteration=0)
    assert len(records) == 2 and records[0]["codes"].shape == (4, 2, 32)
    for rec in records:
        rep = replay(gpt2_adapter, ctrl, [rec], cfg.latent_steps)
        L = rep["token_logp"].shape[1]
        gap = ((rep["token_logp"].detach() - rec["rollout_logp"][:, :L]).abs() * rep["mask"]).max()
        assert gap < 1e-4
        assert rep["mask"].sum(1).tolist() == [len(a) for a in rec["answer_ids"]]
    fused = replay(gpt2_adapter, ctrl, records, cfg.latent_steps)
    single = [replay(gpt2_adapter, ctrl, [r], cfg.latent_steps) for r in records]
    for j, s in enumerate(single):
        L = s["token_logp"].shape[1]
        assert torch.allclose(fused["token_logp"][j * 4:(j + 1) * 4, :L] * s["mask"],
                              s["token_logp"] * s["mask"], atol=1e-4)


def test_grad_mask_and_objective(gpt2_adapter):
    ctrl = SVPController(gpt2_adapter.model, "gpt2", "attn_v", alpha=0.5)
    params = setup_trainable(ctrl)
    assert len(params) == 2
    rec = rollout_groups(gpt2_adapter, ctrl, _items()[:1], _cfg(), iteration=1)[0]
    rep = replay(gpt2_adapter, ctrl, [rec], 3)
    old = rep["token_logp"].detach()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss, clip_frac = surrogate(rep["token_logp"], old, adv, rep["mask"], 0.2)
    expected = -((adv.unsqueeze(1) * rep["mask"]).sum() / 4)
    assert torch.allclose(loss, expected) and clip_frac == 0.0
    kl = k3_kl(old, rep["token_logp"], rep["mask"], 10.0)
    assert float(kl.detach()) == 0.0
    (loss + kl).backward()
    g = params[0].grad
    assert g is not None and g[:, :64].abs().sum() == 0 and g[:, 64:].abs().sum() > 0


def test_stream_sampler_resume(tmp_path):
    data = [{"question": f"q{i}", "answer": str(i) if i % 7 else "1/3"} for i in range(60)]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    cfg = GrpoConfig(train_data=str(path), stream_skip=10, seed=3)
    a = StreamSampler(cfg)
    assert all(it["idx"] >= 10 for it in a.items) and len(a.items) == 50 - 7
    a.next(20)
    a.next(20)
    state = a.state()
    expected = [a.next(8) for _ in range(3)]  # crosses the pass boundary
    b = StreamSampler(cfg)
    b.load_state(state)
    assert [b.next(8) for _ in range(3)] == expected
