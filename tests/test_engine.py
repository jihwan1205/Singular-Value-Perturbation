import torch

from svp.engine import GenerationSettings, generate_batch
from svp.evaluate import evaluate
from svp.data import Example
from svp.perturb import SVPController
from tests.conftest import QUESTIONS


def _settings(**kw):
    base = dict(latent_steps=3, max_new_tokens=12, n_samples=1, batch_size=8)
    base.update(kw)
    return GenerationSettings(**base)


def test_greedy_is_deterministic_and_shapes(gpt2_adapter):
    s = _settings(return_token_logps=True)
    a = generate_batch(gpt2_adapter, QUESTIONS, s)
    b = generate_batch(gpt2_adapter, QUESTIONS, s)
    assert torch.equal(a.sequences, b.sequences)
    assert a.sequences.shape[0] == len(QUESTIONS)
    assert (a.sequences[:, -12:-9] == gpt2_adapter.latent_token_id).all()
    assert (a.sequences[:, -9] == gpt2_adapter.end_token_id).all()
    assert a.token_logps.shape == (len(QUESTIONS), 12 - 3 - 1)
    assert (a.token_logps <= 0).all()


def test_svp_samples_differ_but_layout_is_shared(gpt2_adapter):
    ctrl = SVPController(gpt2_adapter.model, "gpt2", "attn_v", alpha=2.0)
    torch.manual_seed(0)
    out = generate_batch(gpt2_adapter, QUESTIONS[:1], _settings(n_samples=4), ctrl)
    assert out.sequences.shape[0] == 4
    assert ctrl.codes.shape == (4, 2, 32)
    prompt_len = out.sequences.shape[1] - 12
    assert torch.equal(out.sequences[:, :prompt_len], out.sequences[:1, :prompt_len].expand(4, -1))
    assert len({tuple(r.tolist()) for r in out.sequences}) > 1


def test_sampling_is_batch_size_invariant(gpt2_adapter):
    s = _settings(temperature=1.0, top_p=0.9, sample_seed=3)
    full = generate_batch(gpt2_adapter, QUESTIONS[:2], s, total_rows=2)
    parts = [generate_batch(gpt2_adapter, [q], s, row_offset=i, total_rows=2) for i, q in enumerate(QUESTIONS[:2])]
    for i in range(2):
        assert full.generated_texts[i] == parts[i].generated_texts[0]


def test_evaluate_metrics_and_order(gpt2_adapter):
    examples = [Example(i, q, float(i)) for i, q in enumerate(QUESTIONS)]
    metrics, records = evaluate(gpt2_adapter, examples, _settings(n_samples=2, batch_size=2))
    assert [r["idx"] for r in records] == [0, 1, 2, 3]
    assert set(metrics) >= {"pass@1", "pass@2", "coverage", "majority_vote", "sample_accuracy"}
    assert metrics["n_examples"] == 4 and metrics["n_samples"] == 2
