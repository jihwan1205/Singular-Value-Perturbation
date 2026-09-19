"""pass@k evaluation of a latent-reasoning model under SVP.

    python -m svp.evaluate --model coconut --alpha 0.6 --n_samples 16 --seed 0
    python -m svp.evaluate --model coconut --checkpoint jihwan1205/svp-v-coconut-gpt2 --n_samples 1

``--alpha 0`` (or ``--n_samples 1`` with the default alpha) is clean greedy
decoding. ``--temperature``/``--top_p`` add answer-token sampling instead of,
or on top of, weight perturbation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import BENCHMARKS, Example, benchmark_path, load_examples
from .engine import GenerationSettings, generate_batch
from .metrics import coverage, is_correct, majority_vote_accuracy, pass_at_k_curve, sample_accuracy
from .models import MODELS, load_adapter
from .perturb import TARGETS, SVPController


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(adapter, examples: list[Example], settings: GenerationSettings,
             controller: SVPController | None = None, save_texts: bool = False,
             log_every: int = 0) -> tuple[dict, list[dict]]:
    """Run ``settings.n_samples`` completions per example and score them.

    Returns:
        ``(metrics, records)``; records are per example (in input order) with
        extracted answers and correctness flags.
    """
    lengths = [len(adapter.tokenizer(adapter.build_prompt(e.question))["input_ids"]) for e in examples]
    order = sorted(range(len(examples)), key=lambda i: lengths[i])
    n, bs = settings.n_samples, settings.batch_size
    total_rows = len(examples) * n
    records: dict[int, dict] = {}
    start = time.time()
    n_chunks = (len(examples) + bs - 1) // bs
    for ci in range(n_chunks):
        idxs = order[ci * bs:(ci + 1) * bs]
        chunk = [examples[i] for i in idxs]
        out = generate_batch(adapter, [e.question for e in chunk], settings, controller,
                             row_offset=ci * bs * n, total_rows=total_rows)
        for qi, (i, ex) in enumerate(zip(idxs, chunk, strict=True)):
            texts = out.generated_texts[qi * n:(qi + 1) * n]
            answers = [adapter.extract_answer(t) for t in texts]
            rec = {"idx": ex.idx, "gold": ex.gold, "answers": answers,
                   "corrects": [is_correct(a, ex.gold) for a in answers]}
            if save_texts:
                rec["texts"] = texts
            records[i] = rec
        if log_every and (ci % log_every == 0 or ci == n_chunks - 1):
            done = [r for r in records.values()]
            acc = sum(sum(r["corrects"]) for r in done) / max(sum(len(r["corrects"]) for r in done), 1)
            print(f"[chunk {ci + 1}/{n_chunks}] sample_acc={acc:.4f} elapsed={time.time() - start:.0f}s", flush=True)
    ordered = [records[i] for i in range(len(examples))]
    corrects = [r["corrects"] for r in ordered]
    metrics = {
        **pass_at_k_curve(corrects, n),
        "coverage": coverage(corrects),
        "sample_accuracy": sample_accuracy(corrects),
        "majority_vote": majority_vote_accuracy([r["answers"] for r in ordered], [e.gold for e in examples]),
        "n_examples": len(examples),
        "n_samples": n,
        "elapsed_sec": round(time.time() - start, 1),
    }
    return metrics, ordered


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--checkpoint", default=None, help="HF hub id or local dir overriding the model default")
    p.add_argument("--benchmarks", nargs="+", default=["all"],
                   help=f"benchmark names ({', '.join(BENCHMARKS)}), 'all', or JSON paths")
    p.add_argument("--target", default="attn_v", choices=list(TARGETS))
    p.add_argument("--alpha", type=float, default=0.0, help="SVP scale; 0 disables perturbation")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8, help="questions per forward (rows = batch_size * n_samples)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--latent_steps", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--save_texts", action="store_true")
    p.add_argument("--out", default=None, help="output dir (default: results/<auto>)")
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.alpha < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("alpha and temperature must be >= 0, top_p in (0, 1]")
    if args.n_samples > 1 and args.alpha == 0 and args.temperature == 0:
        raise ValueError("n_samples > 1 needs --alpha > 0 and/or --temperature > 0; greedy samples are identical")
    benches = list(BENCHMARKS) if args.benchmarks == ["all"] else args.benchmarks
    adapter = load_adapter(args.model, args.checkpoint, device=args.device)
    controller = SVPController(adapter.model, adapter.arch, args.target, args.alpha) if args.alpha > 0 else None
    settings = GenerationSettings(
        latent_steps=args.latent_steps or adapter.default_latent_steps,
        max_new_tokens=args.max_new_tokens or adapter.default_max_new_tokens,
        n_samples=args.n_samples, batch_size=args.batch_size,
        temperature=args.temperature, top_p=args.top_p, sample_seed=args.seed,
    )
    tag = args.checkpoint.rstrip("/").split("/")[-1] if args.checkpoint else args.model
    noise = f"svp_{args.target}_a{args.alpha:g}" if args.alpha > 0 else "clean"
    if args.temperature > 0:
        noise += f"_T{args.temperature:g}_p{args.top_p:g}"
    out_dir = Path(args.out) if args.out else Path("results") / f"{tag}__{noise}__n{args.n_samples}__seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    summary = {}
    for bench in benches:
        set_seed(args.seed)
        examples = load_examples(benchmark_path(bench), args.max_examples)
        metrics, records = evaluate(adapter, examples, settings, controller, args.save_texts, log_every=20)
        bench_dir = out_dir / Path(bench).stem
        bench_dir.mkdir(exist_ok=True)
        (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with open(bench_dir / "samples.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summary[Path(bench).stem] = metrics
        print(f"{bench}: {json.dumps(metrics)}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
