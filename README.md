# Singular Value Perturbation

Parallel test-time scaling for latent reasoning models (Coconut, CODI), and SVP-V-GRPO.

SVP samples reasoning trajectories by perturbing weights: for a target weight
`W = U diag(s) Vᵀ`, each sample draws `g ~ N(0, I)` per layer and runs the prompt
prefill and every latent step with `s → s (1 + α g)`; the answer is decoded from clean
weights. Default target: the attention value projection (`attn_v`). SVP-V-GRPO uses the
same distribution for GRPO rollouts and trains only the target weights.

## Setup

```bash
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python scripts/convert_codi.py --model gpt2      # -> checkpoints/codi-gpt2
python scripts/convert_codi.py --model llama1b   # -> checkpoints/codi-llama1b
bash scripts/download_data.sh                    # gsm_train.json, GRPO only
python -m pytest tests/
```

Coconut (`ModalityDance/latent-tts-coconut`) is fetched from the Hub on first use.
The six benchmarks (GSM8K, GSM-Hard, MultiArith, SVAMP, ASDiv-A, GSM-Plus) are in `data/`.

## Evaluation

```bash
python -m svp.evaluate --model coconut --n_samples 1                       # greedy
python -m svp.evaluate --model coconut --alpha 0.6 --n_samples 16 --seed 0  # SVP pass@16
python -m svp.evaluate --model codi-llama1b --target attn_k --alpha 0.5 --n_samples 16 --benchmarks gsm8k gsm_hard
python -m svp.evaluate --model coconut --checkpoint checkpoints/svp-v-coconut-gpt2 --n_samples 1
```

Models: `coconut`, `codi-gpt2`, `codi-llama1b`. Targets: `attn_q`, `attn_k`, `attn_v`,
`attn_o`, `mlp_up`, `mlp_down`. Results (pass@k for k ≤ n, coverage, majority vote) go to
`results/<run>/`. Expected greedy GSM8K: 34.12 / 42.46 / 55.57; SVP pass@16 (α 0.6 / 0.5 / 0.5):
≈ 57 / 61 / 71. The paper shared one code across layers; this code draws one per layer.

### Released checkpoint

The SVP-V-GRPO checkpoint is mirrored anonymously for review
([browse](https://anonymous-hf.com/a/uoc7l9fxgpdc/)). It is a plain HuggingFace model
directory — SVP is a training-time exploration mechanism, and deployment is ordinary
greedy decoding.

```bash
curl -L --retry 5 --retry-all-errors https://anonymous-hf.com/api/a/uoc7l9fxgpdc/download/ \
  -o svp-v-coconut-gpt2.zip
unzip svp-v-coconut-gpt2.zip -d checkpoints/svp-v-coconut-gpt2   # ~480 MB
python -m svp.evaluate --model coconut --checkpoint checkpoints/svp-v-coconut-gpt2 --n_samples 1
```

It is the state of the art among Coconut-family latent reasoning models (GPT-2 124M) on
all six benchmarks, ahead of the published Coconut variants and of CODI. Clean greedy
accuracy (%), 64-token budget:

| Model (GPT-2 124M) | GSM8K | GSM-Hard | SVAMP | ASDiv-A | MultiArith | GSM-Plus |
|---|---|---|---|---|---|---|
| Coconut | 34.1 | 7.7 | 35.6 | 60.2 | 80.9 | 17.4 |
| SLPO-Coconut | 34.9 | 7.6 | 34.3 | 58.7 | 82.8 | 18.2 |
| SIM-CoT (Coconut) | 44.7 | 9.3 | 40.6 | 67.2 | 90.5 | 21.5 |
| CODI | 42.5 | 9.3 | 40.0 | 65.4 | 91.9 | 23.1 |
| **SVP-V-GRPO (this checkpoint)** | **50.5** | **11.2** | **45.2** | **72.5** | **94.1** | **28.3** |

This checkpoint is what `configs/grpo_coconut_gpt2.json` reproduces: B = 8, lr 3e-5,
seed 1, 10 epochs, SVP α 0.6 (epoch 9 has the best `eval/pass@1`).

### Exploration controls

The control arms are trained with a **different, larger recipe** (B = 32, lr 6e-5,
seed 0, 15 epochs, reported at epoch 9), so they are not comparable to the row above.
Under that recipe all arms share every setting except the rollout exploration, which is
the comparison the paper makes:

| Rollout exploration (B = 32, lr 6e-5, seed 0, epoch 9) | GSM8K | GSM-Hard | SVAMP | ASDiv-A | MultiArith | GSM-Plus |
|---|---|---|---|---|---|---|
| answer-token sampling, T = 1 | 38.4 | 8.7 | 37.1 | 61.5 | 85.3 | 19.8 |
| entry-wise Gaussian on `W_V`, σ = 0.06 | 40.6 | 9.0 | 39.1 | 62.7 | 88.6 | 21.9 |
| native dropout, p = 0.2 | 45.2 | 9.3 | **44.6** | 71.1 | 90.9 | 24.8 |
| **SVP on `W_V`, α = 0.6 (ours)** | **50.3** | **11.3** | 43.6 | **71.2** | **93.4** | **27.6** |

The trainer in this repo implements the SVP arm; only the SVP-V checkpoint is mirrored
for review, and the control arms are described by the numbers above.

## SVP-V-GRPO

One node with four RTX PRO 6000 (96 GB) GPUs, no container:

```bash
source .venv/bin/activate
bash scripts/download_data.sh          # GSM8K-Aug training stream, once
torchrun --nproc_per_node 4 -m svp.grpo.train --config configs/grpo_coconut_gpt2.json
```

`configs/grpo_coconut_gpt2.json` is the recipe of the released checkpoint: COCONUT
GPT-2, `W_V` only, G = 32 rollouts per prompt, B = 8 prompts per update, μ = 2 inner
epochs, clip 0.2, β_KL 0.02, lr 3e-5 constant, seed 1, SVP α = 0.6, 10 passes over
GSM8K-Aug. Every field can be overridden on the command line, e.g. `--epochs 3
--lr 6e-5 --run_name my-run`.

Data parallelism leaves the update exact. Each iteration draws one *global* batch of
`prompts_per_iter × overprovision` prompts, shards it across ranks, and sum-reduces the
`W_V` gradient, so a four-rank run takes the same optimizer step as a single-process run
and only the wall clock changes: about 15 GPU-hours per pass on one GPU, so roughly four
hours per pass on four. The in-run evaluations are sharded the same way, so no GPU has
to be reserved for them. Per-rank memory is dominated by `rollout_batch` (16) × `G`;
lower `--rollout_batch` if a rank runs out of memory, and single-GPU training is just
`python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json`.

The run logs GSM8K pass@1 / pass@16 every `k_eval` iterations and all six benchmarks at
epoch boundaries to `results/grpo/<run_name>/log.jsonl` (and to wandb with
`--wandb_project`), and writes `ckpt_epoch{n}.pt` at every epoch. Continue an
interrupted run with `--resume results/grpo/<run_name>/ckpt_last.pt`. Pick the epoch
with the best `eval/pass@1` (epoch 9 for the released checkpoint) and export it into a
plain HuggingFace directory:

```bash
python scripts/export_checkpoint.py --ckpt results/grpo/<run_name>/ckpt_epoch9.pt --out checkpoints/my-svp-v
python -m svp.evaluate --model coconut --checkpoint checkpoints/my-svp-v --n_samples 1
```

## Citation

Anonymous submission under review.

MIT license.
