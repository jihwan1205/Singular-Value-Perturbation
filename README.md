# Singular Value Perturbation

Parallel test-time scaling for latent reasoning models (Coconut, CODI), and SVP-V-GRPO.

SVP samples reasoning trajectories by perturbing coefficients in the fixed SVD basis
of a weight matrix. For `W = U diag(s) Vᵀ`, each trajectory independently draws
`g ~ N(0, I)` for each layer and replaces the coefficients with `s ⊙ (1 + αg)`.
These coefficients can become negative, so they are not necessarily singular values
of the perturbed matrix. One sampled set of layer-wise perturbations is held fixed
through prompt prefill and all latent steps; answer tokens are decoded using the clean
projections. The default target is the attention Value projection (`attn_v`).
SVP-V-GRPO uses this distribution for rollouts and trains only the Value projections.

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
≈ 57 / 61 / 71.

### Paper checkpoint and results

The paper's SVP-V-GRPO checkpoint is mirrored anonymously for review
([browse](https://anonymous-hf.com/a/uoc7l9fxgpdc/)). It is a plain HuggingFace model
directory — SVP is a training-time exploration mechanism, and deployment is ordinary
greedy decoding. This is the epoch-9 checkpoint from the `B = 32`, `lr = 6e-5`,
seed-0 run, not the checkpoint produced by the example configuration below.

```bash
curl -L --retry 5 --retry-all-errors https://anonymous-hf.com/api/a/uoc7l9fxgpdc/download/ \
  -o svp-v-coconut-gpt2.zip
unzip svp-v-coconut-gpt2.zip -d checkpoints/svp-v-coconut-gpt2   # ~480 MB
python -m svp.evaluate --model coconut --checkpoint checkpoints/svp-v-coconut-gpt2 --n_samples 1
```

The following clean greedy accuracies (%) are the paper's Figure 3 results for
GPT-2-based continuous latent reasoning models, with a 64-token answer budget.
The four matched GRPO runs start from the same COCONUT checkpoint and use the same
training recipe: `G = 32` rollouts per prompt, up to `B = 32` retained groups from
64 candidate prompts per iteration, learning rate `6e-5`, seed `0`, and the epoch-9
checkpoint of a 15-epoch run. Only rollout exploration differs among those runs.

| Model (GPT-2 124M) | GSM8K | GSM-Hard | MultiArith | SVAMP | ASDiv-A | GSM-Plus |
|---|---|---|---|---|---|---|
| COCONUT (base) | 34.1 | 7.7 | 80.9 | 35.6 | 60.2 | 17.4 |
| SLPO | 34.9 | 7.6 | 82.8 | 34.3 | 58.7 | 18.2 |
| SIM-CoT | 44.7 | 9.3 | 90.5 | 40.6 | 67.2 | 21.5 |
| CoDi | 42.5 | 9.3 | 91.9 | 40.0 | 65.4 | 23.1 |
| Temp-GRPO (answer-token sampling, `T = 1`) | 38.4 | 8.7 | 85.3 | 37.1 | 61.5 | 19.8 |
| Gaussian-V-GRPO (entry-wise noise on `W_V`) | 40.6 | 9.0 | 88.6 | 39.1 | 62.7 | 21.9 |
| Dropout-GRPO (native dropout, `p = 0.2`) | 45.2 | 9.3 | 90.9 | **44.6** | 71.1 | 24.8 |
| **SVP-V-GRPO (paper checkpoint; `α = 0.6`)** | **50.3** | **11.3** | **93.4** | 43.6 | **71.2** | **27.6** |

The trainer in this repository implements the SVP arm. The anonymous mirror contains
the SVP-V checkpoint; the other rows report the paper's evaluations and are not
additional files in that mirror.

## SVP-V-GRPO training example

The included `configs/grpo_coconut_gpt2.json` is a separate `B = 8`, `lr = 3e-5`,
seed-1, 10-epoch example. Its settings do not produce the paper checkpoint above.
To use the paper's `B = 32` hyperparameters with this trainer, override those four
fields as shown below; `G = 32`, `overprovision = 2`, `μ = 2`, `β_KL = 0.02`, and
`α = 0.6` are already set in the configuration.

One node with four RTX PRO 6000 (96 GB) GPUs, no container:

```bash
source .venv/bin/activate
bash scripts/download_data.sh          # GSM8K-Aug training stream, once
# Separate B=8 example
torchrun --nproc_per_node 4 -m svp.grpo.train --config configs/grpo_coconut_gpt2.json
# Paper B=32 settings
torchrun --nproc_per_node 4 -m svp.grpo.train --config configs/grpo_coconut_gpt2.json \
  --prompts_per_iter 32 --lr 6e-5 --seed 0 --epochs 15 --run_name svp-v-paper
```

The example configuration uses COCONUT GPT-2, `W_V` only, `G = 32` rollouts per
prompt, `B = 8` retained groups per update, `μ = 2` optimization passes, clipping
at `0.2`, k3 KL with `β = 0.02`, constant `lr = 3e-5`, seed `1`, SVP `α = 0.6`, and
10 passes over GSM8K-Aug. Every field can be overridden on the command line.

Data parallelism leaves the update exact. Each iteration draws one *global* batch of
`prompts_per_iter × overprovision` prompts, shards it across ranks, and sum-reduces the
`W_V` gradient, so a four-rank run takes the same optimizer step as a single-process run
and only the wall clock changes: for the `B = 8` example, about 15 GPU-hours per pass
on one GPU, or roughly four hours per pass on four. The `B = 32` run has a larger
rollout workload. The in-run evaluations are sharded the same way, so no GPU has
to be reserved for them. Per-rank memory is dominated by `rollout_batch` (16) × `G`;
lower `--rollout_batch` if a rank runs out of memory, and single-GPU training is just
`python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json`.

The run logs GSM8K pass@1 / pass@16 every `k_eval` iterations and all six benchmarks at
epoch boundaries to `results/grpo/<run_name>/log.jsonl` (and to wandb with
`--wandb_project`), and writes `ckpt_epoch{n}.pt` at every epoch. Continue an
interrupted run with `--resume results/grpo/<run_name>/ckpt_last.pt`. Pick the epoch
with the best `eval/pass@1` for an example run and export it into a plain HuggingFace
directory. The paper reports epoch 9 of its 15-epoch run; for that run:

```bash
python scripts/export_checkpoint.py --ckpt results/grpo/svp-v-paper/ckpt_epoch9.pt --out checkpoints/svp-v-paper
python -m svp.evaluate --model coconut --checkpoint checkpoints/svp-v-paper --n_samples 1
```

## Citation

Anonymous submission under review.

MIT license.
