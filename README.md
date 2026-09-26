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
The paper's GPT-2 checkpoint is
[`jihwan1205/svp-v-coconut-gpt2`](https://huggingface.co/jihwan1205/svp-v-coconut-gpt2).

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
python -m svp.evaluate --model coconut --checkpoint jihwan1205/svp-v-coconut-gpt2 --n_samples 1
```

Models: `coconut`, `codi-gpt2`, `codi-llama1b`. Targets: `attn_q`, `attn_k`, `attn_v`,
`attn_o`, `mlp_up`, `mlp_down`. Results (pass@k for k ≤ n, coverage, majority vote) go to
`results/<run>/`. Expected greedy GSM8K: 34.12 / 42.46 / 55.57; SVP pass@16 (α 0.6 / 0.5 / 0.5):
≈ 57 / 61 / 71.

### Released checkpoints

The following clean greedy accuracies (%) are the paper's Figure 3 results for
GPT-2-based continuous latent reasoning models, with a 64-token answer budget.
The matched GRPO runs start from the same COCONUT checkpoint and use the same
training recipe: `G = 32` rollouts per prompt, up to `B = 32` retained groups per
iteration, learning rate `6e-5`, seed `0`, and the epoch-9 checkpoint of a
15-epoch run. Only the rollout exploration differs among those four runs.

| Model (GPT-2 124M) | GSM8K | GSM-Hard | MultiArith | SVAMP | ASDiv-A | GSM-Plus |
|---|---|---|---|---|---|---|
| COCONUT (base) | 34.1 | 7.7 | 80.9 | 35.6 | 60.2 | 17.4 |
| SLPO | 34.9 | 7.6 | 82.8 | 34.3 | 58.7 | 18.2 |
| SIM-CoT | 44.7 | 9.3 | 90.5 | 40.6 | 67.2 | 21.5 |
| CoDi | 42.5 | 9.3 | 91.9 | 40.0 | 65.4 | 23.1 |
| Temp-GRPO (answer-token sampling, `T = 1`) | 38.4 | 8.7 | 85.3 | 37.1 | 61.5 | 19.8 |
| Gaussian-V-GRPO (entry-wise noise on `W_V`) | 40.6 | 9.0 | 88.6 | 39.1 | 62.7 | 21.9 |
| Dropout-GRPO (native dropout, `p = 0.2`) | 45.2 | 9.3 | 90.9 | **44.6** | 71.1 | 24.8 |
| **SVP-V-GRPO ([`svp-v-coconut-gpt2`](https://huggingface.co/jihwan1205/svp-v-coconut-gpt2))** | **50.3** | **11.3** | **93.4** | 43.6 | **71.2** | **27.6** |

The linked SVP-V-GRPO checkpoint is the `B = 32`, `lr = 6e-5`, seed-0 paper run.
The included `configs/grpo_coconut_gpt2.json` instead specifies a separate
`B = 8`, `lr = 3e-5`, seed-1, 10-epoch example. That configuration does not
produce the checkpoint or matched comparison reported in Figure 3.

[`temp-v-coconut-gpt2`](https://huggingface.co/jihwan1205/temp-v-coconut-gpt2) is the
token-sampling checkpoint of that ablation (the epoch-9 row above).
[`dense-v-coconut-gpt2`](https://huggingface.co/jihwan1205/dense-v-coconut-gpt2) is the
Gaussian one, taken at its best-GSM8K epoch rather than epoch 9, so it scores
41.9 / 9.0 / 39.3 / 62.9 / 87.1 / 21.7. The trainer in this repo implements the SVP
arm only; the controls are retrained with the ablation images in [Docker](#docker).
Any released checkpoint is evaluated with the same command:

```bash
for m in svp-v dense-v temp-v; do
  python -m svp.evaluate --model coconut --checkpoint jihwan1205/$m-coconut-gpt2 --n_samples 1
done
```

## SVP-V-GRPO training example

The commands below use the separate `B = 8` configuration in
`configs/grpo_coconut_gpt2.json`. To reproduce the paper's matched `B = 32`
experiment, use the `coconut-gpt2` image in [Prebuilt ablation images](#prebuilt-ablation-images).

```bash
python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json            # 1 GPU
torchrun --nproc_per_node 4 -m svp.grpo.train --config configs/grpo_coconut_gpt2.json
python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json --resume results/grpo/svp-v-coconut-gpt2/ckpt_last.pt
```

The example config trains for 10 passes over GSM8K-Aug (about 15 GPU-hours
per pass on one 96 GB GPU). Every field can be overridden with `--<field> <value>`. The run
logs GSM8K pass@1 / pass@16 every `k_eval` iterations and all six benchmarks at epoch
boundaries to `log.jsonl` (and wandb with `--wandb_project`). Pick the `ckpt_epoch{n}.pt`
with the best `eval/pass@1` (epoch 9 for this example run) and export it:

```bash
python scripts/export_checkpoint.py --ckpt results/grpo/svp-v-coconut-gpt2/ckpt_epoch9.pt --out checkpoints/svp-v-coconut-gpt2
python -m svp.evaluate --model coconut --checkpoint checkpoints/svp-v-coconut-gpt2 --n_samples 1
```

## Docker

```bash
docker build -t svp .
docker run --gpus '"device=0"' --shm-size 16g -v $PWD/results:/workspace/svp/results svp train --run_name my-run
docker run --gpus all -e NPROC=4 --shm-size 16g -v $PWD/results:/workspace/svp/results svp train --run_name my-run
docker run --gpus '"device=0"' -v $PWD/results:/workspace/svp/results svp eval --model coconut --alpha 0.6 --n_samples 16
```

`train` forwards its flags to `svp.grpo.train`, `eval` to `svp.evaluate`; any other command runs as given.

### Prebuilt ablation images

The four GRPO arms of the paper, each self-contained (code, venv, benchmarks, training
stream, start checkpoint). They share the paper's recipe — `W_V` only, `G = 32`,
up to 32 retained groups from 64 candidate prompts per iteration, `lr = 6e-5`,
seed 0, 15 epochs — and differ only in the exploration:

| Image `jihwanshin/svp-v-grpo:` | Rollout exploration | Run name |
|---|---|---|
| `coconut-gpt2` | SVP on `W_V`, α = 0.6 (ours) | `svp15` |
| `coconut-gpt2-temp-ablation` | answer-token sampling, T = 1 | `temp15` |
| `coconut-gpt2-dense-ablation` | entry-wise Gaussian on `W_V`, σ = 0.06 | `dense15` |
| `dropout-grpo` | native dropout, p = 0.2 | `dropout15` |

```bash
docker run --gpus all --shm-size 32g -v $PWD/results:/workspace/SVP/results \
  jihwanshin/svp-v-grpo:coconut-gpt2-dense-ablation dense15 0,1,2,3
```

Arguments are `<run_name> <gpu_list> [train flags]`, one DDP rank per listed GPU
(numbered inside the container); flags after the GPU list override the arm, e.g.
`--epochs 3`. Needs driver 570 or newer and about 41 GB per GPU, plus up to 15 GB on the
last one for the in-run evaluations (`--eval_gpu N` moves them elsewhere). Checkpoints,
`log.jsonl` and the per-epoch six-benchmark results land in
`results/grpo/runs/<run_name>_g32_lr6e-5_mu2/`; `resume <full_run_name> <gpu_list>`
continues an interrupted run, and `scripts/export_grpo_checkpoint.py` turns a
`ckpt_iter*.pt` into a plain HuggingFace model directory.


## Citation

```bibtex
@article{svp2026,
  title  = {Singular Value Perturbation: Unlocking RL Performance via Diverse Latent Reasoning},
  author = {},
  year   = {2026}
}
```

MIT license.
