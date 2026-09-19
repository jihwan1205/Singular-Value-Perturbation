# Self-contained image: code + environment + benchmarks + the GSM8K-Aug
# training stream + the Coconut GPT-2 base checkpoint. Nothing is fetched at
# run time (set WANDB_API_KEY to log to wandb).
#
#   docker build -t svp .
#   docker run --gpus '"device=0"' --shm-size 16g svp train --run_name my-run
#   docker run --gpus all -e NPROC=4 svp train --run_name my-run            # torchrun
#   docker run --gpus '"device=0"' svp eval --model coconut --alpha 0.6 --n_samples 16
#   mount -v /host/results:/workspace/svp/results to keep checkpoints and metrics
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/workspace/svp/.hf
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/workspace/svp/.venv/bin:${PATH}"

WORKDIR /workspace/svp
COPY requirements.txt .
RUN uv venv --python=3.12 && uv pip install -r requirements.txt && uv pip install wandb

COPY svp ./svp
COPY scripts ./scripts
COPY configs ./configs
COPY data ./data
COPY tests ./tests
COPY README.md pyproject.toml LICENSE ./
RUN bash scripts/download_data.sh \
    && hf download ModalityDance/latent-tts-coconut > /dev/null

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh scripts/download_data.sh
ENTRYPOINT ["/entrypoint.sh"]
