#!/usr/bin/env bash
# train [flags]  -> SVP-V-GRPO with configs/grpo_coconut_gpt2.json (NPROC>1: torchrun)
# eval  [flags]  -> python -m svp.evaluate
# anything else  -> executed as given
set -euo pipefail
cd /workspace/svp
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" > /dev/null || echo "wandb login failed; logging to log.jsonl only"
fi
case "${1:-}" in
    train)
        shift
        NPROC=${NPROC:-1}
        if [ "$NPROC" -gt 1 ]; then
            exec torchrun --nproc_per_node "$NPROC" -m svp.grpo.train --config configs/grpo_coconut_gpt2.json "$@"
        fi
        exec python -m svp.grpo.train --config configs/grpo_coconut_gpt2.json "$@" ;;
    eval)
        shift
        exec python -m svp.evaluate "$@" ;;
    "")
        echo "usage: train [flags] | eval [flags] | <command>"; exit 1 ;;
    *)
        exec "$@" ;;
esac
