"""Write a GRPO checkpoint's trained weights into the base model and save a
standalone HF directory that ``svp.evaluate --checkpoint <dir>`` (or plain
``from_pretrained``) can load.

    python scripts/export_checkpoint.py --ckpt results/grpo/<run>/ckpt_epoch9.pt \\
        --out checkpoints/svp-v-coconut-gpt2

Also reads the legacy ``{"v_cols": [...]}`` payloads of the reference runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svp.models import load_adapter  # noqa: E402
from svp.perturb import SVPController  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="coconut")
    p.add_argument("--checkpoint", default=None, help="base checkpoint override")
    p.add_argument("--target", default=None, help="target of a legacy checkpoint (default: attn_v)")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    weights = ck["weights"] if "weights" in ck else ck["v_cols"]
    target = args.target or ck.get("target", "attn_v")
    adapter = load_adapter(args.model, args.checkpoint, device="cpu")
    controller = SVPController(adapter.model, adapter.arch, target, alpha=0.0)
    controller.import_weights(weights)
    controller.remove()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter.model.save_pretrained(out)
    adapter.tokenizer.save_pretrained(out)
    print(f"exported iter {ck.get('iter')} ({target}) -> {out}")


if __name__ == "__main__":
    main()
