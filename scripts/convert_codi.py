"""Convert the official CODI releases into plain HF checkpoints.

    python scripts/convert_codi.py --model gpt2      # -> checkpoints/codi-gpt2
    python scripts/convert_codi.py --model llama1b   # -> checkpoints/codi-llama1b

The released ``pytorch_model.bin`` is the authors' wrapper state dict: a
PEFT backbone with unmerged LoRA (r=128, alpha=32) plus the projector MLP.
This merges the LoRA (``W += 0.25 * B @ A``), builds the base model with the
three extra tokens (pad/bot/eot), and saves the model, the base tokenizer and
``projector.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LORA_SCALE = 32 / 128

SPECS = {
    "gpt2": {"repo": "zen-E/CODI-gpt2", "out": "checkpoints/codi-gpt2",
             "base": "openai-community/gpt2", "vocab": 50260, "transpose_delta": True},
    "llama1b": {"repo": "zen-E/CODI-llama3.2-1b-Instruct", "out": "checkpoints/codi-llama1b",
                "base": "unsloth/Llama-3.2-1B-Instruct", "vocab": 128259, "transpose_delta": False},
}
_PREFIX = "codi.base_model.model."


def merge_lora(raw: dict[str, torch.Tensor], transpose_delta: bool) -> tuple[dict, dict]:
    """GPT-2 targets are Conv1D ``(in, out)`` so the ``(out, in)`` delta is transposed."""
    model_sd, prj_sd, lora_a, lora_b = {}, {}, {}, {}
    for key, tensor in raw.items():
        if key.startswith("prj."):
            prj_sd[key[len("prj."):]] = tensor.float()
        elif key.startswith("dynamic_cls."):
            continue
        elif ".lora_A.default.weight" in key:
            lora_a[key.replace(".lora_A.default.weight", "")] = tensor.float()
        elif ".lora_B.default.weight" in key:
            lora_b[key.replace(".lora_B.default.weight", "")] = tensor.float()
        elif key.startswith(_PREFIX):
            model_sd[key[len(_PREFIX):].replace(".base_layer", "")] = tensor.float()
    assert lora_a.keys() == lora_b.keys()
    for mod_key in lora_a:
        plain = mod_key[len(_PREFIX):].replace(".base_layer", "") + ".weight"
        delta = (lora_b[mod_key] @ lora_a[mod_key]) * LORA_SCALE
        if transpose_delta:
            delta = delta.t()
        if model_sd[plain].shape != delta.shape:
            raise ValueError(f"shape mismatch at {mod_key}: {model_sd[plain].shape} vs {delta.shape}")
        model_sd[plain] = model_sd[plain] + delta
    return model_sd, prj_sd


def convert(name: str) -> None:
    spec = SPECS[name]
    raw_path = hf_hub_download(spec["repo"], "pytorch_model.bin")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    model_sd, prj_sd = merge_lora(raw, spec["transpose_delta"])
    config = AutoConfig.from_pretrained(spec["base"])
    config.vocab_size = spec["vocab"]
    model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    missing = [k for k in missing if "lm_head" not in k]
    if missing or unexpected:
        raise RuntimeError(f"load mismatch: missing={missing}, unexpected={unexpected}")
    model.tie_weights()
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(spec["base"]).save_pretrained(out)
    torch.save(prj_sd, out / "projector.pt")
    print(f"saved {out} ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*SPECS, "all"], default="all")
    args = parser.parse_args()
    for name in SPECS if args.model == "all" else [args.model]:
        convert(name)
