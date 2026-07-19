"""Inferencia rápida con el modelo base + adapter QLoRA.

Uso:
    python -m src.infer --config configs/qlora_structural.yaml \
        --adapter adapters/structural --prompt "..."
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia Nodex Brain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="Adapter opcional")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_name = cfg["model"]["name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=args.max_new_tokens)

    print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
