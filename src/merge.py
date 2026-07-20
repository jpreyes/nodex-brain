"""Fusiona un adapter LoRA sobre el modelo base y exporta el modelo completo.

Uso:
    python -m src.merge --config configs/qlora_structural.yaml \
        --adapter adapters/structural --out models/structural-merged
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge de adapter QLoRA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True, help="Directorio del adapter")
    parser.add_argument("--out", required=True, help="Directorio de salida del merge")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_name = cfg["model"]["name_or_path"]
    trust = cfg["model"].get("trust_remote_code", False)

    base = AutoModelForCausalLM.from_pretrained(
        base_name, device_map="auto", dtype=torch.bfloat16, trust_remote_code=trust,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(base_name, trust_remote_code=trust).save_pretrained(args.out)
    print(f"Modelo fusionado guardado en {args.out}")


if __name__ == "__main__":
    main()
