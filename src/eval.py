"""Evalúa loss y perplejidad en el split de validación.

Uso:
    python -m src.eval --config configs/structural_mistral.yaml \
        --adapter adapters/structural-mistral --limit 200

OJO: la perplejidad depende del tokenizer, así que NO es directamente
comparable entre Mistral y Gemma (tokenizers distintos). Sirve para seguir
el progreso dentro de un mismo modelo. Para decidir "cuál es mejor" usa
`src.compare` (generaciones lado a lado) y/o una métrica a nivel de tarea.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from peft import PeftModel

from .config import load_config
from .data import build_formatter, load_splits
from .train import build_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de Nodex Brain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="Adapter a evaluar")
    parser.add_argument("--limit", type=int, default=None, help="Máx. ejemplos de val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = build_model_and_tokenizer(cfg)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    dataset = load_splits(cfg)["validation"]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    formatter = build_formatter(tokenizer, cfg)
    max_len = cfg["data"]["max_seq_length"]

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for example in dataset:
            text = formatter(example)
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_len
            ).to(model.device)
            out = model(**enc, labels=enc["input_ids"])
            n_tokens = enc["input_ids"].numel()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        "config": args.config,
        "adapter": args.adapter,
        "n_examples": len(dataset),
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(mean_loss), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
