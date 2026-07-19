"""Evalúa NDX-Coder por COMPILACIÓN: genera decks para los prompts de validación,
extrae el código (tras </think>) y mide el % que compila+resuelve.

Es la métrica dura del proyecto (mejor que perplejidad).

Uso (necesita el modelo entrenado + Node/WASM del compilador):
    python -m src.eval_coder --config configs/ndx_coder.yaml \
        --model models/ndx-coder-small --n 200
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compiler import validate
from .config import load_config


def pick_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_deck(text: str) -> str:
    """Quita el scratchpad: devuelve lo que va después de </think> (o todo)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval por compilación de NDX-Coder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Directorio del modelo entrenado")
    parser.add_argument("--n", type=int, default=200, help="Prompts a evaluar")
    parser.add_argument("--data", default=None, help="JSONL a evaluar (def: el val del config)")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--dump", default=None, help="Escribe generaciones a un .jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    prompts, refs = [], []
    data_path = args.data or cfg["data"]["val"]
    with open(data_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.n:
                break
            msgs = json.loads(line)["messages"]
            prompts.append(next(m["content"] for m in msgs if m["role"] == "user"))
            refs.append(next(m["content"] for m in msgs if m["role"] == "assistant"))

    ok = 0
    dump_fh = open(args.dump, "w", encoding="utf-8") if args.dump else None
    for prompt, ref in zip(prompts, refs):
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False
        )
        deck = extract_deck(completion)
        v = validate(deck)
        ok += int(v.ok)
        if dump_fh:
            dump_fh.write(json.dumps(
                {"prompt": prompt, "deck": deck, "ok": v.ok, "error": v.error},
                ensure_ascii=False,
            ) + "\n")

    if dump_fh:
        dump_fh.close()
    n = len(prompts)
    print(json.dumps({
        "model": args.model,
        "n": n,
        "compile_rate": round(ok / n, 4) if n else 0.0,
        "ok": ok,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
