"""Prueba interactiva del NDX-Coder: prompt en español → deck NDX.

Uso:
    python -m src.generate --prompt "Modela una viga de acero de 6 m simplemente apoyada con 20 kN/m."
    python -m src.generate                 # modo interactivo (escribe prompts)
    python -m src.generate --compile       # además compila cada deck generado
    python -m src.generate --raw           # muestra también el scratchpad <think>
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from .config import load_config
from .eval_coder import extract_deck, pick_device
from .train_scratch import build_tokenizer


def generate_deck(model, tokenizer, device, prompt: str, max_new_tokens: int) -> str:
    text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    out = tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
    return out.replace(tokenizer.eos_token, "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia NDX-Coder")
    parser.add_argument("--config", default="configs/ndx_coder.yaml")
    parser.add_argument("--model", default="models/ndx-coder-small")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--raw", action="store_true", help="mostrar el <think> completo")
    parser.add_argument("--compile", action="store_true", help="compilar el deck generado")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    tokenizer = build_tokenizer(cfg["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"NDX-Coder en {device}. Listo.\n")

    validate = None
    if args.compile:
        from .compiler import validate as validate

    def run(prompt: str) -> None:
        full = generate_deck(model, tokenizer, device, prompt, args.max_new_tokens)
        deck = full if args.raw else extract_deck(full)
        print("\n" + "=" * 60)
        print(deck)
        print("=" * 60)
        if validate:
            v = validate(extract_deck(full))
            print("compila:" , "✅ OK" if v.ok else f"❌ {v.error}")

    if args.prompt:
        run(args.prompt)
    else:
        print("Modo interactivo — escribe un prompt (línea vacía o Ctrl+C para salir).")
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nchao")
                break
            if not prompt:
                break
            run(prompt)


if __name__ == "__main__":
    main()
