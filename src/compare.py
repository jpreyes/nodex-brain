"""Compara dos modelos+adapters generando respuestas lado a lado.

Escribe un Markdown con: prompt, respuesta de referencia, y la salida de
cada modelo, para revisión humana (la forma más fiable de decidir "cuál es
mejor" en tareas de ingeniería).

Uso:
    python -m src.compare \
        --config-a configs/structural_mistral.yaml --adapter-a adapters/structural-mistral --name-a Mistral \
        --config-b configs/structural_gemma.yaml   --adapter-b adapters/structural-gemma   --name-b Gemma \
        --prompts-from datasets/structural-sft-1000/val.jsonl --n 10 --out compare_structural.md
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import load_config


def load_prompts(path: str, n: int) -> tuple[list[str], list[str]]:
    """Extrae (prompt_usuario, respuesta_referencia) de un JSONL."""
    prompts: list[str] = []
    refs: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            ex = json.loads(line)
            if "messages" in ex:
                msgs = ex["messages"]
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                ref = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            else:
                user = ex.get("instruction", "")
                if ex.get("input"):
                    user = f"{user}\n\n{ex['input']}"
                ref = ex.get("output", "")
            prompts.append(user)
            refs.append(ref)
    return prompts, refs


def generate_all(config_path: str, adapter: str | None, prompts: list[str],
                 max_new_tokens: int) -> list[str]:
    """Carga un modelo (4-bit) + adapter y genera para todos los prompts."""
    cfg = load_config(config_path)
    q = cfg["quantization"]
    base = cfg["model"]["name_or_path"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outputs: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(gen[0][inputs.shape[-1]:], skip_special_tokens=True)
        outputs.append(text.strip())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparación A/B de Nodex Brain")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--adapter-a", default=None)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--adapter-b", default=None)
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--prompts-from", required=True, help="JSONL con los casos")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default="compare.md")
    args = parser.parse_args()

    prompts, refs = load_prompts(args.prompts_from, args.n)
    # Secuencial para no cargar ambos modelos a la vez en VRAM.
    outs_a = generate_all(args.config_a, args.adapter_a, prompts, args.max_new_tokens)
    outs_b = generate_all(args.config_b, args.adapter_b, prompts, args.max_new_tokens)

    lines = [f"# Comparación: {args.name_a} vs {args.name_b}\n"]
    for i, (prompt, ref, a, b) in enumerate(zip(prompts, refs, outs_a, outs_b), 1):
        lines += [
            f"## Caso {i}\n",
            f"**Prompt:**\n\n{prompt}\n",
            f"**Referencia:**\n\n{ref}\n",
            f"**{args.name_a}:**\n\n{a}\n",
            f"**{args.name_b}:**\n\n{b}\n",
            "---\n",
        ]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Escrito {args.out} con {len(prompts)} casos.")


if __name__ == "__main__":
    main()
