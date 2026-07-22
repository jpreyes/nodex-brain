"""Genera decks sobre Test A con un checkpoint del 2x2 → pares {gold, pred} para eval_tail.

    python -m src.experiments.gen_testa --model models/exp-nano-215m-tsd-ast-s42

MUESTREO ESTRATIFICADO, no uniforme (§13/R-R-E1): entran TODOS los ejemplos cuyo deck gold
usa un kind de cola, más una muestra de cabeza. La potencia del experimento la da la cola
(~567 ejemplos en Test A, IC95 ~±4 pp); submuestrear uniforme recortaría justo donde no
sobra. La cabeza es contraste; 700 da el estratificado completo (~1267 por modelo).

REANUDABLE por ejemplo: si la instancia se cae —pasa— relanzas y sigue desde el último
par escrito. El .jsonl se abre en modo append y se cuenta lo ya hecho.

El bias sale del `tsd_config.json` del checkpoint, nunca de la CLI: es lo que garantiza que
cada brazo se evalúe con el MISMO bias con el que entrenó (§6.5). Los brazos base usan
model.generate() nativo; los +TSD, generate_tsd con KV-cache (verificado en
verify_kv_cache: 3/3 idénticos a la ruta O(n²)).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..tsd.infer import CoT_END, generate_tsd, load_tsd_config
from ..tsd.ultrametric import _kind_of

SPLIT = "../nodex-code/datasets/frozen-split-v2"
# Cola preregistrada: kinds con <1000 ejemplos en train (§12/R-E1). Fijada ANTES de ver
# resultados; moverla después invalidaría el análisis.
COLA = {"accelerogram", "area", "assign", "box", "cable", "combination", "contact",
        "diaphragm", "fiber", "heatBC", "link", "nonlinear", "slab", "soil", "solid",
        "spectrum", "wall"}


def deck_de(ex):
    t = [m for m in ex["messages"] if m["role"] == "assistant"][-1]["content"]
    return t.split(CoT_END)[-1].strip()


def pedido_de(ex):
    return [m for m in ex["messages"] if m["role"] == "user"][0]["content"]


def kinds(deck):
    return {k for k in (_kind_of(l) for l in deck.splitlines()) if k and k != "//"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera sobre Test A para la métrica de cola")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out", default=None, help="por defecto: eval/<nombre-del-modelo>.jsonl")
    ap.add_argument("--cabeza", type=int, default=700, help="cuántos de cabeza (la cola va entera)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="para probar el pipeline")
    args = ap.parse_args()

    name = os.path.basename(args.model.rstrip("/"))
    out_path = args.out or f"eval/{name}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- muestra estratificada, determinista (orden del fichero, sin RNG) --------
    casos = []
    with open(os.path.join(args.split, "test_a.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            d = deck_de(ex)
            casos.append((pedido_de(ex), d, bool(kinds(d) & COLA)))
    cola = [c for c in casos if c[2]]
    cabeza = [c for c in casos if not c[2]][: args.cabeza]
    muestra = cola + cabeza
    if args.limit:
        muestra = muestra[: args.limit]
    print(f"Test A: {len(casos)} ejemplos → muestra {len(muestra)} "
          f"(cola {len(cola)} completa + cabeza {len(cabeza)})")

    # --- reanudación: cuántos pares ya están escritos ----------------------------
    hechos = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            hechos = sum(1 for _ in fh)
    if hechos >= len(muestra):
        print(f"{out_path} ya tiene {hechos} pares — nada que hacer.")
        return
    if hechos:
        print(f"reanudo: {hechos} ya hechos, faltan {len(muestra) - hechos}")

    # --- modelo -------------------------------------------------------------------
    cfg = load_tsd_config(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    # eager solo si hay bias (su máscara 4D lo exige); sin bias, el default es más rápido
    kw = {"attn_implementation": "eager"} if cfg.get("use_tsd") else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **kw)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print(f"modelo: {name}  ·  bias: {json.dumps(cfg, ensure_ascii=False)}")

    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8", newline="\n") as fh:
        for i, (pedido, gold, es_cola) in enumerate(muestra):
            if i < hechos:
                continue
            prompt = tok.apply_chat_template([{"role": "user", "content": pedido}],
                                             tokenize=False, add_generation_prompt=True)
            pred = generate_tsd(model, tok, prompt, args.max_new, tsd_cfg=cfg)
            fh.write(json.dumps({"gold": gold, "pred": pred, "cola": es_cola},
                                ensure_ascii=False) + "\n")
            fh.flush()                       # por si la instancia desaparece a mitad
            n = i + 1 - hechos
            if n % 25 == 0:
                dt = time.perf_counter() - t0
                falta = (len(muestra) - i - 1) * dt / n
                print(f"  {i+1}/{len(muestra)}  {dt/n:.1f}s/gen  quedan ~{falta/60:.0f} min",
                      flush=True)

    print(f"\nescrito {out_path} ({len(muestra)} pares)")
    print(f"siguiente: python -m src.experiments.eval_tail medir --pairs {out_path}")


if __name__ == "__main__":
    main()
