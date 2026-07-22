"""¿Genera lo MISMO la ruta con KV-cache que la ruta de referencia O(n²)?

    python -m src.experiments.verify_kv_cache --model models/exp-nano-215m-tsd-ast-s42

Por qué existe: recomputar la máscara [L,L] en cada paso costaba ~63 h para evaluar los 6
brazos TSD del 2x2. Con KV-cache basta la FILA del token nuevo, y baja a ~4 h. Pero el
cache introduce un supuesto que la ruta lenta no hace:

    el bias entre tokens PASADOS queda congelado en el cache.

En la ruta O(n²) el árbol se reconstruye entero en cada paso, así que si un token nuevo
cambiara el path de uno anterior, ese cambio se aplicaría hacia atrás. Con cache no: los
K,V del pasado ya se calcularon con el bias de entonces.

En teoría los paths pasados son estables (el kind lo fija la primera palabra de la línea,
que ya está escrita), pero eso hay que MEDIRLO, no asumirlo. Si los textos generados
difieren, el KV-cache no se puede usar y hay que replantear la evaluación.

Este test es obligatorio antes de usar --use-cache en la generación de Test A.
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

from ..tsd.infer import generate_tsd, load_tsd_config

PROMPTS = [
    "modela una viga simplemente apoyada de 6 m con carga distribuida de 20 kN/m",
    "marco plano de 2 vanos y 1 piso, columnas de 3 m, acero S275",
    "cercha de 8 m con 4 paneles, apoyo fijo y móvil",
    "losa cuadrada de 5x5 m empotrada en los cuatro bordes",
    "análisis modal de un edificio de 3 pisos con masas concentradas",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Equivalencia KV-cache vs ruta O(n^2)")
    ap.add_argument("--model", required=True, help="checkpoint de un brazo +TSD")
    ap.add_argument("--n", type=int, default=3, help="cuántos prompts comparar")
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    cfg = load_tsd_config(args.model)
    if not cfg.get("use_tsd"):
        sys.exit(f"{args.model} es un brazo BASE (use_tsd=false): no hay bias que comparar. "
                 f"Usa un checkpoint -tsd-*.")
    print(f"modelo : {args.model}")
    print(f"bias   : {json.dumps(cfg, ensure_ascii=False)}\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    iguales, t_cache, t_ref = 0, 0.0, 0.0
    for i, pedido in enumerate(PROMPTS[: args.n], 1):
        prompt = tok.apply_chat_template([{"role": "user", "content": pedido}],
                                         tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        con = generate_tsd(model, tok, prompt, args.max_new, tsd_cfg=cfg, use_cache=True)
        t_cache += time.perf_counter() - t0

        t0 = time.perf_counter()
        sin = generate_tsd(model, tok, prompt, args.max_new, tsd_cfg=cfg, use_cache=False)
        t_ref += time.perf_counter() - t0

        ok = con == sin
        iguales += ok
        print(f"[{i}] {'IDÉNTICO' if ok else 'DIFIERE'}  ({pedido[:45]}…)")
        if not ok:
            # dónde diverge: el primer carácter distinto dice si es un detalle del final
            # o si la generación se bifurcó temprano
            j = next((k for k in range(min(len(con), len(sin))) if con[k] != sin[k]),
                     min(len(con), len(sin)))
            print(f"     diverge en el carácter {j} de {min(len(con), len(sin))}")
            print(f"     cache: …{con[max(0, j-40):j+40]!r}")
            print(f"     ref  : …{sin[max(0, j-40):j+40]!r}")

    n = args.n
    print(f"\n{'=' * 62}")
    print(f"  idénticos      : {iguales}/{n}")
    print(f"  KV-cache       : {t_cache/n:6.1f} s por generación")
    print(f"  referencia O(n²): {t_ref/n:6.1f} s por generación")
    print(f"  aceleración    : {t_ref/max(t_cache, 1e-9):.1f}x")
    print("=" * 62)
    if iguales == n:
        print("\n  PASA — el KV-cache reproduce la generación de referencia.")
        print("  Se puede usar para evaluar los brazos TSD sobre Test A.")
        sys.exit(0)
    print("\n  FALLA — las rutas divergen. NO usar KV-cache: el bias del pasado")
    print("  congelado en el cache SÍ cambia la generación. Hay que evaluar con la")
    print("  ruta lenta (y replantear el n, porque son ~63 h) o repensar el diseño.")
    sys.exit(1)


if __name__ == "__main__":
    main()
