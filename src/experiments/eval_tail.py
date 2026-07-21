"""Métrica de COLA para 2A/2B — recall y precisión de statement kind.

La etiqueta es GRATIS: el corpus es sintético y cada ejemplo trae su deck gold. El
multiset de kinds del gold es la etiqueta perfecta, sin curar nada (§12/R-E1).

    recall(k)    = de los ejemplos cuyo GOLD usa k, ¿en cuántos el modelo emitió k?
                   -> "¿lo selecciona cuando corresponde?"  ESTA es la de la cola
    precisión(k) = de los ejemplos donde el modelo emitió k, ¿en cuántos el gold lo tenía?
                   -> alucinación de statements

POTENCIA — leer antes de interpretar (§13.5/E9). Los kinds más raros dan 10-50 ejemplos
en un test al 12%: con n=10 el IC95 del recall es de ±30 puntos. **El número que decide es
el recall AGREGADO de cola** (n~813, error estándar ~1.7 pp). El desglose por kind es
DESCRIPTIVO; este script marca con (!) los kinds cuyo n no permite concluir.

CORTE cabeza/cola: se fija por frecuencia en TRAIN y se PREREGISTRA — elegirlo después de
mirar resultados invalidaría el análisis. Default: <1000 ocurrencias (§12).

Uso:
    # 1) preregistrar el corte (una vez, se guarda en disco)
    python -m src.experiments.eval_tail preregistrar --train <train.jsonl> --cut 1000

    # 2) medir un modelo: jsonl con {"gold": "<deck>", "pred": "<deck>"} por línea
    python -m src.experiments.eval_tail medir --pairs <generados.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

# la consola de Windows es cp1252 y revienta con los símbolos del reporte (→, ·, IC95).
# En la caja (Linux) no hace falta, pero esta herramienta se corre también en local.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ..tsd.ultrametric import _kind_of

CoT_END = "</think>"
PREREG = "eval_tail_prereg.json"


def deck_kinds(text: str) -> set[str]:
    """Conjunto de statement kinds de un deck. Corta el `<think>` si viene incluido."""
    if CoT_END in text:
        text = text[text.find(CoT_END) + len(CoT_END):]
    return {k for k in (_kind_of(l) for l in text.splitlines()) if k and k != "//"}


def target_of(msgs) -> str:
    return msgs[max(i for i, m in enumerate(msgs) if m["role"] == "assistant")]["content"]


def wilson(k: int, n: int) -> tuple[float, float]:
    """IC95 de una proporción (Wilson). Honesto con n chico, a diferencia del normal."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cmd_preregistrar(args) -> None:
    """Fija el corte cabeza/cola por frecuencia en train. Se hace UNA vez, antes de medir."""
    freq = Counter()
    n = 0
    with open(args.train, encoding="utf-8") as fh:
        for line in fh:
            n += 1
            for k in deck_kinds(target_of(json.loads(line)["messages"])):
                freq[k] += 1
    cola = sorted(k for k, c in freq.items() if c < args.cut)
    cabeza = sorted(k for k, c in freq.items() if c >= args.cut)
    prereg = {"corte": args.cut, "criterio": "ejemplos cuyo deck gold usa el kind, en train",
              "n_train": n, "cola": cola, "cabeza": cabeza,
              "freq": dict(freq.most_common())}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(prereg, fh, indent=2, ensure_ascii=False)
    print(f"corte <{args.cut} sobre {n} ejemplos → {args.out}")
    print(f"  cabeza ({len(cabeza)}): {' '.join(cabeza)}")
    print(f"  cola   ({len(cola)}): {' '.join(cola)}")
    print("\nPREREGISTRADO. No volver a tocarlo después de mirar resultados.")


def cmd_medir(args) -> None:
    if not os.path.isfile(args.prereg):
        raise SystemExit(f"falta {args.prereg}: corre primero `preregistrar`. "
                         f"Fijar el corte DESPUÉS de ver resultados invalida el análisis.")
    with open(args.prereg, encoding="utf-8") as fh:
        prereg = json.load(fh)
    cola = set(prereg["cola"])

    hit, gold_n, pred_n = Counter(), Counter(), Counter()
    n = 0
    with open(args.pairs, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            g, p = deck_kinds(ex["gold"]), deck_kinds(ex["pred"])
            n += 1
            for k in g:
                gold_n[k] += 1
                if k in p:
                    hit[k] += 1
            for k in p:
                pred_n[k] += 1

    print(f"pares evaluados: {n}   ·   corte de cola preregistrado: <{prereg['corte']}\n")

    # --- EL NÚMERO QUE DECIDE: agregado de cola --------------------------------
    print("=" * 68)
    print("AGREGADO (esto es lo que decide — §13.5/E9)")
    print("=" * 68)
    for etiqueta, ks in (("COLA", cola), ("cabeza", set(prereg["cabeza"]))):
        h = sum(hit[k] for k in ks)
        g = sum(gold_n[k] for k in ks)
        lo, hi = wilson(h, g)
        marca = "  <<<" if etiqueta == "COLA" else ""
        print(f"  recall {etiqueta:7s} {h/g if g else 0:6.3f}   IC95 [{lo:.3f}, {hi:.3f}]   n={g}{marca}")

    # --- descriptivo por kind ---------------------------------------------------
    print("\n" + "=" * 68)
    print("POR KIND (descriptivo — (!) = n insuficiente para concluir)")
    print("=" * 68)
    print(f"{'kind':14s}{'n_gold':>7s}{'recall':>8s}{'IC95':>16s}{'precisión':>11s}")
    for k in sorted(gold_n, key=lambda x: (x not in cola, -gold_n[x])):
        g, h, pn = gold_n[k], hit[k], pred_n[k]
        lo, hi = wilson(h, g)
        flag = " (!)" if g < args.n_min else ""
        tag = "·cola" if k in cola else ""
        print(f"{k:14s}{g:7d}{h/g if g else 0:8.3f}  [{lo:.2f},{hi:.2f}]"
              f"{h/pn if pn else 0:11.3f}  {tag}{flag}")

    solo_pred = sorted(set(pred_n) - set(gold_n))
    if solo_pred:
        print(f"\nkinds emitidos que NINGÚN gold pedía (alucinación pura): "
              f"{', '.join(f'{k}({pred_n[k]})' for k in solo_pred)}")

    print("\nNOTA: el recall por kind con n chico es ruido — un ejemplo mueve 10 puntos con")
    print("n=10. Reportar SIEMPRE el agregado de cola como resultado, y esta tabla como")
    print("descripción. Y reportar la adecuación global aparte, sin fusionar (§12/R-E1).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Métrica de cola: recall/precisión por kind")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("preregistrar", help="fija el corte cabeza/cola (una sola vez)")
    a.add_argument("--train", required=True, help="train.jsonl del que se cuenta la frecuencia")
    a.add_argument("--cut", type=int, default=1000)
    a.add_argument("--out", default=PREREG)
    a.set_defaults(fn=cmd_preregistrar)

    b = sub.add_parser("medir", help="mide un modelo sobre pares gold/pred")
    b.add_argument("--pairs", required=True, help='jsonl con {"gold": ..., "pred": ...}')
    b.add_argument("--prereg", default=PREREG)
    b.add_argument("--n-min", type=int, default=30, help="n bajo el cual se marca (!)")
    b.set_defaults(fn=cmd_medir)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
