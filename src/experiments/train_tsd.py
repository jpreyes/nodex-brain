"""EXPERIMENTO 2A — ablación del bias de atención ULTRAMÉTRICO (TSD).

Entrena el MISMO modelo con y sin bias. `--tsd` cambia solo el bias; datos, seed e
hiperparámetros quedan idénticos → ablación limpia. Juzgar por ADECUACIÓN vs baseline
(TSD-CONTRACT: conservar el mecanismo solo si mueve la métrica).

    E=src.experiments.train_tsd; A="--ablation --no-repair"
    python -m $E --config configs/coder_nano.yaml      $A                  # nano  base
    python -m $E --config configs/coder_nano.yaml      $A --tsd --tree ast # nano  +TSD
    python -m $E --config configs/coder_qwen3_06b.yaml $A                  # qwen3 base
    python -m $E --config configs/coder_qwen3_06b.yaml $A --tsd --tree ast # qwen3 +TSD

Este archivo NO es la ruta de producción: reutiliza `src/train_sft.py` (modelo, datos,
preprocesado, TrainingArguments) y solo añade tres cosas:
  1. los paths del árbol por token, vía el hook `extra` del preprocesado compartido,
  2. `TSDCollator`, que arma la máscara 4D = causal + bias ultramétrico,
  3. `attn_implementation="eager"`, que esa máscara aditiva 4D requiere.

Ver EXPERIMENTS.md para el diseño, los descubrimientos y el estado de las decisiones.
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..train_sft import (add_common_args, build_datasets, build_training_args,
                         load_model_and_tokenizer, CoT_END)
from ..tsd.collator import TSDCollator
from ..tsd.ultrametric import get_tree, token_paths

from transformers import Trainer


def make_tsd_extra(tree):
    """Hook del preprocesado compartido: adjunta paths/in_deck del árbol por token.

    El target es "<think>razonamiento</think>\\n<deck>" (413/500 ejemplos del dataset
    combined-cot lo traen). El bias TSD es una hipótesis sobre la ESTRUCTURA DEL AST:
    aplicarlo al scratchpad sería aplicarlo a prosa. Peor con --tree ast, que tomaría la
    primera palabra de cada paso numerado ("1.") como si fuera un statement kind.
    → el árbol se construye SOLO sobre el deck; el CoT queda como contexto sin bias.
      (El CoT SÍ se entrena con loss: eso lo decide `labels` en train_sft, no esto.)
    """
    tree_fn, _K = get_tree(tree)

    def _extra(offs, p_len, target):
        d_off = target.find(CoT_END) + len(CoT_END) if CoT_END in target else 0
        deck = target[d_off:]
        char_paths, K = tree_fn(deck)
        d0 = p_len + d_off
        paths, in_deck = token_paths(offs, (d0, d0 + len(deck)), char_paths, K)
        return {"paths": paths.tolist(), "in_deck": in_deck.tolist()}

    return _extra


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(
        description="Experimento 2A: ablación TSD (con/sin bias ultramétrico)"))
    ap.add_argument("--tsd", action="store_true", help="activa el bias de atención TSD")
    ap.add_argument("--ablation", action="store_true",
                    help="sufija el output con -base/-tsd-<árbol> (para el 2x2)")
    ap.add_argument("--lam", type=float, default=1.0, help="fuerza del bias TSD (λ)")
    ap.add_argument("--kernel", default="linear", choices=["linear", "padic"])
    ap.add_argument("--tree", default="fallback", choices=["fallback", "ast"],
                    help="fallback=bloque/línea (K=2, histórico) · ast=SEAM 2 real (K=4)")
    ap.add_argument("--no-norm", action="store_true",
                    help="NO normalizar D por K (reproduce las corridas históricas; "
                         "rompe la comparabilidad de λ entre árboles de distinta profundidad)")
    ap.add_argument("--no-repair", action="store_true",
                    help="ablación PURA: no mezcla repair (recomendado para el 2x2)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_len = args.seq or cfg["data"]["max_seq_length"]
    tag = "tsd" if args.tsd else "base"

    # eager: la máscara aditiva 4D del collator lo exige (SDPA la descartaría sin avisar).
    tok, model = load_model_and_tokenizer(cfg, tag=tag, attn="eager")
    ds = build_datasets(cfg, args, tok, max_len, extra=make_tsd_extra(args.tree))

    # El árbol va en el nombre: -tsd-fallback y -tsd-ast son experimentos DISTINTOS y no
    # deben pisarse. El baseline es común a ambos (bias 0 no depende del árbol).
    suffix = ("-tsd-" + args.tree) if args.tsd else "-base"
    out = cfg["train"]["output_dir"] + (suffix if args.ablation else "")

    _, K_tree = get_tree(args.tree)
    collator = TSDCollator(pad_token_id=tok.pad_token_id, use_tsd=args.tsd,
                           lam=args.lam, kernel=args.kernel, K=K_tree,
                           norm=not args.no_norm)
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, args, out, tag, keep_cols=True),  # paths -> collator
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=collator, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"[{tag}] guardado en {out}")


if __name__ == "__main__":
    main()
