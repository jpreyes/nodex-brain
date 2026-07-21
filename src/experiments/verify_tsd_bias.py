"""Verifica que el bias de atención TSD REALMENTE llega a la atención.

Solo nano-coder. Se corre en la caja, donde están los checkpoints.

    python -m src.experiments.verify_tsd_bias --config configs/coder_nano.yaml

Tres chequeos independientes; los tres deben pasar:

  [1] ESTRUCTURA — el árbol sobre datos reales no es degenerado.
      Falsea: todo el deck en un solo bloque → bias uniforme → no hay señal
      aunque el bias "entre".

  [2] FORWARD — con y sin bias los logits difieren, y una máscara de control
      con -inf SÍ mueve los logits.
      Falsea: la versión de transformers ignora la máscara 4D aditiva (el modo
      de fallo silencioso más probable). El control separa "el 4D se ignora"
      de "el bias es demasiado chico para notarse".

  [3] PESOS — los checkpoints -base y -tsd difieren.
      Falsea: las dos corridas fueron idénticas. Misma seed, mismos datos,
      mismo orden → si el bias no entró, los pesos coinciden bit a bit.

Salida: PASS/FAIL por chequeo + veredicto. Exit code 0 si los tres pasan.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset

from ..config import load_config
from ..train_scratch import build_model, build_tokenizer
from ..train_sft import make_preprocess
from ..tsd.collator import NEG_INF, TSDCollator
from ..tsd.ultrametric import get_tree, ultrametric_matrix
from .train_tsd import make_tsd_extra

OK, BAD = "PASS", "FAIL"


def hr(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- chequeo 1
def check_structure(feats, K):
    """El árbol sobre datos reales, ¿tiene estructura o es degenerado?"""
    hr("[1] ESTRUCTURA DEL ÁRBOL sobre datos reales")
    frac_in, n_lvl0, n_lvl1, dvals, frac_biased = [], [], [], [], []

    for f in feats:
        n = len(f["input_ids"])
        P = np.asarray(f["paths"][:n], dtype=np.int32)
        idk = np.asarray(f["in_deck"][:n], dtype=bool)
        frac_in.append(idk.mean())
        if idk.sum() < 2:
            continue
        Pin = P[idk]
        n_lvl0.append(len(np.unique(Pin[:, 0])))
        if K > 1:
            n_lvl1.append(len(np.unique(Pin[:, 1])))
        D = ultrametric_matrix(P, K)
        both = idk[:, None] & idk[None, :]
        dvals.append(np.unique(D[both]))
        # pares que reciben bias != 0 (dentro del deck, distinta rama)
        frac_biased.append(float(((D > 0) & both).sum()) / max(1, both.sum()))

    uniq_d = sorted({float(v) for a in dvals for v in a})
    print(f"  ejemplos analizados      : {len(feats)}")
    print(f"  frac. tokens in_deck     : {np.mean(frac_in):.3f}  (el resto = prompt/CoT, sin bias)")
    print(f"  nivel 0 distintos/deck   : media {np.mean(n_lvl0):.1f}  min {min(n_lvl0)}  max {max(n_lvl0)}")
    if n_lvl1:
        print(f"  nivel 1 distintos/deck   : media {np.mean(n_lvl1):.1f}  min {min(n_lvl1)}  max {max(n_lvl1)}")
    print(f"  valores de D observados  : {uniq_d}")
    print(f"  frac. pares con bias!=0  : {np.mean(frac_biased):.3f}")

    ok = len(uniq_d) > 1 and np.mean(n_lvl0) > 1.5 and np.mean(frac_biased) > 0.05
    if not ok:
        print("\n  -> El árbol es DEGENERADO: casi todos los tokens caen en la misma rama.")
        print("     El bias es (casi) uniforme => no aporta señal estructural.")
    print(f"\n  {OK if ok else BAD}")
    return ok


# ---------------------------------------------------------------- chequeo 2
def check_forward(model, feats, K, lam, kernel, pad_id, dtype):
    """¿La máscara 4D llega a la atención? Con control -inf para discriminar."""
    hr(f"[2] FORWARD con/sin bias  (dtype={dtype})")
    model = model.to(dtype=dtype).eval()
    dev = next(model.parameters()).device

    # SOSPECHOSO Nº1: LlamaConfig(attn_implementation=...) puede NO propagarse al
    # instanciar el modelo (según versión de transformers). Si esto no dice "eager",
    # el 4D se descarta y el brazo --tsd entrenó igual que el baseline.
    eff = getattr(model.config, "_attn_implementation", "?")
    print(f"  attn_implementation      : {eff}   <- debe ser 'eager'")

    # norm=False reproduce las corridas históricas (la normalización por K es posterior).
    base_col = TSDCollator(pad_token_id=pad_id, use_tsd=False, K=K, norm=False)
    tsd_col = TSDCollator(pad_token_id=pad_id, use_tsd=True, lam=lam, kernel=kernel,
                          K=K, norm=False)

    batch = feats[:2]
    b_base, b_tsd = base_col(batch), tsd_col(batch)

    # sanity: las máscaras SÍ difieren antes de entrar al modelo
    mdiff = (b_tsd["attention_mask"] - b_base["attention_mask"]).abs().max().item()
    print(f"  |Δ máscara| (collator)   : {mdiff:.6f}   <- si es 0, el bug está en el collator")

    # control: máscara que prohíbe atender a los tokens 1..8 (efecto brutal, obligatorio)
    b_ctrl = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b_base.items()}
    b_ctrl["attention_mask"][:, :, :, 1:9] = NEG_INF

    def logits(b):
        b = {k: v.to(dev) for k, v in b.items()}
        b["attention_mask"] = b["attention_mask"].to(dtype)
        with torch.no_grad():
            return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()

    l_base, l_tsd, l_ctrl = logits(b_base), logits(b_tsd), logits(b_ctrl)
    d_tsd = (l_tsd - l_base).abs().max().item()
    d_ctrl = (l_ctrl - l_base).abs().max().item()

    print(f"  |Δ logits| tsd vs base   : {d_tsd:.6e}")
    print(f"  |Δ logits| CONTROL(-inf) : {d_ctrl:.6e}   <- control positivo")

    if d_ctrl < 1e-4:
        print("\n  -> La máscara 4D se está IGNORANDO por completo (ni -inf mueve los logits).")
        print("     Causa probable: attn_implementation != eager, o la versión de")
        print("     transformers reconstruye la causal mask y descarta la 4D recibida.")
        print("     TODO el 2x2 corrido es inservible.")
        return False
    if d_tsd < 1e-4:
        print("\n  -> El 4D SÍ se honra (el control mueve logits) pero el bias TSD no.")
        print("     Causa probable: bias ~0 => árbol degenerado o lam demasiado chico.")
        return False
    print("\n  -> El bias TSD llega a la atención y mueve la distribución.")
    print(f"  {OK}")
    return True


# ---------------------------------------------------------------- chequeo 3
def check_weights(out_dir):
    """¿Los checkpoints -base y -tsd difieren?"""
    hr("[3] PESOS -base vs -tsd")
    from transformers import AutoModelForCausalLM

    pa = f"{out_dir}-base"
    # "-tsd" = nombre histórico; "-tsd-<árbol>" = esquema nuevo (fallback/ast no se pisan)
    pb = next((p for p in (f"{out_dir}-tsd", f"{out_dir}-tsd-fallback", f"{out_dir}-tsd-ast")
               if os.path.isdir(p)), None)
    if not os.path.isdir(pa) or pb is None:
        print(f"  falta {pa} y/o el checkpoint -tsd -> chequeo OMITIDO.")
        print("  Causa probable: el 2x2 corrió SIN --ablation (TRAIN_ALL.md §A antiguo),")
        print("  así que ambas corridas escribieron en el mismo dir y la base se perdió.")
        return None
    print(f"  comparando {pa}  vs  {pb}")

    ma = AutoModelForCausalLM.from_pretrained(pa, dtype=torch.float32)
    mb = AutoModelForCausalLM.from_pretrained(pb, dtype=torch.float32)
    sa, sb = ma.state_dict(), mb.state_dict()

    if sa.keys() != sb.keys():
        print("  Los state_dict tienen claves distintas (arquitecturas distintas).")
        return False

    diffs = {k: (sa[k] - sb[k]).abs().max().item() for k in sa if sa[k].is_floating_point()}
    n_ident = sum(1 for v in diffs.values() if v == 0.0)
    top = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]

    print(f"  tensores comparados      : {len(diffs)}")
    print(f"  tensores IDÉNTICOS       : {n_ident}")
    print(f"  |Δ| máximo global        : {max(diffs.values()):.6e}")
    print("  top-5 tensores más distintos:")
    for k, v in top:
        print(f"    {v:.6e}  {k}")

    ok = n_ident < len(diffs) * 0.5 and max(diffs.values()) > 1e-6
    if not ok:
        print("\n  -> Los dos checkpoints son (casi) el MISMO modelo.")
        print("     Con misma seed y mismos datos, eso significa que el bias no tuvo efecto.")
    print(f"\n  {OK if ok else BAD}")
    return ok


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Verifica que el bias TSD llega a la atención (nano)")
    ap.add_argument("--config", default="configs/coder_nano.yaml")
    ap.add_argument("--n", type=int, default=64, help="ejemplos a analizar")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--kernel", default="linear", choices=["linear", "padic"])
    ap.add_argument("--tree", default="fallback", choices=["fallback", "ast", "ast-fam"],
                    help="DEJAR EN fallback para diagnosticar lo ya corrido; 'ast' es SEAM 2")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["model"].get("name_or_path"):
        sys.exit("Este verificador es para el nano from-scratch. Usa configs/coder_nano.yaml.")

    _, K = get_tree(args.tree)
    print(f"config={args.config}  árbol={args.tree}  K={K}  lam={args.lam}  kernel={args.kernel}")
    if args.tree == "fallback":
        print("AVISO: árbol=fallback (bloque por línea en blanco). Esto verifica la MECÁNICA")
        print("       del bias, no la hipótesis del AST. Sobre decks reales el fallback")
        print("       promedia 2.1 niveles de distancia -> es casi un indicador binario de")
        print("       'misma línea o no'. Correcto para diagnosticar lo ya corrido.")

    tok = build_tokenizer(cfg["tokenizer"])
    ds = load_dataset("json", data_files={"train": cfg["data"]["train"]})["train"]
    ds = ds.select(range(min(args.n, len(ds))))
    pp = make_preprocess(tok, cfg["data"]["max_seq_length"], extra=make_tsd_extra(args.tree))
    feats = [pp(ex) for ex in ds]

    r1 = check_structure(feats, K)

    # eager igual que el experimento: si no, SDPA descartaría la máscara 4D sin avisar.
    cfg["model"]["attn_implementation"] = "eager"
    model = build_model(cfg, vocab_size=len(tok))
    if torch.cuda.is_available():
        model = model.cuda()
    # bf16 = lo que usa el entrenamiento; fp32 separa "bias nulo" de "bias perdido en bf16"
    r2a = check_forward(model, feats, K, args.lam, args.kernel, tok.pad_token_id, torch.bfloat16)
    r2b = check_forward(model, feats, K, args.lam, args.kernel, tok.pad_token_id, torch.float32)

    r3 = None if args.skip_weights else check_weights(cfg["train"]["output_dir"])

    hr("VEREDICTO")
    print(f"  [1] estructura del árbol : {OK if r1 else BAD}")
    print(f"  [2] forward bf16         : {OK if r2a else BAD}")
    print(f"  [2] forward fp32         : {OK if r2b else BAD}")
    print(f"  [3] pesos base vs tsd    : {'OMITIDO' if r3 is None else (OK if r3 else BAD)}")
    hard = [r1, r2a, r2b] + ([] if r3 is None else [r3])
    if all(hard):
        print("\n  El bias TSD se aplicó. El 2x2 corrido es MECÁNICAMENTE válido")
        print("  (sigue sin ser la hipótesis del AST mientras SEAM 2 esté en fallback).")
        sys.exit(0)
    print("\n  El bias NO se aplicó como se esperaba. Ver el detalle de cada chequeo.")
    sys.exit(1)


if __name__ == "__main__":
    main()
