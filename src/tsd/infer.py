"""Inferencia con bias TSD — celdas 3 y 4 del 2x2.

Decodifica token a token; con use_tsd=True re-arma en cada paso la máscara 4D
(causal + bias ultramétrico) sobre lo generado hasta ahora, tolerante a decks
incompletos. El prompt va neutral (in_deck=False).

=========================== SIMETRÍA TRAIN / INFER ===========================
Los parámetros del bias (árbol, K, λ, kernel, normalización) se LEEN del
`tsd_config.json` que el entrenamiento deja en el output_dir. No se declaran
aquí ni por CLI.

Esto no es comodidad: antes se repetían en dos sitios y se separaron sin que
nadie lo notara. El resultado era que el brazo TSD se entrenaba con un bias y
se evaluaba con otro (árbol fallback K=2 en vez del AST, sin normalizar, y
aplicado también al scratchpad) — o directamente sin bias si el modelo se
cargaba con SDPA, que descarta la máscara 4D en silencio. Las celdas +TSD
habrían medido ruido, sin dar ningún error.

Con un solo origen de verdad, esa asimetría deja de ser posible. `assert_eager()`
cubre el cuarto caso, que ningún parámetro compartido puede detectar.
=============================================================================

OJO (costo): esto corre SOLO por PyTorch/HF, NO por GGUF/llama.cpp, y es O(n^2)
sin KV-cache. Es para el EXPERIMENTO, no para producción. Con use_tsd=False es
un greedy decode normal (baseline, celdas 1 y 2).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .collator import NEG_INF
from .ultrametric import get_tree, kernel_bias, ultrametric_matrix

CoT_END = "</think>"
TSD_CONFIG = "tsd_config.json"


def save_tsd_config(model_dir: str, **cfg) -> None:
    """Deja los parámetros del bias junto al modelo (lo llama el entrenamiento)."""
    with open(os.path.join(model_dir, TSD_CONFIG), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)


def load_tsd_config(model_dir: str) -> dict:
    """Lee los parámetros con los que se ENTRENÓ este checkpoint.

    Si no existe el archivo, el checkpoint es anterior a este mecanismo: se asume
    baseline (sin bias) en vez de adivinar, porque adivinar es exactamente lo que
    causó la asimetría.
    """
    path = os.path.join(model_dir, TSD_CONFIG)
    if not os.path.isfile(path):
        print(f"[tsd] {TSD_CONFIG} no existe en {model_dir} → se asume BASELINE (sin bias). "
              f"Si este checkpoint es un brazo +TSD, su bias se perdió: re-entrénalo o "
              f"escribe el archivo a mano.")
        return {"use_tsd": False}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def assert_eager(model) -> None:
    """La máscara aditiva 4D exige eager. SDPA la descarta SIN AVISAR."""
    eff = getattr(model.config, "_attn_implementation", None)
    if eff != "eager":
        raise RuntimeError(
            f"attn_implementation={eff!r}, se necesita 'eager'. Con SDPA/flash la máscara "
            f"4D del bias TSD se descarta en silencio y la generación saldría SIN bias, "
            f"idéntica al baseline. Carga con from_pretrained(..., attn_implementation='eager').")


def _tsd_mask(prompt_len, gen_text, spans, tree, K, lam, kernel, p, norm, device):
    """Máscara 4D [1,1,L,L] = causal + bias ultramétrico sobre los tokens generados.

    El árbol se construye SOLO sobre el deck (después de `</think>`), igual que en el
    entrenamiento: el scratchpad es prosa y no tiene estructura de AST que sesgar.
    """
    L = prompt_len + len(spans)
    d_off = gen_text.find(CoT_END) + len(CoT_END) if CoT_END in gen_text else 0
    tree_fn, _ = get_tree(tree)
    char_paths, _K = tree_fn(gen_text[d_off:])

    paths = np.full((L, K), -1, dtype=np.int32)
    in_deck = np.zeros(L, dtype=bool)
    for t, start in enumerate(spans):              # start: char inicial del token en gen_text
        local = start - d_off
        if 0 <= local < char_paths.shape[0] and char_paths[local, 0] >= 0:
            paths[prompt_len + t] = char_paths[local]
            in_deck[prompt_len + t] = True

    D = ultrametric_matrix(paths, K)
    B = kernel_bias(D, lam, kernel, p, K=K if norm else None)
    both = in_deck[:, None] & in_deck[None, :]
    B[~both] = 0.0
    np.fill_diagonal(B, 0.0)
    m = torch.triu(torch.full((L, L), NEG_INF), diagonal=1)
    m += torch.from_numpy(B)
    return m.to(device)[None, None]


def _paths_de(gen_text, spans, prompt_len, L, tree, K):
    """paths[L,K] e in_deck[L] del estado actual. El prompt nunca recibe bias."""
    d_off = gen_text.find(CoT_END) + len(CoT_END) if CoT_END in gen_text else 0
    tree_fn, _ = get_tree(tree)
    char_paths, _K = tree_fn(gen_text[d_off:])
    paths = np.full((L, K), -1, dtype=np.int32)
    in_deck = np.zeros(L, dtype=bool)
    for t, start in enumerate(spans):
        local = start - d_off
        if 0 <= local < char_paths.shape[0] and char_paths[local, 0] >= 0:
            paths[prompt_len + t] = char_paths[local]
            in_deck[prompt_len + t] = True
    return paths, in_deck


def _bias_row(paths, in_deck, t, K, lam, kernel, p, norm):
    """Fila del bias: el token t contra todos los anteriores. O(L·K), no O(L²·K).

    Es lo que hace viable el KV-cache: con cache, cada paso solo atiende UNA consulta
    (el token nuevo) contra todas las claves, así que basta esta fila. Recomputar la
    matriz entera en cada paso costaba 63 h para los 6 brazos TSD del 2x2; con esto
    bajan a ~4 h.
    """
    pt = paths[t]
    eq = (paths == pt) & (paths != -1) & (pt != -1)[None, :]
    cpl = np.cumprod(eq, axis=1).sum(axis=1)
    D = (K - cpl).astype(np.float32)
    B = kernel_bias(D, lam, kernel, p, K=K if norm else None)
    B[~(in_deck & bool(in_deck[t]))] = 0.0
    B[t] = 0.0
    return B


@torch.no_grad()
def generate_tsd(model, tok, prompt: str, max_new_tokens: int = 512,
                 tsd_cfg: dict | None = None, model_dir: str | None = None,
                 use_cache: bool = True) -> str:
    """Genera con el MISMO bias con el que se entrenó el checkpoint.

    tsd_cfg   : dict de `load_tsd_config`. Si es None y se pasa model_dir, se lee de ahí.
    model_dir : carpeta del checkpoint (donde vive tsd_config.json).
    """
    if tsd_cfg is None:
        tsd_cfg = load_tsd_config(model_dir) if model_dir else {"use_tsd": False}
    use_tsd = bool(tsd_cfg.get("use_tsd"))
    if use_tsd:
        assert_eager(model)
        tree = tsd_cfg["tree"]
        K = tsd_cfg["K"]
        lam = tsd_cfg.get("lam", 1.0)
        kernel = tsd_cfg.get("kernel", "linear")
        p = tsd_cfg.get("p", 2.0)
        norm = tsd_cfg.get("norm", True)
        print(f"[tsd] generando CON bias: árbol={tree} K={K} λ={lam} kernel={kernel} norm={norm}")

    device = next(model.parameters()).device
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    eos = tok.eos_token_id

    # Sin bias no hay nada que rearmar: HF genera con KV-cache y es ~15x más rápido.
    if not use_tsd:
        out = model.generate(torch.tensor([prompt_ids], device=device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or eos)
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    gen: list[int] = []
    spans: list[int] = []          # offset char inicial de cada token generado
    prev_len = 0
    past = None

    for _ in range(max_new_tokens):
        if use_cache:
            if past is None:                       # 1er paso: el prompt entero, sin bias
                out = model(input_ids=torch.tensor([prompt_ids], device=device),
                            use_cache=True)
            else:
                # Solo el token nuevo. Su máscara es UNA fila [1,1,1,L]: la consulta
                # atiende a todas las claves, así que basta el bias de ese token contra
                # los anteriores. No hay que reconstruir la matriz [L,L].
                L = prompt_len + len(gen)
                paths, in_deck = _paths_de(tok.decode(gen, skip_special_tokens=False),
                                           spans, prompt_len, L, tree, K)
                row = _bias_row(paths, in_deck, L - 1, K, lam, kernel, p, norm)
                m = torch.from_numpy(row).to(device)[None, None, None, :]
                out = model(input_ids=torch.tensor([[gen[-1]]], device=device),
                            past_key_values=past, attention_mask=m, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
        else:
            # Ruta de REFERENCIA, O(n²): recomputa todo en cada paso. Solo para el test
            # de equivalencia — es la que costaba 63 h en el 2x2.
            seq = torch.tensor([prompt_ids + gen], device=device)
            mask = None
            if gen:
                mask = _tsd_mask(prompt_len, tok.decode(gen, skip_special_tokens=False),
                                 spans, tree, K, lam, kernel, p, norm, device)
            logits = model(input_ids=seq, attention_mask=mask).logits[0, -1]

        nxt = int(logits.argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        spans.append(prev_len)
        prev_len = len(tok.decode(gen, skip_special_tokens=False))

    return tok.decode(gen, skip_special_tokens=True).strip()
