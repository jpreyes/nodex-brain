"""Inferencia con bias TSD opcional (celdas 3 y 4 del 2x2).

Decodifica token a token; si use_tsd=True, en cada paso re-arma la máscara 4D
(causal + bias ultramétrico) sobre lo generado hasta ahora (parseo incremental
por bloques/líneas — el mismo fallback de ultrametric.py, tolerante a decks
incompletos). El prompt va neutral (in_deck=False).

OJO (costo de despliegue): esto corre SOLO por PyTorch/HF, NO por GGUF/llama.cpp,
y es O(n^2) sin KV-cache. Es para el EXPERIMENTO/eval, no para producción.
Con use_tsd=False es un greedy decode normal (baseline / celdas 1 y 2).
"""
from __future__ import annotations

import numpy as np
import torch

from .collator import NEG_INF
from .ultrametric import deck_char_paths, kernel_bias, ultrametric_matrix


def _tsd_mask(prompt_len: int, gen_ids: list[int], deck_text: str,
              spans: list[int], lam: float, kernel: str, p: float, device) -> torch.Tensor:
    """Máscara 4D [1,1,L,L] = causal + bias ultramétrico sobre los tokens generados."""
    L = prompt_len + len(gen_ids)
    char_paths, K = deck_char_paths(deck_text)
    paths = np.full((L, K), -1, dtype=np.int32)
    in_deck = np.zeros(L, dtype=bool)
    for t, start in enumerate(spans):                     # spans: char inicial de cada token generado
        if 0 <= start < char_paths.shape[0] and char_paths[start, 0] >= 0:
            paths[prompt_len + t] = char_paths[start]
            in_deck[prompt_len + t] = True
    D = ultrametric_matrix(paths, K)
    B = kernel_bias(D, lam, kernel, p)
    both = in_deck[:, None] & in_deck[None, :]
    B[~both] = 0.0
    np.fill_diagonal(B, 0.0)
    m = torch.triu(torch.full((L, L), NEG_INF), diagonal=1)
    m += torch.from_numpy(B)
    return m.to(device)[None, None]


@torch.no_grad()
def generate_tsd(model, tok, prompt: str, max_new_tokens: int = 512,
                 use_tsd: bool = False, lam: float = 1.0, kernel: str = "linear", p: float = 2.0) -> str:
    device = next(model.parameters()).device
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    eos = tok.eos_token_id

    gen: list[int] = []
    spans: list[int] = []          # offset char inicial de cada token generado, en el deck decodificado
    prev_len = 0

    for _ in range(max_new_tokens):
        seq = torch.tensor([prompt_ids + gen], device=device)
        mask = None
        if use_tsd and gen:
            deck_text = tok.decode(gen, skip_special_tokens=False)
            mask = _tsd_mask(prompt_len, gen, deck_text, spans, lam, kernel, p, device)
        logits = model(input_ids=seq, attention_mask=mask).logits[0, -1]
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        # span incremental del nuevo token en el deck decodificado
        full = tok.decode(gen, skip_special_tokens=False)
        spans.append(prev_len)
        prev_len = len(full)

    return tok.decode(gen, skip_special_tokens=True).strip()
