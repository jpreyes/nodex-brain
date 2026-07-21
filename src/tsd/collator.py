"""Collator TSD: arma la máscara de atención 4D = causal + bias ultramétrico.

Cada ejemplo llega con paths compactos por token ([K] ints) + in_deck. El bias
ultramétrico se computa AQUÍ (barato, O(L^2*K) por batch) para no cachear
matrices [L,L] gigantes. HF suma esta máscara 4D a los scores de atención
(requiere attn_implementation="eager").

use_tsd=False → bias 0 → baseline EXACTO (misma ruta de código, solo cambia el
bias), para una ablación con/sin TSD limpia.
"""
from __future__ import annotations

import numpy as np
import torch

from .ultrametric import FALLBACK_DEPTH, kernel_bias, ultrametric_matrix

NEG_INF = torch.finfo(torch.float32).min


class TSDCollator:
    def __init__(self, pad_token_id, use_tsd=True, lam=1.0, kernel="linear", p=2.0,
                 K=FALLBACK_DEPTH, norm=True):
        self.pad_token_id = pad_token_id
        self.use_tsd = use_tsd
        self.lam = lam
        self.kernel = kernel
        self.p = p
        self.K = K
        # norm=True → D se escala a [0,1], así λ significa lo mismo para K=2 y K=4.
        # Sin esto, cambiar de árbol cambia la FUERZA del bias además de su estructura
        # (D media 0.97 con fallback vs 3.42 con ast) y el Δ medido es inatribuible.
        # Las corridas históricas fueron SIN normalizar (equivale a norm=False).
        self.norm = norm

    def _bias(self, paths, in_deck, n):
        P = np.asarray(paths[:n], dtype=np.int32)          # [n,K]
        D = ultrametric_matrix(P, self.K)                  # [n,n]
        B = kernel_bias(D, self.lam, self.kernel, self.p,
                        K=self.K if self.norm else None)   # [n,n] <=0
        idk = np.asarray(in_deck[:n], dtype=bool)
        both = idk[:, None] & idk[None, :]
        B[~both] = 0.0
        np.fill_diagonal(B, 0.0)
        return torch.from_numpy(B)

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        B = len(features)
        input_ids = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        mask4d = torch.zeros((B, 1, L, L), dtype=torch.float32)
        causal = torch.triu(torch.full((L, L), NEG_INF), diagonal=1)

        for b, f in enumerate(features):
            ids = f["input_ids"]
            n = len(ids)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
            labels[b, :n] = torch.tensor(f["labels"], dtype=torch.long)
            m = causal.clone()
            m[:, n:] = NEG_INF                                   # no atender al padding
            if self.use_tsd:
                m[:n, :n] += self._bias(f["paths"], f["in_deck"], n)
            mask4d[b, 0] = m

        return {"input_ids": input_ids, "attention_mask": mask4d, "labels": labels}
