"""TSD — bias de atención ULTRAMÉTRICO para el NDX-Coder.

Convierte la estructura jerárquica de un deck NDX en una matriz de distancia
ultramétrica entre tokens (distancia = altura del LCA en el árbol), que luego se
inyecta como bias aditivo de atención (ver src/tsd/collator.py).

Idea (tu TSD): tokens estructuralmente cercanos en el AST deben atenderse más;
lejanos, menos. La distancia es ULTRAMÉTRICA (cumple la desigualdad ultramétrica
fuerte) porque sale de un árbol: d(i,j) = altura del ancestro común más bajo.

================================  DOS SEAMS  ================================
SEAM 1 (kernel TSD): `kernel_bias()` trae un default lineal y uno p-ádico.
    Reemplázalo por tu kernel real (Haar/Kozyrev, funcional de energía, etc.).
SEAM 2 (AST real): `deck_char_paths()` usa un árbol de FALLBACK por bloques
    (líneas en blanco) → líneas. Enchúfalo al AST real de nodex-compiler
    (lower/analyze), que ya expone spans [startCol,endCol] por statement:
    cada token pasaría a llevar el PATH de nodos root→statement, y todo lo
    demás (LCA vía prefijo común, bias) queda igual pero con jerarquía rica.
===========================================================================
"""
from __future__ import annotations

import numpy as np

# Nº de niveles bajo la raíz en el árbol de fallback (bloque, línea).
FALLBACK_DEPTH = 2


def deck_char_paths(deck: str) -> tuple[np.ndarray, int]:
    """Path jerárquico por carácter del deck. Fallback: (bloque, línea).

    Devuelve (paths, K) con paths shape [len(deck), K], enteros >=0.
    Bloques separados por líneas en blanco; líneas por '\\n'.
    SEAM 2: sustituir por el path root→statement del AST de nodex-compiler.
    """
    n = len(deck)
    paths = np.full((n, FALLBACK_DEPTH), -1, dtype=np.int32)
    block = 0
    line = 0
    prev_blank = True
    pos = 0
    for raw in deck.splitlines(keepends=True):
        stripped = raw.strip()
        if stripped == "":
            # línea en blanco → cierra el bloque actual
            prev_blank = True
        else:
            if prev_blank:
                block += 1
                prev_blank = False
            line += 1
            paths[pos:pos + len(raw), 0] = block
            paths[pos:pos + len(raw), 1] = line
        pos += len(raw)
    return paths, FALLBACK_DEPTH


def token_paths(
    offsets: list[tuple[int, int]],
    deck_span: tuple[int, int],
    char_paths: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mapea cada token (por su offset en chars del TEXTO COMPLETO) a su path.

    offsets   : lista [(start,end)] por token, en chars del texto completo.
    deck_span : (ini, fin) del deck dentro del texto completo.
    Devuelve (paths[T,K], in_deck[T] bool). Tokens fuera del deck (prompt/pad)
    van con path sentinela y in_deck=False → no reciben bias.
    """
    T = len(offsets)
    d0, d1 = deck_span
    paths = np.full((T, K), -1, dtype=np.int32)
    in_deck = np.zeros(T, dtype=bool)
    for t, (s, _e) in enumerate(offsets):
        if d0 <= s < d1:
            local = s - d0
            if 0 <= local < char_paths.shape[0] and char_paths[local, 0] >= 0:
                paths[t] = char_paths[local]
                in_deck[t] = True
    return paths, in_deck


def ultrametric_matrix(paths: np.ndarray, K: int) -> np.ndarray:
    """Matriz ultramétrica D[T,T] = K - (largo del prefijo común de paths).

    Vectorizado. d = 0 (mismo statement/línea) … K (sin ancestro común salvo raíz).
    """
    # equal[i,j,k] = paths[i,k]==paths[j,k] AND ambos != -1
    P = paths[:, None, :] == paths[None, :, :]        # [T,T,K]
    valid = (paths[:, None, :] != -1) & (paths[None, :, :] != -1)
    eq = P & valid                                     # [T,T,K]
    # prefijo común = suma de AND acumulado a lo largo de K
    cpl = np.cumprod(eq, axis=2).sum(axis=2)           # [T,T]
    return (K - cpl).astype(np.float32)


def kernel_bias(D: np.ndarray, lam: float, kernel: str = "linear", p: float = 2.0) -> np.ndarray:
    """SEAM 1 — kernel TSD: distancia ultramétrica → bias aditivo (<=0).

    linear : B = -lam * D                      (default, simple y monótono)
    padic  : B = -lam * (1 - p**(-D))          (saturación tipo p-ádico)
    Sustituye/añade tu kernel real aquí.
    """
    if kernel == "linear":
        B = -lam * D
    elif kernel == "padic":
        B = -lam * (1.0 - np.power(p, -D))
    else:
        raise ValueError(f"kernel TSD desconocido: {kernel}")
    return B.astype(np.float32)


def tsd_bias_matrix(
    offsets: list[tuple[int, int]],
    deck: str,
    deck_span: tuple[int, int],
    lam: float = 1.0,
    kernel: str = "linear",
    p: float = 2.0,
) -> np.ndarray:
    """Pipeline completo: offsets+deck → bias aditivo B[T,T] (0 fuera del deck)."""
    char_paths, K = deck_char_paths(deck)
    paths, in_deck = token_paths(offsets, deck_span, char_paths, K)
    D = ultrametric_matrix(paths, K)
    B = kernel_bias(D, lam=lam, kernel=kernel, p=p)
    # bias sólo entre tokens del deck; cualquier par con prompt/pad → 0
    both = in_deck[:, None] & in_deck[None, :]
    B[~both] = 0.0
    np.fill_diagonal(B, 0.0)
    return B
