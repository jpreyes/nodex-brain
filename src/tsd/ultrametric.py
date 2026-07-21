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

# ============================ SEAM 2 — ÁRBOL AST ============================
# El AST del DSL es PLANO, no anidado. Medido, no supuesto:
#   · nodex-compiler/src/dsl/parser.js:18 = `lines.map(parseLine)` → 1 statement por línea.
#   · La capa de plantillas (for/repeat/if/def/let) la expande preprocess.js ANTES de
#     parsear; en corpus/ndx_decks.txt hay 0 ocurrencias de esas directivas.
# Por lo tanto la profundidad NO viene del anidamiento sino del TIPO del statement.
#
# Path por token, de grueso a fino (K=4):
#   0 familia semántica   ~7   (tabla FAMILIES, definida aquí — no existe en el compiler)
#   1 statement kind      37 observados de 60 en grammar-spec.js
#   2 instancia           nº de línea del statement
#   3 slot                0=keyword · 1=nombre · 2=argumentos
#
# Lo que esto compra sobre el fallback: los niveles 0-1 son TIPADOS y NO-LOCALES. Dos
# `beam` separados por 300 tokens quedan ultramétricamente cerca. El fallback solo sabe
# de proximidad posicional, que es lo que RoPE ya provee. Si el método aporta, aporta ahí.
AST_DEPTH = 4

# El kind es el primer token de la línea — misma regla que parseLine(), así que no hace
# falta invocar node para los niveles 0-2. Las 60 keys de STATEMENT_GRAMMAR agrupadas.
FAMILIES: dict[str, str] = {
    # definición de propiedades
    **{k: "definicion" for k in ("model", "material", "section", "usermat", "units", "include")},
    # geometría / topología
    **{k: "geometria" for k in ("node", "beam", "column", "slab", "wall", "cable", "area",
                                "solid", "grid", "generate", "arch", "disk", "box",
                                "triangulate", "subdivide", "infill", "fiber", "rna")},
    # conexiones entre elementos
    **{k: "conexion" for k in ("bolt", "weld", "contact", "link", "rigid", "couple",
                               "rbe2", "rbe3", "hinge", "spring", "diaphragm", "warping")},
    # condiciones de borde
    **{k: "condicion" for k in ("fix", "support", "prescribe", "mass", "soil", "heatBC")},
    # acciones
    **{k: "carga" for k in ("load", "combination", "spectrum", "accelerogram")},
    # análisis y post-proceso
    **{k: "analisis" for k in ("solve", "check", "nonlinear", "output", "sensor", "twin",
                               "calibrate", "objective", "constraints", "candidate", "norm",
                               "uel", "set", "assign", "override")},
    # metadatos: el comentario de cabecera (1 por deck en el corpus)
    "//": "meta",
}
_FAM_ID = {f: i for i, f in enumerate(sorted(set(FAMILIES.values())))}
_FAM_OTRO = len(_FAM_ID)          # kind desconocido → familia propia, no se mezcla
# IDs FIJOS y ordenados: ds.map(num_proc=16) forkea; un dict perezoso daría IDs distintos
# por proceso para el mismo kind. Los kinds fuera de la tabla colapsan en un único id
# (_KIND_OTRO): quedan en su propia rama, sin mezclarse con los conocidos.
_KIND_ID = {k: i for i, k in enumerate(sorted(FAMILIES))}
_KIND_OTRO = len(_KIND_ID)


def _kind_of(line: str) -> str | None:
    """Kind del statement = primer token de la línea (regla de parseLine)."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("//"):
        return "//"
    tok = s.split(None, 1)[0]
    return tok if tok else None


def ast_char_paths(deck: str) -> tuple[np.ndarray, int]:
    """SEAM 2 REAL — path [familia, kind, instancia, slot] por carácter del deck.

    Devuelve (paths [len(deck), AST_DEPTH], AST_DEPTH). Líneas en blanco → -1 (sin bias).
    Reemplaza a deck_char_paths(); esta última se conserva como fallback para la
    comparación honesta fallback-vs-AST.
    """
    n = len(deck)
    paths = np.full((n, AST_DEPTH), -1, dtype=np.int32)
    stmt = 0
    pos = 0
    for raw in deck.splitlines(keepends=True):
        kind = _kind_of(raw)
        if kind is None:
            pos += len(raw)
            continue
        stmt += 1
        fam = _FAM_ID.get(FAMILIES.get(kind, ""), _FAM_OTRO)
        kid = _KIND_ID.get(kind, _KIND_OTRO)

        # slots: 0 = keyword, 1 = nombre, 2 = argumentos. Cubre la forma dominante de
        # grammar-spec.js (`<kind> <name> <resto>`); en las que no la siguen (p.ej.
        # `fix <target>`) el nivel 3 degrada a un corte posicional, sin romper el prefijo.
        lead = len(raw) - len(raw.lstrip())
        body = raw[lead:]
        w0 = len(kind)
        rest = body[w0:]
        gap = len(rest) - len(rest.lstrip())
        name_start = w0 + gap
        name_end = name_start + len(body[name_start:].split(None, 1)[0]) if body[name_start:].strip() else len(body)

        a = pos + lead
        paths[a:pos + len(raw), 0] = fam
        paths[a:pos + len(raw), 1] = kid
        paths[a:pos + len(raw), 2] = stmt
        paths[a:a + w0, 3] = 0
        paths[a + w0:a + name_end, 3] = 1
        paths[a + name_end:pos + len(raw), 3] = 2
        pos += len(raw)
    return paths, AST_DEPTH


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


def get_tree(name: str):
    """Selector de árbol → (fn, K). `fallback` es el default histórico; `ast` es SEAM 2.

    Tenerlos ambos vivos no es indecisión: la comparación fallback-vs-AST a λ normalizado
    es la que separa "la jerarquía aporta" de "cualquier bias de localidad aporta".
    """
    if name == "fallback":
        return deck_char_paths, FALLBACK_DEPTH
    if name == "ast":
        return ast_char_paths, AST_DEPTH
    raise ValueError(f"árbol TSD desconocido: {name} (usa 'fallback' o 'ast')")


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


def kernel_bias(D: np.ndarray, lam: float, kernel: str = "linear", p: float = 2.0,
                K: int | None = None) -> np.ndarray:
    """SEAM 1 — kernel TSD: distancia ultramétrica → bias aditivo (<=0).

    linear : B = -lam * D                      (default, simple y monótono)
    padic  : B = -lam * (1 - p**(-D))          (saturación tipo p-ádico)
    Sustituye/añade tu kernel real aquí.

    K : si se pasa, normaliza D a [0,1] antes del kernel. IMPRESCINDIBLE para comparar
        árboles de distinta profundidad. Sobre un deck real, D media = 0.92 con el
        fallback (K=2) y 3.51 con el AST (K=4): a λ constante el cambio de árbol
        CUADRUPLICA la fuerza del bias, y el Δ medido mezcla estructura con fuerza.
        Normalizando, λ significa lo mismo en ambos y la comparación es limpia.
    """
    if K:
        D = D / float(K)
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
