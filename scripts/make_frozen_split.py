#!/usr/bin/env python
"""Congela el split de evaluación de EXPERIMENTS.md §5.2. Determinista, sin aleatoriedad.

El problema que resuelve (§5): el split train/val actual FILTRA — 9,2% de los decks de val
están literalmente en train, y el 100% de las estructuras de val se vieron en train. Toda
métrica mirada hasta ahora está inflada.

Produce dos test sets, porque uno solo no sirve (§5.1):

  Test A — in-distribution. Dedupe exacto, luego 12% por hash del deck. Estructuras vistas,
           PARÁMETROS nuevos. Es el primario: los kinds raros siguen en train (aprendibles) y
           aparecen en test con parámetros nuevos, así que la métrica de cola es medible.

  Test B — estructural. FIRMAS completas fuera del entrenamiento. Estructuras nunca
           vistas. Secundario: mide generalización real. Exige reentrenar con `train.jsonl`
           de acá — un modelo entrenado sobre el corpus completo NO puede evaluarse contra B.

FIRMA = el CONJUNTO ordenado de kinds del deck, sin orden ni repeticiones (56 en el pool).

Antes se usaba la secuencia completa (115). Se cambió porque medimos que **no aísla**
(§19.1): el 70% de las 115 firmas-secuencia tiene un hermano con el mismo conjunto de kinds
en otro orden, así que dejar fuera 12 secuencias dejaba 8,3 de 12 con un hermano estructural
EN TRAIN. Test B no medía estructuras no vistas sino ORDENAMIENTOS no vistos, que es mucho
más débil y no responde §5.3.

La gruesa es la conservadora: apartar un conjunto de kinds aparta todas sus secuencias, así
que la fuga es 0 bajo AMBAS definiciones (se comprueban las dos). Y cuesta lo mismo — 6
firmas-conjunto dan ~1.770 ejemplos, igual que las 12 firmas-secuencia de antes.

Firmas PROTEGIDAS (nunca van a Test B): las que son el único hogar de algún kind. Si esa
firma sale del entrenamiento, el kind desaparece por completo y no se mide "difícil" sino
"imposible" (§5.1).

    python scripts/make_frozen_split.py [--out <dir>] [--test-b-sigs 8] [--test-a-pct 12]
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

SRC = "../nodex-code/datasets/generator-combined-cot-sft-40185"
OUT = "../nodex-code/datasets/frozen-split-v2"   # v1 usaba la firma fina (§19.1)


def deck_of(ex):
    """El deck del target, sin el scratchpad."""
    t = [m for m in ex["messages"] if m["role"] == "assistant"][-1]["content"]
    return t.split("</think>")[-1].strip()


def kinds_of(deck):
    """Kinds del deck, en orden de aparición y con repeticiones."""
    ks = []
    for line in deck.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        ks.append(line.split()[0])
    return ks


def signature(deck):
    """FIRMA (gruesa): conjunto ordenado de kinds. 56 en el pool. Ver el docstring."""
    return ",".join(sorted(set(kinds_of(deck))))


def signature_fine(deck):
    """Firma fina: la secuencia completa. Solo para COMPROBAR que tampoco filtra."""
    return ",".join(kinds_of(deck))


def h(s):
    """Hash estable entre corridas y máquinas (hash() de Python está salado por proceso)."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--test-b-sigs", type=int, default=8,
                    help="cuántas FIRMAS aparta para Test B. Es el n que importa: para la "
                         "pregunta estructural la unidad de análisis es la firma, no el "
                         "ejemplo (§19.3)")
    ap.add_argument("--test-a-pct", type=int, default=12)
    args = ap.parse_args()

    # 1. Pool: train + val JUNTOS. El val viejo filtra (§5), así que se disuelve y se
    #    reparte de nuevo; conservarlo sería conservar la fuga.
    pool = []
    for f in ("train.jsonl", "val.jsonl"):
        with open(os.path.join(args.src, f), encoding="utf-8") as fh:
            pool += [json.loads(line) for line in fh if line.strip()]
    print(f"pool: {len(pool)} ejemplos (train + val)")

    # 2. Dedupe EXACTO por deck. El split va por deck completo, nunca por línea: dos líneas
    #    del mismo deck son casi duplicadas y separarlas es fuga trivial (§5.2).
    seen, uniq = set(), []
    for ex in pool:
        d = deck_of(ex)
        k = hashlib.sha1(d.encode("utf-8")).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        uniq.append((ex, d, signature(d), signature_fine(d)))
    print(f"tras dedupe exacto: {len(uniq)} ({len(pool) - len(uniq)} duplicados fuera)")

    por_firma = defaultdict(list)
    for item in uniq:
        por_firma[item[2]].append(item)
    print(f"firmas distintas: {len(por_firma)}")

    # 3. Firmas protegidas: únicas dueñas de algún kind.
    firmas_de_kind = defaultdict(set)
    for sig in por_firma:
        for k in set(sig.split(",")):
            firmas_de_kind[k].add(sig)
    protegidas = {next(iter(v)) for k, v in firmas_de_kind.items() if len(v) == 1}
    monopolios = sorted(k for k, v in firmas_de_kind.items() if len(v) == 1)
    print(f"kinds que viven en UNA sola firma: {len(monopolios)} -> {' '.join(monopolios)}")
    print(f"firmas protegidas (no pueden ir a Test B): {len(protegidas)}")

    # 4. Test B: las `--test-b-sigs` firmas elegibles de hash más bajo. Determinista y sin
    #    criterio de conveniencia — elegirlas "a ojo" invitaría a moverlas después.
    #
    #    Se selecciona por NÚMERO DE FIRMAS, no de ejemplos. Para la pregunta que Test B
    #    contesta —¿generaliza a estructuras no vistas?— la unidad de análisis es la
    #    ESTRUCTURA: con 3 firmas y 1.864 ejemplos no se puede separar "generaliza" de "esas
    #    tres eran fáciles", porque no hay varianza entre estructuras que estimar. Y como los
    #    tamaños son muy desiguales (5 a 1.500 ejemplos por firma), fijar un objetivo en
    #    ejemplos dejaba el número de firmas al azar.
    elegibles = sorted(sorted(s for s in por_firma if s not in protegidas), key=h)
    test_b_sigs = elegibles[: args.test_b_sigs]
    test_b = [i for s in test_b_sigs for i in por_firma[s]]
    print(f"Test B: {len(test_b_sigs)} firmas, {len(test_b)} ejemplos "
          f"({100 * len(test_b) / len(uniq):.1f}% del pool deduplicado)")
    print("  ejemplos por firma: " + " ".join(str(len(por_firma[s])) for s in test_b_sigs))

    # 5. Test A: `--test-a-pct` % del resto, por hash del deck.
    resto = [i for s, items in por_firma.items() if s not in set(test_b_sigs) for i in items]
    test_a = [i for i in resto if h(i[1]) % 100 < args.test_a_pct]
    train = [i for i in resto if h(i[1]) % 100 >= args.test_a_pct]
    print(f"Test A: {len(test_a)} ejemplos · train: {len(train)} ejemplos")

    # 6. Comprobaciones que deben pasar SIEMPRE. Si alguna falla, el split no sirve.
    decks_train = {i[1] for i in train}
    fuga_a = sum(1 for i in test_a if i[1] in decks_train)
    fuga_b = sum(1 for i in test_b if i[1] in decks_train)
    sigs_train = {i[2] for i in train}
    fuga_b_estructural = sum(1 for s in test_b_sigs if s in sigs_train)
    sigs_fine_train = {i[3] for i in train}
    fuga_b_fina = sum(1 for i in test_b if i[3] in sigs_fine_train)
    kinds_train = {k for s in sigs_train for k in s.split(",")}
    kinds_perdidos = sorted({k for s in test_b_sigs for k in s.split(",")} - kinds_train)
    print("\ncomprobaciones:")
    print(f"  fuga exacta A->train      : {fuga_a}   (debe ser 0)")
    print(f"  fuga exacta B->train      : {fuga_b}   (debe ser 0)")
    print(f"  fuga estructural B->train : {fuga_b_estructural}   (debe ser 0, firma GRUESA)")
    print(f"  fuga estructural B (fina) : {fuga_b_fina}   (debe ser 0 tambien; la gruesa la implica)")
    print(f"  kinds que Test B saca del entrenamiento: {kinds_perdidos or 'ninguno'}   (debe ser ninguno)")
    assert fuga_a == 0 and fuga_b == 0 and fuga_b_estructural == 0 and not kinds_perdidos
    assert fuga_b_fina == 0, 'la firma gruesa deberia implicar la fina'

    # 7. Escribir. Formato `messages`, igual que la fuente.
    os.makedirs(args.out, exist_ok=True)
    for nombre, items in (("train.jsonl", train), ("test_a.jsonl", test_a), ("test_b.jsonl", test_b)):
        with open(os.path.join(args.out, nombre), "w", encoding="utf-8") as fh:
            for ex, _d, _s, _sf in items:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    manifest = {
        "generado_por": "scripts/make_frozen_split.py",
        "fuente": args.src,
        "definicion_de_firma": "CONJUNTO ordenado de kinds del deck (sin orden ni repeticiones)",
        "por_que_gruesa": "la fina (secuencia) no aisla: 70% de sus firmas tiene un hermano con el mismo conjunto de kinds, asi que Test B medía ordenamientos no vistos, no estructuras (EXPERIMENTS.md §19.1)",
        "determinista": "sha1 del deck / de la firma; sin RNG, sin semilla",
        "split_por": "deck completo (nunca por línea)",
        "test_a": {"n": len(test_a), "criterio": f"hash(deck) %% 100 < {args.test_a_pct}",
                   "mide": "estructuras vistas, parámetros nuevos"},
        "test_b": {"n": len(test_b), "firmas": len(test_b_sigs),
                   "criterio": f"las {args.test_b_sigs} firmas elegibles de hash más bajo",
                   "unidad_de_analisis": "la FIRMA, no el ejemplo: n=%d estructuras" % len(test_b_sigs),
                   "ejemplos_por_firma": [len(por_firma[s]) for s in test_b_sigs],
                   "mide": "estructuras nunca vistas",
                   "requiere": "reentrenar con este train.jsonl; un modelo entrenado sobre el corpus completo NO puede evaluarse contra B"},
        "train": {"n": len(train), "firmas": len(sigs_train)},
        "firmas_test_b": test_b_sigs,
        "kinds_monopolio_protegidos": monopolios,
        "comprobado": {"fuga_exacta_a": fuga_a, "fuga_exacta_b": fuga_b,
                       "fuga_estructural_b": fuga_b_estructural,
                       "fuga_estructural_b_firma_fina": fuga_b_fina,
                       "kinds_perdidos": kinds_perdidos},
        "CONGELADO": "no regenerar. Si cambia el corpus, crear un split nuevo con otro nombre.",
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # Ejemplos de COLA por test: es lo que fija la potencia de la métrica (§13.5/E9, §19.3).
    COLA = {"accelerogram", "area", "assign", "box", "cable", "combination", "contact",
            "diaphragm", "fiber", "heatBC", "link", "nonlinear", "slab", "soil", "solid",
            "spectrum", "wall"}
    print("\nejemplos que usan al menos un kind de COLA (fija la potencia del recall):")
    for nombre, items in (("Test A", test_a), ("Test B", test_b), ("train", train)):
        n = sum(1 for i in items if set(i[2].split(",")) & COLA)
        print(f"  {nombre:7s} {n:6d} de {len(items):6d}")

    fr = Counter(len(v) for v in por_firma.values())
    print(f"\nescrito en {args.out}/ (train.jsonl · test_a.jsonl · test_b.jsonl · MANIFEST.json)")
    print(f"ejemplos por firma: min {min(fr)} · max {max(fr)}")


if __name__ == "__main__":
    main()
