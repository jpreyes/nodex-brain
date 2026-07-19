"""Cosecha texto de los datasets `messages` para (a) entrenar el tokenizer y
(b) un corpus de decks NDX puros para pretraining de "lenguaje NDX".

Uso:
    python -m src.harvest_ndx \
        --datasets ../nodex-code/datasets/generator-combined-cot-sft-40185/train.jsonl \
        --out-corpus corpus/tokenizer.txt \
        --out-decks corpus/ndx_decks.txt

- `--out-corpus`: todo lo que el modelo lee/escribe (user + assistant, incl. <think>
  y deck) → para entrenar el tokenizer sobre la distribución real.
- `--out-decks`: solo el deck NDX (sin <think>) → corpus de sintaxis para
  pretraining opcional.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest de corpus NDX")
    parser.add_argument("--datasets", nargs="+", required=True, help="Uno o más .jsonl")
    parser.add_argument("--out-corpus", default=None, help="Texto para el tokenizer")
    parser.add_argument("--out-decks", default=None, help="Solo decks NDX (sin think)")
    args = parser.parse_args()

    corpus_fh = None
    decks_fh = None
    if args.out_corpus:
        Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
        corpus_fh = open(args.out_corpus, "w", encoding="utf-8")
    if args.out_decks:
        Path(args.out_decks).parent.mkdir(parents=True, exist_ok=True)
        decks_fh = open(args.out_decks, "w", encoding="utf-8")

    n_examples = 0
    n_decks = 0
    for path in args.datasets:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                msgs = json.loads(line)["messages"]
                n_examples += 1
                for m in msgs:
                    content = m["content"]
                    if corpus_fh:
                        corpus_fh.write(content)
                        corpus_fh.write("\n")
                    if decks_fh and m["role"] == "assistant":
                        deck = strip_think(content)
                        if deck:
                            decks_fh.write(deck)
                            decks_fh.write("\n\n")
                            n_decks += 1

    if corpus_fh:
        corpus_fh.close()
    if decks_fh:
        decks_fh.close()

    print(f"Ejemplos procesados: {n_examples}")
    if args.out_corpus:
        print(f"Corpus tokenizer  -> {args.out_corpus}")
    if args.out_decks:
        print(f"Decks NDX ({n_decks}) -> {args.out_decks}")


if __name__ == "__main__":
    main()
