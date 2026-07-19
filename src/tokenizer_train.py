"""Entrena un tokenizer BPE byte-level propio para NDX-Coder (desde cero).

Byte-level BPE (sin `unk`, alfabeto de 256 bytes) → maneja español + símbolos NDX
(`m²`, `m⁴`, `kN`) sin caracteres desconocidos. Tokens especiales para el chat
template y el scratchpad (`<think>` / `</think>`) como unidades atómicas.

Uso:
    python -m src.tokenizer_train --corpus corpus/tokenizer.txt \
        --out tokenizer/ndx --vocab-size 16000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Tokens especiales — atómicos, nunca se parten.
SPECIAL_TOKENS = [
    "<|endoftext|>",   # fin de secuencia / padding
    "<|user|>",        # inicio del turno de usuario
    "<|assistant|>",   # inicio del turno del asistente
    "<think>",         # apertura del scratchpad
    "</think>",        # cierre del scratchpad
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el tokenizer de NDX-Coder")
    parser.add_argument("--corpus", required=True, help="Texto de entrenamiento")
    parser.add_argument("--out", required=True, help="Directorio de salida")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([args.corpus], trainer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    vocab = tokenizer.get_vocab()
    print(f"vocab final: {len(vocab)}")
    print(f"guardado en: {out / 'tokenizer.json'}")
    # sanity: los especiales deben ser 1 token cada uno
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.encode(tok).ids
        print(f"  {tok!r:16} -> {len(ids)} token(s)")


if __name__ == "__main__":
    main()
