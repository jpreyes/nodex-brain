#!/usr/bin/env bash
# Exporta el NDX-Coder entrenado a GGUF y lo cuantiza para correr en CPU+RAM
# (llama.cpp). Corre esto DESPUÉS de entrenar (con models/ndx-coder-small listo).
#
# Uso:  bash scripts/export_gguf.sh [MODEL_DIR] [OUT_DIR]
set -euo pipefail

MODEL_DIR="${1:-models/ndx-coder-small}"
OUT_DIR="${2:-gguf}"
NAME="$(basename "$MODEL_DIR")"          # nombra los .gguf según el modelo
LLAMA_DIR="${LLAMA_CPP_DIR:-../llama.cpp}"

mkdir -p "$OUT_DIR"

# 1. llama.cpp (clonar + build de las herramientas de cuantización si falta)
if [ ! -d "$LLAMA_DIR" ]; then
  echo ">> clonando llama.cpp en $LLAMA_DIR"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
# OJO: NO instalar el requirements.txt completo de llama.cpp — reinstala torch
# (a veces la build CPU) y rompe el entorno de entrenamiento. Solo lo mínimo que
# el converter necesita (torch/transformers ya están instalados).
pip install -q gguf numpy sentencepiece protobuf

# 1a. Parche del converter: nuestro tokenizer es BPE byte-level (estilo GPT-2),
#     pero su hash no está en la lista de llama.cpp → en vez de abortar, cae a
#     "gpt-2" (el pre-tokenizer correcto para ByteLevel BPE).
for f in $(grep -rl 'BPE pre-tokenizer was not recognized' "$LLAMA_DIR" 2>/dev/null); do
  sed -i 's/raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")/res = "gpt-2"  # NDX-Coder ByteLevel BPE fallback/' "$f"
done

# 1b. Parche del tokenizer_config: transformers 5 lo guarda de forma que el
#     converter de llama.cpp no puede recargarlo (clase "TokenizersBackend" y
#     extra_special_tokens como lista en vez de dict). Lo normalizamos.
echo ">> parcheando tokenizer_config.json para el converter"
python - "$MODEL_DIR/tokenizer_config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d["tokenizer_class"] = "PreTrainedTokenizerFast"
if isinstance(d.get("extra_special_tokens"), list):
    d["extra_special_tokens"] = {}
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
print("  ok")
PY

# 2. HF -> GGUF (f16). El converter lee el tokenizer.json (BPE byte-level) y la
#    arquitectura Llama de config.json.
echo ">> convirtiendo a GGUF f16"
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
  --outfile "$OUT_DIR/$NAME-f16.gguf" --outtype f16

# 3. Cuantizar (necesita el binario llama-quantize; build si no está)
QUANT="$LLAMA_DIR/build/bin/llama-quantize"
if [ ! -x "$QUANT" ]; then
  echo ">> compilando herramientas de llama.cpp"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$LLAMA_DIR/build" --target llama-quantize -j
fi

echo ">> cuantizando Q4_K_M"
"$QUANT" "$OUT_DIR/$NAME-f16.gguf" "$OUT_DIR/$NAME-Q4_K_M.gguf" Q4_K_M

echo ""
echo "Listo. Tamaños:"
ls -lh "$OUT_DIR"/*.gguf | awk '{print "  "$5"\t"$9}'
echo ""
echo "Q4 listo: $OUT_DIR/$NAME-Q4_K_M.gguf"
echo "Para liberar disco puedes borrar el f16:  rm $OUT_DIR/$NAME-f16.gguf"
