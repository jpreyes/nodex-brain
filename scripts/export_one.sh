#!/usr/bin/env bash
# =============================================================================
# export_one.sh — exporta UN modelo ya entrenado a GGUF (f16 + Q4_K_M), con los
# mismos parches de tokenizer/config que usa run_all.sh. Para cuando el
# entrenamiento salió bien pero el export falló o quedó mal: re-exportar cuesta
# minutos, re-entrenar cuesta horas.
#
# Uso:
#   bash scripts/export_one.sh <model_dir> <base_id> <nombre-familia>
#
# Ejemplos:
#   bash scripts/export_one.sh models/ndx-coder-granite-350m-base \
#        ibm-granite/granite-4.0-350m ndx-coder-granite-350m
#   bash scripts/export_one.sh models/ndx-coder-qwen3-0.6b-base \
#        Qwen/Qwen3-0.6B-Base ndx-coder-qwen3-0.6b
#
# Para el nano (tokenizer propio) NO uses esto: bash scripts/export_gguf.sh.
# Para QLoRA (gemma4 e2b/e4b) hay que mergear antes; eso lo hace run_all.sh.
# =============================================================================
set -uo pipefail
export PYTHONIOENCODING=utf-8

BRAIN="${BRAIN:-/workspace/nodex-brain}"
LLAMA="${LLAMA:-/workspace/llama.cpp}"
OUT="${OUT:-$BRAIN/gguf}"
cd "$BRAIN"; mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

DIR="${1:?falta <model_dir>}"; BASE="${2:?falta <base_id>}"; NAME="${3:?falta <nombre-familia>}"
[ -d "$DIR" ] || { log "no existe $DIR"; exit 1; }

# transformers 5 pierde campos al re-guardar. Dos parches, mismos que run_all.sh:
#  - tokenizer: guarda tokenizer_class="TokenizersBackend" y a veces pierde vocab/merges.
#  - config: p.ej. granite pierde `layer_types` (28 x "attention") → el converter marca
#    las 28 capas como recurrentes y emite granitehybrid SIN tensores ssm → llama.cpp
#    falla con "missing tensor blk.0.ssm_in.weight". El fine-tune no cambia la
#    arquitectura, así que restauramos del cache de HF lo que falte.
SNAP=$(ls -d ~/.cache/huggingface/hub/models--${BASE//\//--}/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$SNAP" ]; then
  for t in tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt tokenizer.model; do
    [ -f "$SNAP/$t" ] && cp -f "$SNAP/$t" "$DIR/$t"
  done
  [ -f "$SNAP/config.json" ] && python - "$DIR/config.json" "$SNAP/config.json" <<'PY'
import json, sys
out, ref = sys.argv[1], sys.argv[2]
a = json.load(open(out, encoding="utf-8")); b = json.load(open(ref, encoding="utf-8"))
missing = [k for k in b if k not in a]
for k in missing: a[k] = b[k]
if missing:
    json.dump(a, open(out, "w", encoding="utf-8"), indent=2)
    print("config restaurado:", missing)
PY
else
  log "sin cache de HF para $BASE — exporto sin parches (puede fallar)"
fi

log "convert → $OUT/$NAME-f16.gguf"
python "$LLAMA/convert_hf_to_gguf.py" "$DIR" --outfile "$OUT/$NAME-f16.gguf" --outtype f16 || exit 1
log "quantize → $OUT/$NAME-Q4_K_M.gguf"
"$LLAMA/build/bin/llama-quantize" "$OUT/$NAME-f16.gguf" "$OUT/$NAME-Q4_K_M.gguf" Q4_K_M || exit 1

# Verificación: el gguf debe traer su chat_template (nodex-code arma el prompt con ella;
# sin plantilla el modelo queda fuera de distribución y el deck no sale).
python - "$OUT/$NAME-Q4_K_M.gguf" <<'PY'
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
arch = str(r.fields["general.architecture"].contents())
tpl = "tokenizer.chat_template" in r.fields
print(f"OK  arch={arch}  tensores={len(r.tensors)}  chat_template={'SI' if tpl else 'NO (nodex-code usará el fallback <|user|>)'}")
PY
log "listo: $NAME"
