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

# transformers 5 pierde tokenizer y claves de config al re-guardar (ver fix_hf_export.py).
# Si esto falla NO seguimos: convertir igual produce un gguf que no carga.
log "reparando carpeta HF contra $BASE"
python scripts/fix_hf_export.py "$DIR" "$BASE" || { log "no se pudo reparar $DIR — abortado"; exit 1; }

log "convert → $OUT/$NAME-f16.gguf"
python "$LLAMA/convert_hf_to_gguf.py" "$DIR" --outfile "$OUT/$NAME-f16.gguf" --outtype f16 || exit 1
log "quantize → $OUT/$NAME-Q4_K_M.gguf"
"$LLAMA/build/bin/llama-quantize" "$OUT/$NAME-f16.gguf" "$OUT/$NAME-Q4_K_M.gguf" Q4_K_M || exit 1

# Verificación del gguf. Dos cosas que ya fallaron en silencio y costaron horas:
#  - sin chat_template, nodex-code cae al fallback <|user|> y el modelo queda fuera de
#    distribución (qwen3 daba 1/20 decks por esto).
#  - un arch híbrido que declara capas recurrentes pero no trae tensores ssm NO CARGA
#    ("missing tensor blk.0.ssm_in.weight"). Es el bug de granite: head_count_kv=[0]*N
#    significa "todas las capas son mamba", y si no hay tensores ssm, mienten.
python - "$OUT/$NAME-Q4_K_M.gguf" <<'PY' || { log "gguf INVÁLIDO — no lo entregues"; exit 1; }
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
arch = str(r.fields["general.architecture"].contents())
n_ssm = sum(1 for t in r.tensors if "ssm" in t.name)
tpl = "tokenizer.chat_template" in r.fields
print(f"arch={arch}  tensores={len(r.tensors)}  ssm={n_ssm}  chat_template={'SI' if tpl else 'NO'}")

bad = []
if not tpl:
    bad.append("sin tokenizer.chat_template -> nodex-code usaría el fallback <|user|>")
kv = r.fields.get(f"{arch}.attention.head_count_kv")
if kv is not None:
    vals = list(kv.contents()) if hasattr(kv.contents(), "__len__") else [kv.contents()]
    n_rec = sum(1 for v in vals if int(v) == 0)
    if n_rec and not n_ssm:
        bad.append(f"{n_rec}/{len(vals)} capas declaradas recurrentes pero 0 tensores ssm "
                   f"-> el modelo NO va a cargar (config.json sin layer_types correcto)")
if bad:
    print("FALLA:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("OK")
PY
log "listo: $NAME"
