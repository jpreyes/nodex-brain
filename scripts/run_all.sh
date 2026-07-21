#!/usr/bin/env bash
# =============================================================================
# run_all.sh — entrena TODA la familia ndx-coder + ablación TSD, exporta a GGUF
# y sube TODO (modelos + gguf f16 + Q4) a Dropbox/Google Drive vía rclone.
#
# Uso:
#   bash scripts/run_all.sh smoke [remote]        # valida el pipeline entero en TINY (rápido)
#   bash scripts/run_all.sh full  remote:carpeta  # corridas reales + gguf + upload
#
# <remote> = un remote de rclone YA configurado (`rclone config`, tipo Dropbox o
#            Google Drive), p.ej.  store:nodex-models
#
# Filosofía: smoke usa EXACTAMENTE el mismo código que full (subset + max-steps),
# así lo que pasa en smoke pasa en full (salvo OOM/tiempo). Fallos NO abortan todo:
# se acumulan y se reportan al final para repararlos.
# =============================================================================
set -uo pipefail
export PYTHONIOENCODING=utf-8    # logs con unicode robustos en cualquier consola
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # menos OOM por fragmentación (vocabs grandes)

MODE="${1:-full}"
REMOTE="${2:-}"
SMOKE_ARGS=""
# --no-repair en smoke: el mixing añade los 54k de repair aunque --subset recorte la generación
# → el smoke mapearía 54k. El mixing ya está validado; el smoke solo confirma carga+train+export.
[ "$MODE" = "smoke" ] && SMOKE_ARGS="--subset 32 --max-steps 3 --batch 2 --no-repair"

BRAIN="${BRAIN:-/workspace/nodex-brain}"
CODE="${CODE:-/workspace/nodex-code}"
LLAMA="${LLAMA:-/workspace/llama.cpp}"
OUT="$BRAIN/gguf"
cd "$BRAIN"; mkdir -p "$OUT" logs
log(){ echo "[$(date +%H:%M:%S)] $*"; }
FAILS=()
run(){ local tag="$1"; shift; log "▶ $tag"; if "$@" > "logs/$tag.log" 2>&1; then log "  ✓ $tag"; else log "  ✗ $tag (ver logs/$tag.log)"; FAILS+=("$tag"); fi; }

# --- 0. setup idempotente -----------------------------------------------------
setup(){
  python -c "import torch,transformers,peft,datasets,trl,gguf,sentencepiece" 2>/dev/null || \
    pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -q transformers trl peft datasets accelerate tokenizers bitsandbytes numpy gguf sentencepiece protobuf "huggingface_hub[cli]"
  # datos (viven en nodex-code) — falla temprano si faltan
  for f in generator-combined-cot-sft-40185/train.jsonl generator-combined-cot-sft-40185/val.jsonl repair-sft/train.jsonl; do
    [ -f "$CODE/datasets/$f" ] || { log "FALTA dataset: $CODE/datasets/$f"; exit 1; }
  done
  [ -f "$BRAIN/tokenizer/ndx/tokenizer.json" ] || { log "FALTA tokenizer del nano: $BRAIN/tokenizer/ndx"; exit 1; }
  # llama.cpp (converter + quantize)
  [ -d "$LLAMA" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
  if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
    cmake -S "$LLAMA" -B "$LLAMA/build" -DLLAMA_CURL=OFF >/dev/null && cmake --build "$LLAMA/build" --target llama-quantize -j
  fi
}

# --- export HF dir → gguf f16 + Q4 --------------------------------------------
export_nano(){   # tokenizer custom → usa el script con parches (ya nombra ndx-coder-nano-215m-*)
  bash scripts/export_gguf.sh models/ndx-coder-nano-215m "$OUT"
}
# Delegamos en export_one.sh en vez de duplicar los parches acá: una sola
# implementación de la reparación (scripts/fix_hf_export.py) y de la validación del
# gguf. La versión anterior tenía sus propios fix_hf_tokenizer/fix_hf_config que
# adivinaban la ruta del cache de HF con un glob y, si no la encontraban, retornaban
# EN SILENCIO — así granite salió dos veces con head_count_kv=[0]*28 y 0 tensores ssm
# (no carga) sin que el log dijera nada.
export_pretrained(){  # $1=model_dir $2=base_id $3=nombre-familia
  local dir="$1" base="$2" name="$3"
  [ -d "$dir" ] || { log "  (no existe $dir, skip export)"; return 1; }
  BRAIN="$BRAIN" LLAMA="$LLAMA" OUT="$OUT" bash scripts/export_one.sh "$dir" "$base" "$name"
}
export_qlora(){  # $1=adapter_dir $2=config $3=base_id $4=name  (QLoRA → merge → gguf)
  local adir="$1" cfg="$2" base="$3" name="$4"
  [ -d "$adir" ] || { log "  (no existe $adir, skip)"; return 1; }
  python -m src.merge --config "$cfg" --adapter "$adir" --out "${adir}-merged" --device cpu || return 1
  export_pretrained "${adir}-merged" "$base" "$name"; local rc=$?
  rm -rf "${adir}-merged"    # merged intermedio (regenerable; el f16 GGUF es el respaldo). Bases NO se borran.
  return $rc
}

# =============================================================================
setup
log "===== MODO: $MODE ====="

# --- PRODUCCIÓN: familia ndx-coder (con repair mixing) — nano PRIMERO ---------
# Todo via src.train_sft: enmascara el prompt (loss SOLO en el assistant). NO train_scratch:
# TRL entrena la secuencia completa + packing → con repair mezclado el modelo aprende a
# emitir el texto de USUARIO ("El compilador rechazó el deck…") en vez del deck.
# Este script NO corre experimentos: viven en src/experiments/ y se lanzan aparte
# (ver EXPERIMENTS.md). Producción no debe depender de que un experimento termine.
run prod-nano          python -m src.train_sft --config configs/coder_nano.yaml          $SMOKE_ARGS
run prod-qwen3-0.6b    python -m src.train_sft --config configs/coder_qwen3_06b.yaml     $SMOKE_ARGS
run prod-gemma4-e2b    python -m src.train_sft --config configs/coder_gemma4_e2b.yaml    $SMOKE_ARGS
run prod-gemma4-e4b    python -m src.train_sft --config configs/coder_gemma4_e4b.yaml    $SMOKE_ARGS

# --- EXPORT GGUF -------------------------------------------------------------
# En smoke exportamos SOLO el nano (valida el path completo sin gastar en 6 exports).
log "===== export GGUF ====="
run export-nano export_nano
if [ "$MODE" = "full" ]; then
  run export-qwen3    export_pretrained models/ndx-coder-qwen3-0.6b   Qwen/Qwen3-0.6B-Base      ndx-coder-qwen3-0.6b
  run export-gemma4e2b export_qlora models/ndx-coder-gemma4-e2b configs/coder_gemma4_e2b.yaml google/gemma-4-E2B-it ndx-coder-gemma4-e2b
  run export-gemma4e4b export_qlora models/ndx-coder-gemma4-e4b configs/coder_gemma4_e4b.yaml google/gemma-4-E4B-it ndx-coder-gemma4-e4b
fi

# --- entregar el nano repair-trained a nodex-code (para el A/B de code) -------
[ -f "$OUT/ndx-coder-nano-215m-Q4_K_M.gguf" ] && mkdir -p "$CODE/packages/coder/model" && \
  cp -f "$OUT/ndx-coder-nano-215m-Q4_K_M.gguf" "$CODE/packages/coder/model/" && log "nano → nodex-code/packages/coder/model/"

# --- UPLOAD (rclone: Dropbox o Google Drive) — TODO, incluido el f16 ----------
if [ -n "$REMOTE" ] && [ "$MODE" = "full" ]; then
  command -v rclone >/dev/null || curl https://rclone.org/install.sh | bash
  log "===== upload a $REMOTE (gguf f16+Q4) ====="
  run upload-gguf rclone copy "$OUT" "$REMOTE/gguf" --transfers 4    # f16+Q4 = 'todo incluido el f16'
  # los HF models/ son grandes y redundantes con el f16 → opt-in: UPLOAD_MODELS=1 bash run_all.sh full ...
  [ "${UPLOAD_MODELS:-0}" = "1" ] && run upload-models rclone copy "$BRAIN/models" "$REMOTE/models" --transfers 4 --exclude "checkpoint-*/**"
elif [ -n "$REMOTE" ]; then
  log "smoke → NO subo (el upload es solo para el full)"
else
  log "sin REMOTE → no subo."
fi

# --- resumen -----------------------------------------------------------------
echo; log "===== FIN ($MODE) ====="
if [ ${#FAILS[@]} -eq 0 ]; then log "TODO OK ✅"; else log "FALLARON (${#FAILS[@]}): ${FAILS[*]}  → revisa logs/"; exit 1; fi
