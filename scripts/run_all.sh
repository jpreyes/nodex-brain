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

# --- tokenizer fix para pretrained (quirk de transformers 5 al re-guardar) ----
# transformers 5 guarda tokenizer_class="TokenizersBackend" y a veces pierde vocab/merges
# → el converter no lo reconoce. Restauramos el tokenizer PRISTINO del cache de HF.
fix_hf_tokenizer(){  # $1=model_dir  $2=base_id (org/name)
  local dir="$1" base="$2"
  local snap; snap=$(ls -d ~/.cache/huggingface/hub/models--${base//\//--}/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$snap" ] || { log "  (sin cache de $base para fix de tokenizer; intento directo)"; return 0; }
  for t in tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt tokenizer.model; do
    [ -f "$snap/$t" ] && cp -f "$snap/$t" "$dir/$t"
  done
}

# --- export HF dir → gguf f16 + Q4 --------------------------------------------
export_nano(){   # tokenizer custom → usa el script con parches
  bash scripts/export_gguf.sh models/ndx-coder-small "$OUT" && \
    mv -f "$OUT/ndx-coder-small-f16.gguf"    "$OUT/ndx-coder-nano-215m-f16.gguf" 2>/dev/null; \
    mv -f "$OUT/ndx-coder-small-Q4_K_M.gguf" "$OUT/ndx-coder-nano-215m-Q4_K_M.gguf" 2>/dev/null; true
}
export_pretrained(){  # $1=model_dir $2=base_id $3=nombre-familia
  local dir="$1" base="$2" name="$3"
  [ -d "$dir" ] || { log "  (no existe $dir, skip export)"; return 1; }
  fix_hf_tokenizer "$dir" "$base"
  python "$LLAMA/convert_hf_to_gguf.py" "$dir" --outfile "$OUT/$name-f16.gguf" --outtype f16 && \
    "$LLAMA/build/bin/llama-quantize" "$OUT/$name-f16.gguf" "$OUT/$name-Q4_K_M.gguf" Q4_K_M
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

# --- TRACK B: familia de PRODUCCIÓN (con repair mixing) — nano PRIMERO --------
run prod-nano          python -m src.train_scratch --config configs/ndx_coder.yaml           $SMOKE_ARGS
run prod-gemma3-270m   python -m src.train_tsd     --config configs/coder_gemma3_270m.yaml   $SMOKE_ARGS
run prod-granite-350m  python -m src.train_tsd     --config configs/coder_granite_350m.yaml  $SMOKE_ARGS
run prod-qwen3-0.6b    python -m src.train_tsd     --config configs/coder_qwen3_06b.yaml     $SMOKE_ARGS
run prod-gemma4-e2b    python -m src.train_tsd     --config configs/coder_gemma4_e2b.yaml    $SMOKE_ARGS
run prod-gemma4-e4b    python -m src.train_tsd     --config configs/coder_gemma4_e4b.yaml    $SMOKE_ARGS

# --- TRACK A: ablación TSD 2x2 (pura, sin repair) ----------------------------
run tsd-nano-base      python -m src.train_tsd --config configs/coder_nano.yaml        --no-repair       $SMOKE_ARGS
run tsd-nano-tsd       python -m src.train_tsd --config configs/coder_nano.yaml        --no-repair --tsd $SMOKE_ARGS
run tsd-gemma3-base    python -m src.train_tsd --config configs/coder_gemma3_270m.yaml --no-repair       $SMOKE_ARGS
run tsd-gemma3-tsd     python -m src.train_tsd --config configs/coder_gemma3_270m.yaml --no-repair --tsd $SMOKE_ARGS

# --- EXPORT GGUF -------------------------------------------------------------
# En smoke exportamos SOLO el nano (valida el path completo sin gastar en 6 exports).
log "===== export GGUF ====="
run export-nano export_nano
if [ "$MODE" = "full" ]; then
  run export-gemma3   export_pretrained models/ndx-coder-gemma3-270m  google/gemma-3-270m       ndx-coder-gemma3-270m
  run export-granite  export_pretrained models/ndx-coder-granite-350m ibm-granite/granite-4.0-350m ndx-coder-granite-350m
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
