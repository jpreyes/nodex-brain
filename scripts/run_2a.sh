#!/usr/bin/env bash
# =============================================================================
# run_2a.sh — el 2x2 del experimento 2A: 4 configuraciones x 3 semillas = 12 corridas.
#
#   bash scripts/smoke.sh          # PRIMERO. Si no da 5/5, no lances esto.
#   nohup bash scripts/run_2a.sh > logs/2a/nohup.out 2>&1 &
#
# Propiedades que un `for` suelto no da:
#   · REANUDABLE — salta las corridas ya terminadas (mira model.safetensors + tsd_config).
#     Si se cae la caja o matas el proceso, relanzas y sigue donde iba.
#   · un fallo NO aborta el resto: se acumulan y se reportan al final.
#   · un log por corrida en logs/2a/, y verificación de `frac in_deck` en los brazos +TSD.
#
# Entrena sobre el SPLIT CONGELADO (exp_*.yaml). Los coder_*.yaml son producción y cargan
# el corpus completo, que contiene Test A y Test B.
# =============================================================================
set -uo pipefail
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BRAIN="${BRAIN:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$BRAIN"
LOG="logs/2a"; mkdir -p "$LOG"
SEEDS="${SEEDS:-42 1337 7}"
E="src.experiments.train_tsd"
A="--ablation --no-repair"
FAILS=(); HECHAS=0; SALTADAS=0

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# $1=etiqueta $2=config $3=output_dir esperado $4...=flags extra
correr(){
  local tag="$1" cfg="$2" out="$3"; shift 3
  if [ -f "$out/model.safetensors" ] && [ -f "$out/tsd_config.json" ]; then
    log "· $tag ya está — salto"; SALTADAS=$((SALTADAS+1)); return 0
  fi
  log "▶ $tag"
  python -m "$E" --config "$cfg" $A "$@" > "$LOG/$tag.log" 2>&1
  local rc=$?
  # El VEREDICTO son los artefactos, no el exit code. Medido: qwen3 completaba el
  # entrenamiento, guardaba el modelo y el tsd_config, imprimía su última línea... y salía
  # con código != 0 al apagar el intérprete (teardown del dataloader / CUDA). Fiarse solo
  # del código marcaba como fallidas 6 corridas de 1h30 que estaban perfectas.
  if [ -f "$out/model.safetensors" ] && [ -f "$out/tsd_config.json" ]; then
    [ $rc -ne 0 ] && log "  ! $tag — exit $rc pero los artefactos están completos: se da por buena"
    rc=0
  fi
  if [ $rc -eq 0 ]; then
    local frac
    frac=$(grep -oE "frac in_deck=[0-9.]+" "$LOG/$tag.log" | head -1 | cut -d= -f2)
    # 0 aborta solo; ~0.87 significa que el corte del CoT se rompió y la corrida NO vale
    if [ -n "$frac" ] && ! awk -v f="$frac" 'BEGIN{exit !(f>0.20 && f<0.75)}'; then
      log "  ✗ $tag — frac in_deck=$frac fuera de rango, esta corrida NO sirve"
      FAILS+=("$tag (frac=$frac)"); return 1
    fi
    log "  ✓ $tag${frac:+  (frac in_deck=$frac)}"
    HECHAS=$((HECHAS+1))
  else
    log "  ✗ $tag — ver $LOG/$tag.log"; tail -5 "$LOG/$tag.log" | sed 's/^/      /'
    FAILS+=("$tag")
  fi
}

log "===== 2A · semillas: $SEEDS ====="
for s in $SEEDS; do
  correr "nano-base-s$s"     configs/exp_nano.yaml      "models/exp-nano-215m-base-s$s"        --seed "$s"
  correr "nano-tsd-s$s"      configs/exp_nano.yaml      "models/exp-nano-215m-tsd-ast-s$s"     --seed "$s" --tsd --tree ast
  correr "qwen3-base-s$s"    configs/exp_qwen3_06b.yaml "models/exp-qwen3-0.6b-base-s$s"       --seed "$s"
  correr "qwen3-tsd-s$s"     configs/exp_qwen3_06b.yaml "models/exp-qwen3-0.6b-tsd-ast-s$s"    --seed "$s" --tsd --tree ast
done

echo; log "===== FIN ====="
log "hechas: $HECHAS · saltadas (ya estaban): $SALTADAS · fallidas: ${#FAILS[@]}"
[ ${#FAILS[@]} -gt 0 ] && { printf '  - %s\n' "${FAILS[@]}"; echo "  logs en $LOG/"; exit 1; }
echo
log "checkpoints:"; ls -d models/exp-*-s* 2>/dev/null | sed 's/^/  /'
echo
log "siguiente: NO mires la loss de validación. La métrica es eval_tail sobre Test A."
log "  python -m src.experiments.eval_tail medir --pairs <generados.jsonl>"
