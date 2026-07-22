#!/usr/bin/env bash
# =============================================================================
# run_2a.sh — el 2x2 del experimento 2A: 4 configuraciones x 3 semillas = 12 corridas.
#
#   bash scripts/smoke.sh          # PRIMERO. Si no da 5/5, no lances esto.
#
#   # LANZAR ASÍ, no con nohup a secas: `nohup` deja el proceso atado al terminal y una
#   # caída de sesión lo mata igual. Perdimos 12 h así, con una corrida muerta al 80%.
#   setsid nohup bash scripts/run_2a.sh > logs/2a/run.out 2>&1 < /dev/null &
#   disown
#
#   # Y NO uses `tail -f` si tu consola no permite Ctrl-C (Jupyter de vast): quedas
#   # atrapado, acabas usando Ctrl-Z, y al cerrar la pestaña bash manda SIGHUP a los jobs
#   # suspendidos. Usa consultas puntuales:
#   tail -5 logs/2a/run.out
#
# Propiedades que un `for` suelto no da:
#   · REANUDABLE en DOS niveles — salta las corridas terminadas (model.safetensors +
#     tsd_config), y dentro de una corrida interrumpida retoma desde su último
#     checkpoint-N en vez de repetir desde cero.
#   · RESPALDO INCREMENTAL a la nube tras cada corrida (RCLONE_REMOTE): si la instancia
#     desaparece —pasa— lo perdido es como mucho la corrida en curso, no las 9 anteriores.
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
# Remote de rclone donde respaldar cada corrida al terminar. Vacío = no respalda.
#   RCLONE_REMOTE=db:nodex-models setsid nohup bash scripts/run_2a.sh ...
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
FAILS=(); HECHAS=0; SALTADAS=0

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Sube una corrida terminada. Se llama tras CADA corrida, no al final: una instancia que
# desaparece a mitad del 2x2 no debe costar más que la corrida en curso.
# Se excluyen los checkpoint-*: son estado intermedio del optimizador, pesan mucho y el
# modelo final ya está en model.safetensors.
respaldar(){
  local out="$1"
  [ -z "$RCLONE_REMOTE" ] && return 0
  command -v rclone >/dev/null || { log "  ! rclone no está: no respaldo"; return 0; }
  if rclone copy "$out" "$RCLONE_REMOTE/$(basename "$out")" \
       --exclude "checkpoint-*/**" --transfers 4 >/dev/null 2>&1; then
    log "  ↑ respaldado en $RCLONE_REMOTE/$(basename "$out")"
  else
    log "  ! falló el respaldo de $(basename "$out") — sigo (la corrida está en disco)"
  fi
}

# Último checkpoint-N de una corrida interrumpida, para retomar desde ahí.
ultimo_checkpoint(){
  ls -d "$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1
}

# Espera a que la GPU tenga NEED MiB libres antes de lanzar la siguiente corrida.
# EL SCRIPT ES SECUENCIAL, pero `python -m` devuelve el control al shell cuando main()
# termina, y el proceso tarda en SOLTAR la memoria de GPU en el teardown (más aún con
# dataloader workers). Sin esta espera, la corrida N+1 arranca sobre los ~24 GB que la N
# todavía no liberó: en el nano da igual (dos caben), pero qwen3 necesita ~24 de 32 GB y
# la segunda muere por OOM. Ese fue el fallo que perdió 4 corridas de qwen3.
esperar_gpu(){
  local need="${1:-26000}" i free
  command -v nvidia-smi >/dev/null || return 0          # local sin GPU: no bloquea
  for i in $(seq 1 100); do                              # ~5 min de techo
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$free" ] && return 0
    [ "$free" -ge "$need" ] && return 0
    [ $((i % 5)) -eq 1 ] && log "  … GPU: ${free} MiB libres, espero ${need} (la corrida anterior aún libera)"
    sleep 3
  done
  log "  ! la GPU no se liberó tras 5 min — lanzo igual, puede dar OOM"
}

# $1=etiqueta $2=config $3=output_dir esperado $4...=flags extra
correr(){
  local tag="$1" cfg="$2" out="$3"; shift 3
  if [ -f "$out/model.safetensors" ] && [ -f "$out/tsd_config.json" ]; then
    log "· $tag ya está — salto"; SALTADAS=$((SALTADAS+1)); return 0
  fi
  # Corrida interrumpida con checkpoints dentro: retomar en vez de repetir desde cero.
  # La que murió al 80% dejó checkpoint-1500 y se rehízo entera — 1h de GPU tirada.
  local ck; ck=$(ultimo_checkpoint "$out")
  if [ -n "$ck" ]; then
    log "· $tag tiene checkpoint-$ck — retomo desde ahí"
    set -- "$@" --resume
  fi
  esperar_gpu 26000                    # qwen3 necesita ~24 GB; no arrancar sobre restos
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
    respaldar "$out"          # tras CADA corrida, no al final
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
