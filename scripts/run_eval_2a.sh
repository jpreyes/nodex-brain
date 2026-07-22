#!/usr/bin/env bash
# =============================================================================
# run_eval_2a.sh — genera sobre Test A con los 12 checkpoints y mide la cola.
#
#   setsid nohup bash scripts/run_eval_2a.sh > logs/eval/run.out 2>&1 < /dev/null &
#   disown
#
#   # y para mirar (NO uses tail -f si tu consola no permite Ctrl-C):
#   tail -5 logs/eval/run.out
#   ls eval/*.jsonl | wc -l
#
# Orden deliberado: primero los 6 BASE, que son rápidos (model.generate con KV-cache
# nativo). Si la instancia se cae, ya tienes la mitad del 2x2 medida en vez de nada.
# Los +TSD van después: ~46 min cada uno (el cuello es el bias en CPU, no la GPU).
#
# Todo reanudable: gen_testa reanuda por EJEMPLO (append al .jsonl), este script salta los
# modelos ya completos. Un corte cuesta como mucho la generación en curso.
# =============================================================================
set -uo pipefail
export PYTHONIOENCODING=utf-8

BRAIN="${BRAIN:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$BRAIN"
LOG="logs/eval"; mkdir -p "$LOG" eval
CABEZA="${CABEZA:-700}"     # estratificado completo. La cola va ENTERA siempre.
FALLOS=()

log(){ echo "[$(date +%H:%M:%S)] $*"; }

generar(){
  local m="$1" name; name=$(basename "$m")
  [ -d "$m" ] || { log "· $name no existe — salto"; return 0; }
  log "▶ $name"
  if python -m src.experiments.gen_testa --model "$m" --cabeza "$CABEZA" \
       > "$LOG/$name.log" 2>&1; then
    log "  ✓ $name  ($(wc -l < "eval/$name.jsonl") pares)"
  else
    log "  ✗ $name — ver $LOG/$name.log"; tail -5 "$LOG/$name.log" | sed 's/^/      /'
    FALLOS+=("$name")
  fi
}

log "===== generación sobre Test A (cola completa + $CABEZA de cabeza) ====="

# 1) los BASE primero: baratos, y aseguran media tabla si algo se cae
for s in 42 1337 7; do
  generar "models/exp-nano-215m-base-s$s"
  generar "models/exp-qwen3-0.6b-base-s$s"
done

# 2) los +TSD: el bias se reconstruye por token, así que son los caros
for s in 42 1337 7; do
  generar "models/exp-nano-215m-tsd-ast-s$s"
  generar "models/exp-qwen3-0.6b-tsd-ast-s$s"
done

# --- medición ------------------------------------------------------------------
log "===== recall de cola por modelo ====="
PRE=eval_tail_prereg.json
if [ ! -f "$PRE" ]; then
  log "preregistro el corte cabeza/cola (una sola vez, ANTES de mirar resultados)"
  python -m src.experiments.eval_tail preregistrar \
    --train ../nodex-code/datasets/frozen-split-v2/train.jsonl --cut 1000 > "$LOG/prereg.log" 2>&1
fi

for f in eval/*.jsonl; do
  echo; echo "########## $(basename "$f" .jsonl)"
  python -m src.experiments.eval_tail medir --pairs "$f" --prereg "$PRE" 2>&1 | sed -n '1,12p'
done | tee "$LOG/recall.txt"

echo
log "===== FIN ====="
[ ${#FALLOS[@]} -gt 0 ] && { log "fallaron: ${FALLOS[*]}"; exit 1; }
log "tabla completa en $LOG/recall.txt"
log "el número que decide es el recall AGREGADO de cola, no el desglose por kind (§13.5/E9)"
