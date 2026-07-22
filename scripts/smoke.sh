#!/usr/bin/env bash
# =============================================================================
# smoke.sh — valida de punta a punta los DOS entrenadores antes de una corrida larga.
#
#   bash scripts/smoke.sh
#
# Por qué existe: el refactor producción/experimento solo pasó `py_compile`. Un
# entrenamiento real son horas; este smoke son minutos y cubre los modos de fallo
# SILENCIOSOS — los que no dan error y solo se descubren comparando pesos después:
#
#   · los paths del árbol no llegan al collator (remove_unused_columns) -> bias = 0
#   · el corte del CoT deja de aplicarse -> el bias vuelve a cubrir el scratchpad
#   · tsd_config.json no se escribe -> la inferencia no sabe con qué bias entrenó
#   · producción arrastra la máscara 4D del experimento
#
# Falla RUIDOSAMENTE: cualquier invariante rota corta el script con exit != 0.
# =============================================================================
set -uo pipefail
export PYTHONIOENCODING=utf-8

BRAIN="${BRAIN:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$BRAIN"
TINY="--subset 64 --max-steps 3 --batch 2 --no-repair"

# El smoke escribe en su PROPIO config, no en el del experimento. Antes usaba
# exp_nano.yaml con --ablation, o sea models/exp-nano-215m-{base,tsd-ast}-s42: las MISMAS
# carpetas del 2x2. Un smoke de 3 pasos las pisaba con un modelo que genera basura, y
# run_2a las daba por completas (tienen model.safetensors + tsd_config) y las saltaba.
# Se descubrio al generar: recall 0.000 y predicciones de puras comas.
SMOKE_CFG=configs/_smoke_nano.yaml
sed 's|output_dir: models/exp-nano-215m|output_dir: models/_smoke-nano|'     configs/exp_nano.yaml > "$SMOKE_CFG"
LOG="logs/smoke"; mkdir -p "$LOG"
FAILS=()

ok(){ echo "  ✓ $*"; }
bad(){ echo "  ✗ $*"; FAILS+=("$*"); }
hr(){ echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

# --- 1. PRODUCCIÓN ------------------------------------------------------------
# Producción usa el corpus COMPLETO (coder_*.yaml), a propósito: más datos = mejor modelo,
# y no se evalúa contra Test A/B. Los experimentos usan exp_*.yaml, que apuntan al split
# congelado — entrenar el 2x2 con el corpus completo invalidaría los dos tests.
hr "[1] producción — src.train_sft"
if python -m src.train_sft --config configs/coder_nano.yaml $TINY > "$LOG/prod.log" 2>&1; then
  ok "corrió sin error"
  # producción NO debe usar eager: es el costo que el split vino a quitar (§3.0)
  grep -q "from-scratch" "$LOG/prod.log" && ok "modelo construido" || bad "no reporta el modelo"
else
  bad "src.train_sft falló — ver $LOG/prod.log"; tail -15 "$LOG/prod.log"
fi

# --- 2. EXPERIMENTO base ------------------------------------------------------
hr "[2] experimento — brazo base (sin bias)"
python -m src.experiments.train_tsd --config "$SMOKE_CFG" --ablation $TINY \
  > "$LOG/exp_base.log" 2>&1 \
  && ok "corrió sin error" || { bad "brazo base falló"; tail -15 "$LOG/exp_base.log"; }

# --- 3. EXPERIMENTO +TSD con el árbol AST ------------------------------------
hr "[3] experimento — brazo +TSD, árbol ast (K=3)"
if python -m src.experiments.train_tsd --config "$SMOKE_CFG" --ablation --tsd \
     --tree ast $TINY > "$LOG/exp_tsd.log" 2>&1; then
  ok "corrió sin error"
  FRAC=$(grep -oE "frac in_deck=[0-9.]+" "$LOG/exp_tsd.log" | head -1 | cut -d= -f2)
  if [ -z "$FRAC" ]; then
    bad "no imprimió frac in_deck (la guarda no se ejecutó)"
  else
    echo "     frac in_deck = $FRAC"
    # 0 -> el bias sería idénticamente cero (el brazo TSD no lo sería)
    # ~0.87 -> el corte del CoT dejó de aplicarse y el bias cubre el scratchpad
    awk -v f="$FRAC" 'BEGIN{exit !(f>0.20 && f<0.75)}' \
      && ok "frac in_deck en rango (esperado ~0.50)" \
      || bad "frac in_deck=$FRAC fuera de rango: 0 = bias nulo; ~0.87 = cubre el CoT"
  fi
else
  bad "brazo +TSD falló"; tail -15 "$LOG/exp_tsd.log"
fi

# --- 4. simetría train/infer --------------------------------------------------
hr "[4] tsd_config.json — el puente train/infer"
CFG=models/_smoke-nano-tsd-ast-s42/tsd_config.json
if [ -f "$CFG" ]; then
  cat "$CFG" | sed 's/^/     /'
  python - "$CFG" <<'PY' && ok "config coherente con lo entrenado" || bad "tsd_config.json incoherente"
import json,sys
c=json.load(open(sys.argv[1],encoding="utf-8"))
assert c["use_tsd"] is True, c
assert c["tree"]=="ast" and c["K"]==3, c
assert c["norm"] is True, c
PY
else
  bad "no se escribió $CFG — la inferencia no sabría con qué bias entrenó (§6.5)"
fi

# --- 5. el verificador del bias ----------------------------------------------
hr "[5] verify_tsd_bias --tree ast"
python -m src.experiments.verify_tsd_bias --config "$SMOKE_CFG" --tree ast \
  --skip-weights > "$LOG/verify.log" 2>&1 \
  && ok "4/4 PASS" || { bad "el verificador falló"; tail -25 "$LOG/verify.log"; }

# --- resumen ------------------------------------------------------------------
hr "RESUMEN"
if [ ${#FAILS[@]} -eq 0 ]; then
  rm -rf models/_smoke-nano* "$SMOKE_CFG"     # sin residuos
  echo "  TODO OK — los dos entrenadores corren de punta a punta."
  echo "  (esto NO valida la calidad del entrenamiento, solo que el pipeline no está roto)"
else
  echo "  FALLARON (${#FAILS[@]}):"; printf '    - %s\n' "${FAILS[@]}"
  echo "  logs en $LOG/"
  exit 1
fi
