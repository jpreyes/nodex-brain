#!/usr/bin/env bash
# A/B: entrena Mistral y Gemma en el MISMO dataset y los compara.
# Uso:  bash scripts/ab_compare.sh
set -euo pipefail

# 1) Entrenar ambos
python -m src.train --config configs/structural_mistral.yaml
python -m src.train --config configs/structural_gemma.yaml

# 2) Loss/perplejidad por modelo (proxy; no comparable entre tokenizers)
python -m src.eval --config configs/structural_mistral.yaml --adapter adapters/structural-mistral --limit 200
python -m src.eval --config configs/structural_gemma.yaml   --adapter adapters/structural-gemma   --limit 200

# 3) Comparación lado a lado para revisión humana
python -m src.compare \
  --config-a configs/structural_mistral.yaml --adapter-a adapters/structural-mistral --name-a Mistral \
  --config-b configs/structural_gemma.yaml   --adapter-b adapters/structural-gemma   --name-b Gemma \
  --prompts-from datasets/structural-sft-1000/val.jsonl --n 10 --out compare_structural.md

echo "Listo. Revisa compare_structural.md"
