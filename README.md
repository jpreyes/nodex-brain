# Nodex Brain

Capa LLM de Nodex: modelos y fine-tuning con QLoRA.

## Objetivos

1. **Modelos** — crear y servir los modelos LLM que utiliza Nodex.
2. **Fine-tuning (QLoRA)** — entrenar adapters QLoRA sobre datasets SFT para especializar los modelos en las tareas de Nodex.

## Dominios

Los datasets de entrenamiento cubren, entre otros: análisis estructural (general e industrial), sísmico, geotécnico, mecánico, hormigón armado, puentes y generación de DSL/grid.

## Estructura

```
nodex-brain/
├── configs/                 # configs YAML con herencia vía `extends`
│   ├── _common.yaml             # LoRA + hiperparámetros (idénticos entre modelos)
│   ├── base_mistral.yaml        # base: Mistral Nemo 12B (Apache-2.0)
│   ├── base_gemma.yaml          # base: Gemma 3 12B (Gemma Terms)
│   └── structural_*.yaml        # runs por (dataset × modelo)
├── src/                     # código
│   ├── config.py                # carga de YAML con `extends`
│   ├── data.py                  # carga de JSONL → chat template
│   ├── train.py                 # entrenamiento QLoRA (peft + trl)
│   ├── eval.py                  # loss/perplejidad en validación
│   ├── compare.py               # generaciones A/B lado a lado
│   ├── merge.py                 # merge del adapter al base
│   └── infer.py                 # inferencia
├── scripts/                 # orquestación (ab_compare.ps1 / .sh)
├── datasets/                # datos SFT (no versionado)
├── adapters/ | models/      # salidas (no versionado)
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Gemma 3 es un repo *gated* en Hugging Face: acepta la licencia en su página y
> ejecuta `huggingface-cli login` antes de descargarlo.

## Uso

Entrenar un adapter QLoRA:

```bash
python -m src.train --config configs/structural_mistral.yaml
```

Fusionar el adapter al modelo base:

```bash
python -m src.merge --config configs/structural_mistral.yaml \
    --adapter adapters/structural-mistral --out models/structural-mistral-merged
```

Inferencia:

```bash
python -m src.infer --config configs/structural_mistral.yaml \
    --adapter adapters/structural-mistral --prompt "..."
```

## Comparación A/B (Mistral vs Gemma)

Entrena el mismo dataset con dos bases y compáralos. Las configs comparten
`_common.yaml` (LoRA + hiperparámetros idénticos) para que la comparación sea justa:

```bash
bash scripts/ab_compare.sh          # Windows: .\scripts\ab_compare.ps1
```

Esto entrena ambos, imprime loss/perplejidad por modelo y genera
`compare_structural.md` con las salidas lado a lado para revisión humana.

> La perplejidad **no** es directamente comparable entre modelos con tokenizers
> distintos; úsala como proxy de progreso. La decisión "cuál es mejor" apóyala en
> `compare.py` (revisión humana) y/o una métrica a nivel de tarea.

## NDX-Coder (modelo pequeño desde cero)

Modelo ~215M **entrenado desde cero** (init aleatorio, tokenizer propio) que traduce
`español → código NDX` con scratchpad `<think>`. Objetivo: correr en CPU+RAM (≤300 MB
tras exportar a GGUF). Métrica dura: **% de decks que compilan** (`src/compiler.py`).

```bash
# 1) Cosechar corpus (tokenizer + decks) desde el dataset con scratchpad
python -m src.harvest_ndx \
    --datasets ../nodex-code/datasets/generator-combined-cot-sft-40185/train.jsonl \
    --out-corpus corpus/tokenizer.txt --out-decks corpus/ndx_decks.txt

# 2) Entrenar el tokenizer BPE byte-level propio (~6-16k, con <think>/</think>)
python -m src.tokenizer_train --corpus corpus/tokenizer.txt --out tokenizer/ndx

# 3) Entrenar el modelo DESDE CERO (GPU) — full fine-tune, sin QLoRA
python -m src.train_scratch --config configs/ndx_coder.yaml

# 4) Evaluar por compilación (genera → extrae deck tras </think> → compila)
python -m src.eval_coder --config configs/ndx_coder.yaml --model models/ndx-coder-small --n 200
```

Pasos 1–2 corren en CPU (local); 3–4 necesitan GPU. Config en
[`configs/ndx_coder.yaml`](configs/ndx_coder.yaml).

## Verificador (nodex-compiler)

`src/compiler.py` valida código NDX compilándolo + resolviéndolo con el
`nodex-compiler` (repo hermano en `../nodex-compiler`). Es la base del eval por
compilación y del loop de auto-mejora. Requiere Node y el WASM de Nodex.

```bash
# resolución del CLI:  NODEXC_CLI  →  ../nodex-compiler/src/cli.js  →  nodexc en PATH
python -m src.compiler deck.ndx        # {"ok": true/false, "error": ...}
```

## Notas

- Los datasets, modelos, checkpoints y adapters no se versionan (ver `.gitignore`).
- El modelo base se ajusta en `configs/base_mistral.yaml` / `configs/base_gemma.yaml`
  (`model.name_or_path`). Para más VRAM, sube a Mistral Small 24B / Gemma 3 27B.

## Licencia

Código bajo [Apache-2.0](LICENSE). Los pesos que produzcas heredan la licencia de
su modelo base (Mistral → Apache-2.0; Gemma → Gemma Terms of Use, con sus avisos
y convención de nombres al distribuir).
