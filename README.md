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

## Notas

- Los datasets, modelos, checkpoints y adapters no se versionan (ver `.gitignore`).
- El modelo base se ajusta en `configs/base_mistral.yaml` / `configs/base_gemma.yaml`
  (`model.name_or_path`). Para más VRAM, sube a Mistral Small 24B / Gemma 3 27B.

## Licencia

Código bajo [Apache-2.0](LICENSE). Los pesos que produzcas heredan la licencia de
su modelo base (Mistral → Apache-2.0; Gemma → Gemma Terms of Use, con sus avisos
y convención de nombres al distribuir).
