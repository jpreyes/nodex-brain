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
├── configs/            # configs de entrenamiento (YAML con herencia vía `extends`)
│   ├── base_model.yaml     # modelo base, cuantización 4-bit, LoRA, hiperparámetros
│   └── qlora_*.yaml        # config por dominio
├── src/                # código
│   ├── config.py           # carga de YAML con `extends`
│   ├── data.py             # carga de JSONL → chat template
│   ├── train.py            # entrenamiento QLoRA (peft + trl)
│   ├── merge.py            # merge del adapter al base
│   └── infer.py            # inferencia
├── datasets/           # datos SFT (no versionado)
├── adapters/ | models/ # salidas (no versionado)
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Entrenar un adapter QLoRA:

```bash
python -m src.train --config configs/qlora_structural.yaml
```

Fusionar el adapter al modelo base:

```bash
python -m src.merge --config configs/qlora_structural.yaml \
    --adapter adapters/structural --out models/structural-merged
```

Inferencia:

```bash
python -m src.infer --config configs/qlora_structural.yaml \
    --adapter adapters/structural --prompt "..."
```

## Notas

- Los datasets, modelos, checkpoints y adapters no se versionan en este repositorio (ver `.gitignore`).
- Ajusta el modelo base en `configs/base_model.yaml` (`model.name_or_path`).

## Licencia

[Apache-2.0](LICENSE).
