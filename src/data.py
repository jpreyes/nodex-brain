"""Carga y formateo de datasets SFT (JSONL) al chat template del tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset


def load_splits(cfg: dict[str, Any]):
    """Carga los splits train/val declarados en el config.

    Espera `cfg["data"]["dataset_dir"]` y los nombres de archivo
    `train_split` / `val_split` (JSONL).
    """
    data_cfg = cfg["data"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    data_files = {
        "train": str(dataset_dir / data_cfg["train_split"]),
        "validation": str(dataset_dir / data_cfg["val_split"]),
    }
    return load_dataset("json", data_files=data_files)


def build_formatter(tokenizer, cfg: dict[str, Any]):
    """Devuelve una función que convierte un ejemplo en texto listo para SFT.

    Soporta dos formatos de entrada comunes en los .jsonl:
      * {"messages": [{"role": ..., "content": ...}, ...]}
      * {"instruction": ..., "input": ..., "output": ...}
    """

    def format_example(example: dict[str, Any]) -> str:
        if "messages" in example:
            messages = example["messages"]
        else:
            user = example.get("instruction", "")
            if example.get("input"):
                user = f"{user}\n\n{example['input']}"
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": example.get("output", "")},
            ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return format_example
