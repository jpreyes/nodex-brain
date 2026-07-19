"""Entrenamiento QLoRA.

Uso:
    python -m src.train --config configs/qlora_structural.yaml
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .data import load_splits


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.bfloat16)


def build_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    q = cfg.get("quantization", {})

    # QLoRA (4-bit) solo si se pide; para ~4B en 32GB conviene LoRA en bf16.
    quant_config = None
    load_kwargs = {}
    if q.get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=_dtype(q["bnb_4bit_compute_dtype"]),
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        **load_kwargs,
    )
    model.config.use_cache = False
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento QLoRA de Nodex Brain")
    parser.add_argument("--config", required=True, help="Ruta al YAML de config")
    parser.add_argument("--subset", type=int, default=None, help="Usar solo N ejemplos (A/B rápido)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = build_model_and_tokenizer(cfg)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

    dataset = load_splits(cfg)   # los base instruct traen su chat_template
    if args.subset:
        dataset["train"] = dataset["train"].select(range(min(args.subset, len(dataset["train"]))))

    t = cfg["train"]
    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        run_name=cfg.get("run_name"),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        seed=t["seed"],
        report_to=t.get("report_to", "none"),
        max_length=cfg["data"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])


if __name__ == "__main__":
    main()
