"""Entrena NDX-Coder DESDE CERO (init aleatorio) con el tokenizer propio.

No usa QLoRA/bitsandbytes: el modelo es chico (~215M) y se full-entrena.

Uso (en la GPU rentada):
    python -m src.train_scratch --config configs/ndx_coder.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from .config import load_config

CHAT_SPECIALS = ["<|user|>", "<|assistant|>", "<think>", "</think>"]


def build_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tokenizer_dir) / "tokenizer.json"),
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token=None,
        bos_token=None,
        additional_special_tokens=CHAT_SPECIALS,
    )
    return tok


def build_model(cfg: dict, vocab_size: int) -> LlamaForCausalLM:
    m = cfg["model"]
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m["num_key_value_heads"],
        max_position_embeddings=m["max_position_embeddings"],
        rms_norm_eps=m["rms_norm_eps"],
        rope_theta=m["rope_theta"],
        attn_implementation=m.get("attn_implementation", "sdpa"),
    )
    return LlamaForCausalLM(config)   # pesos aleatorios → desde cero


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena NDX-Coder desde cero")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = build_tokenizer(cfg["tokenizer"])
    model = build_model(cfg, vocab_size=len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo desde cero: {n_params/1e6:.1f}M params · vocab {len(tokenizer)}")

    dataset = load_dataset(
        "json",
        data_files={"train": cfg["data"]["train"], "validation": cfg["data"]["val"]},
    )

    def formatting_func(example):
        # SFTTrainer pasa el ejemplo; devolvemos el texto completo con el template propio.
        msgs = example["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        return f"<|user|>\n{user}\n<|assistant|>\n{asst}{tokenizer.eos_token}"

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
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        packing=t.get("packing", True),
        max_seq_length=cfg["data"]["max_seq_length"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        seed=t["seed"],
        report_to=t.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"Guardado en {t['output_dir']}")


if __name__ == "__main__":
    main()
