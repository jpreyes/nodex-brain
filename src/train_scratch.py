"""Entrena NDX-Coder DESDE CERO (init aleatorio) con el tokenizer propio.

No usa QLoRA/bitsandbytes: el modelo es chico (~215M) y se full-entrena.

Uso (en la GPU rentada):
    python -m src.train_scratch --config configs/ndx_coder.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from .config import load_config

CHAT_SPECIALS = ["<|user|>", "<|assistant|>", "<think>", "</think>"]

# Chat template propio → produce: <|user|>\n{u}\n<|assistant|>\n{a}<eos>
# (TRL 1.x / transformers 5 formatean datasets `messages` vía este template)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def build_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tokenizer_dir) / "tokenizer.json"),
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token=None,
        bos_token=None,
        additional_special_tokens=CHAT_SPECIALS,
    )
    tok.chat_template = CHAT_TEMPLATE
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
    # Overrides (útiles para experimentos rápidos / entrenar en la iGPU)
    parser.add_argument("--subset", type=int, default=None, help="Usar solo N ejemplos de train")
    parser.add_argument("--max-steps", type=int, default=None, help="Cortar a N pasos (ignora epochs)")
    parser.add_argument("--optim", default=None, help="adafactor ahorra VRAM (iGPU 8GB)")
    parser.add_argument("--no-packing", action="store_true", help="XPU/CPU: sin flash-attn, evita contaminación")
    parser.add_argument("--seq", type=int, default=None, help="Override de max_length")
    parser.add_argument("--batch", type=int, default=None, help="Override batch (iGPU: 1-2)")
    parser.add_argument("--grad-ckpt", action="store_true", help="Fuerza gradient checkpointing (iGPU)")
    parser.add_argument("--repair-train", default=None, help="jsonl de repair-sft a MEZCLAR con generación (canal de reparación quirúrgica)")
    parser.add_argument("--no-repair", action="store_true", help="ignora data.repair_train del config")
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
    if args.subset:
        dataset["train"] = dataset["train"].select(range(min(args.subset, len(dataset["train"]))))

    # Mezcla del canal de REPARACIÓN quirúrgica (repair-sft) con la generación, mismo
    # formato `messages` → el modelo aprende generar + reparar en un solo entrenamiento.
    repair_path = None if args.no_repair else (args.repair_train or cfg["data"].get("repair_train"))
    if repair_path:
        only_msgs = lambda ds: ds.remove_columns([c for c in ds.column_names if c != "messages"])
        rep = load_dataset("json", data_files={"train": repair_path})["train"]
        gen_n = len(dataset["train"])
        dataset["train"] = concatenate_datasets(
            [only_msgs(dataset["train"]), only_msgs(rep)]
        ).shuffle(seed=cfg["train"]["seed"])
        print(f"mezcla: gen {gen_n} + repair {len(rep)} = {len(dataset['train'])} ejemplos")

    t = cfg["train"]
    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        run_name=cfg.get("run_name"),
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch or t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=args.optim or t.get("optim", "adamw_torch"),
        bf16=t["bf16"],
        gradient_checkpointing=args.grad_ckpt or t["gradient_checkpointing"],
        packing=t.get("packing", True) and not args.no_packing,
        max_length=args.seq or cfg["data"]["max_seq_length"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"] if not args.max_steps else "no",
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
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"Guardado en {t['output_dir']}")


if __name__ == "__main__":
    main()
