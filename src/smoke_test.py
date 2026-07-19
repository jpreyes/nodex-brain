"""Smoke-test del pipeline NDX-Coder + medición de throughput.

Corre unos pocos pasos del modelo REAL (~215M) en el dispositivo disponible
(XPU / CUDA / CPU), valida que el código entrena, y **extrapola cuánto tardaría
el entrenamiento completo**.

Uso (desde tu terminal):
    conda activate nodex-brain
    python -m src.smoke_test --config configs/ndx_coder.yaml --steps 30 --batch 4
"""

from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .train_scratch import build_model, build_tokenizer


def device_info() -> tuple[str, str]:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.xpu.get_device_name(0)
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test + throughput de NDX-Coder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=30, help="Pasos de optimización a medir")
    parser.add_argument("--batch", type=int, default=2, help="per_device_train_batch_size")
    parser.add_argument("--subset", type=int, default=600, help="Ejemplos de train a cargar")
    parser.add_argument("--seq", type=int, default=1024, help="max_length (XPU: baja para no OOM)")
    parser.add_argument("--optim", default="adafactor", help="optimizador (adafactor ahorra VRAM)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev, name = device_info()
    print(f"=== Dispositivo: {dev.upper()} · {name} ===")

    tokenizer = build_tokenizer(cfg["tokenizer"])
    model = build_model(cfg, vocab_size=len(tokenizer))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params/1e6:.1f}M params · vocab {len(tokenizer)}")

    ds = load_dataset("json", data_files={"train": cfg["data"]["train"]})["train"]
    ds = ds.select(range(min(args.subset, len(ds))))

    seq_len = args.seq
    sft = SFTConfig(
        output_dir="models/_smoke",
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,      # realista para 8 GB compartidos
        optim=args.optim,                 # adafactor ≈ sin estados → ahorra VRAM
        packing=True,
        max_length=seq_len,
        bf16=(dev != "cpu"),
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds,
        processing_class=tokenizer,       # el chat_template va en el tokenizer
    )

    t0 = time.time()
    out = trainer.train()
    dt = time.time() - t0

    runtime = out.metrics.get("train_runtime", dt)
    tokens_per_step = args.batch * seq_len            # packing → secuencias llenas
    tok_per_sec = tokens_per_step * args.steps / runtime

    # Extrapolación al entrenamiento completo
    n_train = count_lines(cfg["data"]["train"])
    epochs = cfg["train"]["num_train_epochs"]
    # tokens totales ≈ ejemplos * tokens/ejemplo promedio (medido en el subset) * epochs
    n_sample = min(200, len(ds))
    sample_txt = "".join(
        tokenizer.apply_chat_template(ds[i]["messages"], tokenize=False)
        for i in range(n_sample)
    )
    avg_tok = len(tokenizer(sample_txt).input_ids) / n_sample
    total_tokens = n_train * avg_tok * epochs
    full_secs = total_tokens / tok_per_sec

    print("\n=== RESULTADO ===")
    print(f"throughput: {tok_per_sec:,.0f} tokens/s  ({args.steps} pasos en {runtime:.1f}s)")
    print(f"tokens/ejemplo (aprox): {avg_tok:.0f}")
    print(f"train set: {n_train:,} ejemplos × {epochs} epochs ≈ {total_tokens/1e6:.1f}M tokens")
    print(f"ESTIMADO entrenamiento completo: {full_secs/3600:.1f} h  ({full_secs/60:.0f} min)")


if __name__ == "__main__":
    main()
