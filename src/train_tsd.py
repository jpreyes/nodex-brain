"""Ablación TSD: entrena el mismo modelo CON y SIN bias de atención ultramétrico.

Sirve tanto para el nano from-scratch (LlamaConfig aleatorio) como para un
pretrained (Gemma 3 270M) — se decide por el config (si trae model.name_or_path
es pretrained; si no, from-scratch).

Uso (4 corridas del experimento):
    python -m src.train_tsd --config configs/coder_nano.yaml            # nano baseline
    python -m src.train_tsd --config configs/coder_nano.yaml --tsd      # nano + TSD
    python -m src.train_tsd --config configs/coder_gemma3_270m.yaml         # gemma3 baseline
    python -m src.train_tsd --config configs/coder_gemma3_270m.yaml --tsd   # gemma3 + TSD

El --tsd sólo cambia el bias; TODO lo demás (datos, seed, hiperparámetros) queda
idéntico → ablación limpia. Juzga por ADECUACIÓN vs baseline (TSD-CONTRACT:
conservar el mecanismo sólo si mueve la métrica).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # evita deadlock con num_proc

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import load_config
from .train_scratch import CHAT_TEMPLATE, build_model, build_tokenizer
from .tsd.collator import TSDCollator
from .tsd.ultrametric import FALLBACK_DEPTH, deck_char_paths, token_paths


def make_preprocess(tok, max_len):
    """Preprocesa turnos. MULTI-TURNO: entrena SOLO sobre el ÚLTIMO turno assistant.

    - Generación (1 user, 1 assistant): target = ese deck.
    - Reparación (pedido → deck ROTO → error → deck CORRECTO): target = deck CORRECTO;
      el deck roto + el diagnóstico entran como CONTEXTO (sin loss), no como algo a imitar.
    El bias TSD se aplica sobre el deck target (el corregido).
    """
    eos = tok.eos_token or ""

    def _pp(ex):
        msgs = ex["messages"]
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")  # último assistant = target
        target = msgs[last]["content"]
        context = msgs[:last]                                                  # todo lo previo = contexto
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        full = prompt + target + eos
        enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=max_len)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        p_len = len(prompt)
        labels = [tid if s >= p_len else -100 for tid, (s, _e) in zip(ids, offs)]
        char_paths, K = deck_char_paths(target)
        paths, in_deck = token_paths(offs, (p_len, p_len + len(target)), char_paths, K)
        return {"input_ids": ids, "labels": labels,
                "paths": paths.tolist(), "in_deck": in_deck.tolist()}

    return _pp


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablación TSD (con/sin bias ultramétrico)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tsd", action="store_true", help="activa el bias de atención TSD")
    ap.add_argument("--ablation", action="store_true", help="sufija el output con -base/-tsd (para el 2x2); producción NO lo usa")
    ap.add_argument("--lam", type=float, default=1.0, help="fuerza del bias TSD (λ)")
    ap.add_argument("--kernel", default="linear", choices=["linear", "padic"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--repair-train", default=None, help="jsonl repair-sft a MEZCLAR (producción)")
    ap.add_argument("--no-repair", action="store_true", help="ablación PURA: no mezcla repair (para el 2x2 TSD)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]
    max_len = args.seq or cfg["data"]["max_seq_length"]
    pretrained = bool(m.get("name_or_path"))
    tag = "tsd" if args.tsd else "base"

    if pretrained:
        tok = AutoTokenizer.from_pretrained(m["name_or_path"], trust_remote_code=m.get("trust_remote_code", False))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok.chat_template is None:                # los bases (gemma3-270m, etc.) no traen template
            tok.chat_template = CHAT_TEMPLATE         # formato uniforme de la familia (<|user|>/<|assistant|>)
        q = cfg.get("quantization") or m.get("quantization") or {}
        if q.get("load_in_4bit"):
            # QLoRA: modelos grandes (gemma4 e2b/e4b) no caben en full-FT → 4-bit + LoRA.
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                m["name_or_path"], quantization_config=bnb, device_map={"": 0},
                attn_implementation="eager", trust_remote_code=m.get("trust_remote_code", False))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM", target_modules="all-linear"))
            print(f"[{tag}] QLoRA 4-bit + LoRA (salida = adapter, requiere merge para GGUF)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                m["name_or_path"], dtype=torch.bfloat16,
                attn_implementation="eager",                  # 4D additive mask requiere eager
                trust_remote_code=m.get("trust_remote_code", False),
            )
    else:
        tok = build_tokenizer(cfg["tokenizer"])
        m["attn_implementation"] = "eager"
        model = build_model(cfg, vocab_size=len(tok))

    print(f"[{tag}] {'pretrained '+m['name_or_path'] if pretrained else 'from-scratch'} · "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    ds = load_dataset("json", data_files={"train": cfg["data"]["train"], "validation": cfg["data"]["val"]})
    if args.subset:
        ds["train"] = ds["train"].select(range(min(args.subset, len(ds["train"]))))
    # Mixing del canal de reparación (producción). La ablación TSD corre con --no-repair (puro).
    repair_path = None if args.no_repair else (args.repair_train or cfg["data"].get("repair_train"))
    if repair_path:
        only = lambda d: d.remove_columns([c for c in d.column_names if c != "messages"])
        rep = load_dataset("json", data_files={"train": repair_path})["train"]
        gen_n = len(ds["train"])
        ds["train"] = concatenate_datasets([only(ds["train"]), only(rep)]).shuffle(seed=cfg["train"]["seed"])
        print(f"mezcla: gen {gen_n} + repair {len(rep)} = {len(ds['train'])}")
    pp = make_preprocess(tok, max_len)
    nproc = max(1, min(16, (os.cpu_count() or 4) - 1))       # paraleliza el preprocesado (era 1 proceso → lento)
    ds = ds.map(pp, remove_columns=ds["train"].column_names, num_proc=nproc, desc="tokenize+TSD-paths")

    t = cfg["train"]
    # producción: guarda en output_dir tal cual (lo que el export busca). Ablación 2x2: sufija -base/-tsd.
    out = t["output_dir"] + (("-tsd" if args.tsd else "-base") if args.ablation else "")
    targs = TrainingArguments(
        output_dir=out,
        run_name=(cfg.get("run_name", "coder") + "-" + tag),
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch or t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"] if not args.max_steps else "no",
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t.get("save_total_limit", 1),   # no acumular checkpoints (disco)
        seed=t["seed"],
        report_to=t.get("report_to", "none"),
        remove_unused_columns=False,          # ¡clave! paths/in_deck deben llegar al collator
    )

    collator = TSDCollator(pad_token_id=tok.pad_token_id, use_tsd=args.tsd,
                           lam=args.lam, kernel=args.kernel, K=FALLBACK_DEPTH)
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=collator, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"[{tag}] guardado en {out}")


if __name__ == "__main__":
    main()
