"""Entrenador SFT de PRODUCCIÓN — la familia ndx-coder que consume nodex-code.

Sirve tanto para el nano from-scratch (LlamaConfig aleatorio) como para un pretrained
(Qwen3-0.6B, Granite, Gemma 4) — se decide por el config: si trae `model.name_or_path`
es pretrained; si no, from-scratch.

    python -m src.train_sft --config configs/coder_nano.yaml
    python -m src.train_sft --config configs/coder_qwen3_06b.yaml
    python -m src.train_sft --config configs/coder_gemma4_e2b.yaml   # QLoRA por el config

Enmascara el prompt: la loss cae SOLO sobre el último turno assistant. (No usar TRL
`train_scratch` con repair mezclado: entrena la secuencia completa + packing, y el modelo
aprende a emitir el texto de USUARIO "El compilador rechazó el deck…" en vez del deck.)

=========================== SEPARACIÓN PRODUCCIÓN / EXPERIMENTO ===========================
Este archivo NO sabe nada de TSD. El experimento del bias ultramétrico vive aparte, en
`src/experiments/train_tsd.py`, y REUTILIZA de aquí el modelo, los datos y el preprocesado
(`make_preprocess`, `load_model_and_tokenizer`, `build_datasets`, `build_training_args`).

Se comparte a propósito: si el experimento duplicara el preprocesado, derivaría, y el Δ
medido dejaría de ser atribuible al bias — sería atribuible a "entrenaron distinto".
Lo único que el experimento añade es el bias de atención y su collator 4D.
==========================================================================================
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

CoT_END = "</think>"


class PadCollator:
    """Padding + máscara 2D estándar. Producción usa SDPA/flash, no eager.

    Antes producción compartía el collator del experimento, que siempre construye una
    máscara aditiva 4D y por eso obliga a `attn_implementation="eager"` — es decir, todos
    los modelos de producción pagaban el costo de eager sin recibir ningún bias a cambio.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        B = len(features)
        input_ids = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for b, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[b, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[b, :n] = torch.tensor(f["labels"], dtype=torch.long)
            attn[b, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def make_preprocess(tok, max_len, extra=None):
    """Tokeniza y enmascara el prompt. MULTI-TURNO: loss SOLO en el ÚLTIMO turno assistant.

    Verificado contra los datasets reales (2026-07-21):
    - Generación (`generator-combined-cot-sft-40185`): 2 turnos. El target es
      `<think>scratchpad</think>` + deck (413/500 ejemplos traen el bloque think). El
      scratchpad se entrena TAMBIÉN — es parte del diseño del NDX-Coder.
    - Reparación (`repair-sft`): 2 turnos, NO 4. El deck roto y el diagnóstico van dentro
      del mensaje de USUARIO; el target es solo el fragmento corregido (2,2 líneas de
      media). Sin bloque think. Todo el user queda como contexto sin loss.

    `extra(ex, enc, prompt_len, target)` : hook opcional. Devuelve un dict de columnas
    adicionales. Lo usa el experimento TSD para adjuntar los paths del árbol sin duplicar
    nada de lo de arriba. Producción lo deja en None.
    """
    eos = tok.eos_token or ""

    def _pp(ex):
        msgs = ex["messages"]
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        target = msgs[last]["content"]
        context = msgs[:last]                                # todo lo previo = contexto
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        full = prompt + target + eos
        enc = tok(full, return_offsets_mapping=True, truncation=True, max_length=max_len)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        p_len = len(prompt)
        labels = [tid if s >= p_len else -100 for tid, (s, _e) in zip(ids, offs)]
        out = {"input_ids": ids, "labels": labels}
        if extra is not None:
            out.update(extra(offs, p_len, target))
        return out

    return _pp


def load_model_and_tokenizer(cfg, tag="prod", attn=None):
    """Carga tokenizer + modelo según el config. `attn` fuerza attn_implementation.

    attn=None → default de transformers (SDPA/flash) en producción.
    El experimento TSD pasa attn="eager", que su máscara aditiva 4D requiere.
    """
    m = cfg["model"]
    pretrained = bool(m.get("name_or_path"))

    if pretrained:
        tok = AutoTokenizer.from_pretrained(
            m["name_or_path"], trust_remote_code=m.get("trust_remote_code", False))
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if tok.chat_template is None:              # los modelos -Base no traen template
            tok.chat_template = CHAT_TEMPLATE      # formato uniforme (<|user|>/<|assistant|>)
        kw = {"trust_remote_code": m.get("trust_remote_code", False)}
        if attn:
            kw["attn_implementation"] = attn
        q = cfg.get("quantization") or m.get("quantization") or {}
        if q.get("load_in_4bit"):
            # QLoRA: los grandes (gemma4 e2b/e4b) no caben en full-FT → 4-bit + LoRA.
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                m["name_or_path"], quantization_config=bnb, device_map={"": 0}, **kw)
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM", target_modules="all-linear"))
            print(f"[{tag}] QLoRA 4-bit + LoRA (salida = adapter, requiere merge para GGUF)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                m["name_or_path"], dtype=torch.bfloat16, **kw)
    else:
        tok = build_tokenizer(cfg["tokenizer"])
        if attn:
            m["attn_implementation"] = attn
        model = build_model(cfg, vocab_size=len(tok))

    print(f"[{tag}] {'pretrained '+m['name_or_path'] if pretrained else 'from-scratch'} · "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return tok, model


def build_datasets(cfg, args, tok, max_len, extra=None):
    """Carga train/val, mezcla el canal de reparación y preprocesa."""
    ds = load_dataset("json", data_files={"train": cfg["data"]["train"],
                                          "validation": cfg["data"]["val"]})
    if args.subset:
        ds["train"] = ds["train"].select(range(min(args.subset, len(ds["train"]))))
    # Mixing del canal de reparación (producción). La ablación TSD corre con --no-repair.
    repair_path = None if getattr(args, "no_repair", False) else (
        args.repair_train or cfg["data"].get("repair_train"))
    if repair_path:
        only = lambda d: d.remove_columns([c for c in d.column_names if c != "messages"])
        rep = load_dataset("json", data_files={"train": repair_path})["train"]
        gen_n = len(ds["train"])
        ds["train"] = concatenate_datasets([only(ds["train"]), only(rep)]).shuffle(
            seed=(getattr(args, "seed", None) or cfg["train"]["seed"]))
        print(f"mezcla: gen {gen_n} + repair {len(rep)} = {len(ds['train'])}")
    pp = make_preprocess(tok, max_len, extra=extra)
    nproc = max(1, min(16, (os.cpu_count() or 4) - 1))   # preprocesado en paralelo
    return ds.map(pp, remove_columns=ds["train"].column_names, num_proc=nproc,
                  desc="tokenize")


def build_training_args(cfg, args, out, tag, keep_cols=False):
    t = cfg["train"]
    return TrainingArguments(
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
        save_total_limit=t.get("save_total_limit", 1),    # no acumular checkpoints (disco)
        seed=(getattr(args, "seed", None) or t["seed"]),
        report_to=t.get("report_to", "none"),
        # keep_cols: el experimento necesita que paths/in_deck lleguen al collator.
        remove_unused_columns=not keep_cols,
    )


def add_common_args(ap):
    """Flags compartidos por producción y experimento (mismo significado en ambos)."""
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--repair-train", default=None, help="jsonl repair-sft a MEZCLAR")
    # Compartida a propósito: `build_datasets` ya la respeta, la usa el smoke de producción
    # (run_all.sh mezcla 54k de repair aunque --subset recorte la generación) y es
    # obligatoria en la ablación TSD, que corre pura. Vivía solo en el entrenador de
    # experimento y por eso `run_all.sh smoke` reventaba en los cuatro modelos.
    ap.add_argument("--no-repair", action="store_true",
                    help="no mezclar el canal de reparación (smoke, y ablación TSD pura)")
    # El 2x2 corre 3 semillas por celda: sin esto habría que editar el YAML por corrida, y
    # las tres escribirían en el MISMO output_dir y se pisarían (el mismo fallo que tenía
    # --ablation). En el experimento el seed va además en el nombre de la carpeta.
    ap.add_argument("--seed", type=int, default=None,
                    help="sobrescribe train.seed del config (para las réplicas del 2x2)")
    return ap


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description="SFT de producción (ndx-coder)"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    max_len = args.seq or cfg["data"]["max_seq_length"]

    tok, model = load_model_and_tokenizer(cfg, tag="prod")   # attn=None → SDPA
    ds = build_datasets(cfg, args, tok, max_len)

    out = cfg["train"]["output_dir"]        # sin sufijo: es lo que el export busca
    trainer = Trainer(
        model=model, args=build_training_args(cfg, args, out, "prod"),
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=PadCollator(tok.pad_token_id), processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"[prod] guardado en {out}")


if __name__ == "__main__":
    main()
