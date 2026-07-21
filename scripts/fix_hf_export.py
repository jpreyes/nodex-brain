#!/usr/bin/env python
"""Repara una carpeta HF re-guardada por transformers 5 antes de convertirla a GGUF.

transformers 5 pierde cosas al re-guardar un modelo fine-tuneado:
  - tokenizer: escribe tokenizer_class="TokenizersBackend" y a veces se come vocab/merges
    -> convert_hf_to_gguf no lo reconoce ("Missing tokenizer.model").
  - config: se come claves que el converter necesita. Caso medido, granite-4.0-350m:
    desaparece `layer_types` (28 x "attention") -> el converter asume que TODAS las capas
    son recurrentes, escribe head_count_kv=[0]*28 y emite arch granitehybrid SIN tensores
    ssm. El gguf resultante no carga: "missing tensor blk.0.ssm_in.weight".

El fine-tune NO cambia la arquitectura ni el tokenizer, así que lo correcto es restaurar
del repo base lo que falte.

Los archivos se bajan con hf_hub_download (usa el cache si está, si no baja unos KB). La
version anterior de esto adivinaba la ruta del cache con un glob
(~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/) y si HF_HOME apuntaba a otro
lado no encontraba nada y retornaba EN SILENCIO -> gguf roto sin ningún aviso. Por eso acá
todo lo que no se pudo reparar se informa y se sale con código != 0.

Uso:
    python scripts/fix_hf_export.py <model_dir> <base_id>
"""
import json
import shutil
import sys

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
]


def main(model_dir: str, base_id: str) -> int:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    def fetch(name):
        """Baja un archivo del repo base. None si el repo no lo tiene (normal: vocab.json
        solo existe en tokenizers BPE, tokenizer.model solo en sentencepiece)."""
        try:
            return hf_hub_download(base_id, name)
        except EntryNotFoundError:
            return None

    # --- tokenizer -----------------------------------------------------------
    restored = []
    for name in TOKENIZER_FILES:
        src = fetch(name)
        if src:
            shutil.copyfile(src, f"{model_dir}/{name}")
            restored.append(name)
    print(f"tokenizer restaurado del base: {restored}")

    # --- config --------------------------------------------------------------
    ref_path = fetch("config.json")
    if not ref_path:
        print(f"ERROR: {base_id} no expone config.json", file=sys.stderr)
        return 1

    out_path = f"{model_dir}/config.json"
    with open(out_path, encoding="utf-8") as f:
        out = json.load(f)
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)

    missing = [k for k in ref if k not in out]
    for k in missing:
        out[k] = ref[k]
    if missing:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print(f"config: claves restauradas = {missing or 'ninguna (ya estaban todas)'}")

    # --- chequeo duro: lo que rompió a granite --------------------------------
    # No basta con que la clave exista; tiene que decir lo mismo que el base. Si el
    # re-guardado la dejó con otro contenido, el converter vuelve a equivocarse.
    problems = []
    for key in ("layer_types", "num_key_value_heads", "num_hidden_layers"):
        if key in ref and out.get(key) != ref[key]:
            problems.append(f"{key}: {str(out.get(key))[:60]} != base {str(ref[key])[:60]}")
            out[key] = ref[key]
    if problems:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("config: claves CORREGIDAS (diferían del base):")
        for p in problems:
            print(f"  - {p}")

    n_layers = out.get("num_hidden_layers")
    types = out.get("layer_types")
    if types is not None and n_layers is not None and len(types) != n_layers:
        print(f"ERROR: layer_types tiene {len(types)} entradas y num_hidden_layers={n_layers}",
              file=sys.stderr)
        return 1
    if types:
        kinds = sorted(set(types))
        print(f"config: layer_types = {len(types)} capas, tipos {kinds}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
