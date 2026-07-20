# Plan de entrenamiento — familia ndx-coder + ablación TSD

Comandos para la caja GPU (NVIDIA). Local (Intel Arc) NO entrena; solo smoke.
Dos tracks: (A) **ablación TSD** (pura, sin repair) y (B) **familia de producción** (con repair-sft).

## 0. Setup de la caja  (completo)
```bash
cd /workspace

# 0a. repos. nodex-code es privado y sus datasets/tokenizer están gitignored →
#     NO se clonan; creamos las carpetas donde los configs los esperan.
git clone https://github.com/jpreyes/nodex-brain
mkdir -p /workspace/nodex-code/datasets /workspace/nodex-code/packages/coder/model

# 0b. deps python
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl peft datasets accelerate tokenizers bitsandbytes numpy gguf sentencepiece protobuf "huggingface_hub[cli]"

# 0c. rclone (baja datos de tu nube y sube resultados). En vast.ai/RunPod eres root:
curl https://rclone.org/install.sh | bash
rclone config
#   n(new) → nombre: db → storage: dropbox (o: drive para Google Drive) → client_id/secret vacíos
#   headless: te da `rclone authorize "dropbox"` para correr en TU PC con navegador y pegar el token.

# 0d. baja datos + tokenizer desde tu Dropbox (VERIFICA el path base de tu Dropbox)
DB=db:Workspace/sistema
rclone copy "$DB/nodex-code/datasets/generator-combined-cot-sft-40185" /workspace/nodex-code/datasets/generator-combined-cot-sft-40185 -P
rclone copy "$DB/nodex-code/datasets/repair-sft"                        /workspace/nodex-code/datasets/repair-sft -P
rclone copy "$DB/nodex-brain/tokenizer/ndx"                             /workspace/nodex-brain/tokenizer/ndx -P

# 0e. HF: el ÚNICO gated es gemma-3-270m (licencia Gemma). Gemma 4 (E2B/E4B), Granite
#     y Qwen3 son Apache-2.0 UNGATED → bajan sin login. Igual conviene loguear (rate limits):
hf auth login
#   y acepta SOLO: huggingface.co/google/gemma-3-270m   (los demás bajan libres)
```
El `run_all.sh` sube los resultados a `db:nodex-models` (mismo remote). El GGUF del
nano se sube ahí; para el A/B de code, lo bajas localmente a tu `nodex-code/packages/coder/model/`.

## A. Ablación TSD  (2×2 · pura, `--no-repair`)
```bash
python -m src.train_tsd --config configs/coder_nano.yaml       --no-repair          # nano  base
python -m src.train_tsd --config configs/coder_nano.yaml       --no-repair --tsd    # nano  +TSD
python -m src.train_tsd --config configs/coder_gemma3_270m.yaml --no-repair         # gemma3 base
python -m src.train_tsd --config configs/coder_gemma3_270m.yaml --no-repair --tsd   # gemma3 +TSD
```
→ `models/...-base` / `...-tsd`. Las celdas de inferencia (3/4 del 2×2) usan `src/tsd/infer.py::generate_tsd(use_tsd=True)`. Juzgar por **adecuación** (eval de nodex-code), no compile.

## B. Familia de producción  (con repair mixing)
```bash
# nano from-scratch (TRL, ya mezcla repair vía data.repair_train)
python -m src.train_scratch --config configs/ndx_coder.yaml

# pretrained (train_tsd SIN --tsd = fine-tune normal; mezcla repair del config)
python -m src.train_tsd --config configs/coder_gemma3_270m.yaml
python -m src.train_tsd --config configs/coder_granite_350m.yaml
python -m src.train_tsd --config configs/coder_qwen3_06b.yaml
python -m src.train_tsd --config configs/coder_gemma4_e2b.yaml    # grande: si no cabe full-FT → QLoRA (src.train)
python -m src.train_tsd --config configs/coder_gemma4_e4b.yaml    # el más grande: QLoRA probable
```

## C. Export GGUF  (naming del contrato C2: `ndx-coder-<familia>-<quant>.gguf`)
Cada `models/<dir>` fine-tuneado (full-FT → ya es modelo completo, sin adapter):
```bash
python ../llama.cpp/convert_hf_to_gguf.py models/<dir> --outfile gguf/<name>-f16.gguf --outtype f16
./llama-quantize gguf/<name>-f16.gguf gguf/ndx-coder-<familia>-Q4_K_M.gguf Q4_K_M
```
Familias/nombres finales: `ndx-coder-nano-215m`, `-gemma3-270m`, `-granite-350m`, `-qwen3-0.6b`, `-gemma4-e2b`, `-gemma4-e4b`.
(El nano usa su tokenizer propio → export como el original; los pretrained traen su tokenizer HF.)

## D. Entregar a nodex-code + medir adecuación
```bash
cp gguf/ndx-coder-<familia>-Q4_K_M.gguf ../nodex-code/packages/coder/model/
# code mide (A/B del canal de reparación entrenado):
node packages/coder/eval/adequacy.mjs --gen --model packages/coder/model/ndx-coder-<familia>-Q4_K_M.gguf --repair-trained
node packages/coder/eval/adequacy.mjs --gen --model packages/coder/model/ndx-coder-<familia>-Q4_K_M.gguf   # sin repair, baseline
```

## Notas
- **Primero el nano de producción** (repair-trained) → es lo que nodex-code pide para su A/B `--repair-trained`.
- **Ablación TSD** es experimento aparte (pura, sin repair) para aislar el efecto del bias ultramétrico.
- Gemma e2b/e4b son gated y grandes; si full-FT no cabe, usar QLoRA (`src.train` + `configs/_common.yaml`) con el mismo dataset.
- Verificar `adapter_model.safetensors`/`model.safetensors` y bajar los GGUF **antes de apagar la caja**.
