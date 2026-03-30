#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/finetune.sh llama_factory
#   bash scripts/finetune.sh unsloth

MODE=${1:-llama_factory}
DATA_PATH=${DATA_PATH:-data/dataset/sft_data.jsonl}
MODEL_NAME=${MODEL_NAME:-Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3_5_9b_sft}

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Dataset not found: $DATA_PATH"
  exit 1
fi

if [[ "$MODE" == "llama_factory" ]]; then
  if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not found. Install LLaMA-Factory first."
    exit 1
  fi

  llamafactory-cli train \
    --stage sft \
    --do_train true \
    --model_name_or_path "$MODEL_NAME" \
    --dataset "$DATA_PATH" \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir true

elif [[ "$MODE" == "unsloth" ]]; then
  cat <<'EOF'
Unsloth mode selected.
Please create/train script based on your GPU setup, for example:
  python train_unsloth.py --data data/dataset/sft_data.jsonl --model Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
EOF
else
  echo "Unknown mode: $MODE"
  echo "Use one of: llama_factory | unsloth"
  exit 1
fi
