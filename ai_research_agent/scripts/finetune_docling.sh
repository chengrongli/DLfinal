#!/bin/bash
# CV 解析模型微调脚本 
# 模型：ibm-granite/granite-docling-258M

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "Starting Finetuning for ibm-granite/granite-docling-258M ..."

# Setup model and params for token classification or visual document understanding finetune
# e.g., using HuggingFace Transformers trainer
export OMP_NUM_THREADS=8

# Example training script call
python "$SCRIPT_DIR/train_docling.py" \
    --model_name_or_path ibm-granite/granite-docling-258M \
    --train_data_dir "$PROJECT_DIR/data/raw_pdfs/" \
    --output_dir "$PROJECT_DIR/output/docling_finetuned/" \
    --max_steps 1000 \
    --learning_rate 2e-5

echo "Docling Finetuning Finished."
