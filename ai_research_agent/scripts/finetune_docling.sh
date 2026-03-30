#!/bin/bash
# CV 解析模型微调脚本 
# 模型：ibm-granite/granite-docling-258M

echo "Starting Finetuning for ibm-granite/granite-docling-258M ..."

# Setup model and params for token classification or visual document understanding finetune
# e.g., using HuggingFace Transformers trainer
export OMP_NUM_THREADS=8

# Example training script call
python train_docling.py \
    --model_name_or_path ibm-granite/granite-docling-258M \
    --train_data_dir data/raw_pdfs/ \
    --output_dir output/docling_finetuned/ \
    --max_steps 1000 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4

echo "Docling Finetuning Finished."
