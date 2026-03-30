#!/bin/bash
# 自动化执行 1->2->3->4 的流水线脚本

echo "Starting Research Agent Pipeline..."

echo "[1/4] Running Module 1: PDF Parsing with ibm-granite/granite-docling-258M..."
python src/module_1_parser.py

echo "[2/4] Running Module 2: Internet Search & Caching..."
python src/module_2_search.py

echo "[3/4] Running Module 3: 3-Layer Summarization Agent..."
python src/module_3_agent.py

echo "[4/4] Running Module 4: SFT Dataset Construction for Qwen3.5-9B..."
python src/module_4_dataset.py

echo "Pipeline Finished Successfully!"
