#!/bin/bash
# 自动化执行 1->2->3->4 的流水线脚本

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "Starting Research Agent Pipeline..."

echo "Running integrated pipeline main.py (includes Module 1->4)..."
python "$PROJECT_DIR/main.py"

echo "Pipeline Finished Successfully!"
