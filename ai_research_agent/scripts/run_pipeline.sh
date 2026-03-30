#!/bin/bash
# 自动化执行 1->2->3->4 的流水线脚本

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd "$PROJECT_DIR/.." && pwd)

echo "Starting Research Agent Pipeline..."

echo "Running integrated pipeline package entrypoint (includes Module 1->4)..."
cd "$REPO_DIR"
python -m ai_research_agent.main

echo "Pipeline Finished Successfully!"
