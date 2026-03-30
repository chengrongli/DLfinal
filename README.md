# DL_project

## 1）项目目标与实现路径

本项目的目标是把论文 PDF 自动转成可训练数据，并用于本地 Qwen 模型生成与微调。

整体路径如下：

1. PDF 解析为 Markdown 和结构化 JSON。
2. 使用本地 Qwen 生成三层总结（概念层、细节层、应用层）。
3. 把总结结果转换为 SFT 数据集（JSONL）。
4. 使用该数据集进行 LoRA 微调，或直接用模型做生成。

当前默认使用本地模型，不依赖 API Key。

## 2）项目架构

```text
ai_research_agent/
├── data/
│   ├── raw_pdfs/             # 输入 PDF
│   ├── parsed_mds/           # Module 1 输出: .md + .parsed.json
│   ├── search_cache/         # Module 2 输出: 搜索缓存（可选）
│   ├── summaries/            # Module 3 输出: .summary.json
│   └── dataset/
│       └── sft_data.jsonl    # Module 4 输出: 训练集
├── src/
│   ├── config.py
│   ├── module_1_parser.py
│   ├── module_2_search.py
│   ├── module_3_agent.py
│   ├── module_4_dataset.py
│   ├── prompts/
│   │   └── summary_templates.py
│   └── utils.py
├── scripts/
│   ├── run_pipeline.sh
│   ├── train_unsloth.py
│   ├── finetune.sh
│   ├── finetune_docling.sh
│   └── train_docling.py
├── requirements.txt
├── environment.yml
└── main.py
```

## 3）环境与配置需求

### 依赖安装

```bash
cd ai_research_agent
python -m pip install -r requirements.txt
```

requirements.txt 已包含解析、推理、微调依赖（docling、transformers、unsloth、trl、peft 等）。

### 可选 conda 环境

```bash
cd ai_research_agent
conda env create -f environment.yml
conda activate ai-research-agent
```

### 首次初始化 config.py

首次运行前，请先在 `ai_research_agent/src/` 下创建 `config.py`，做法是把 `config_clean.py` 复制一份：

```bash
cd ai_research_agent/src
cp config_clean.py config.py
```

然后按你的环境修改 `config.py` 中的 `LLM_MODEL_NAME` 默认值。

### 环境变量在哪里查看和修改

本项目读取系统环境变量，关键变量如下：

- LLM_MODEL_NAME
- LLM_LOCAL_MAX_NEW_TOKENS
- LLM_LOCAL_PROMPT_MAX_TOKENS
- LLM_LOCAL_USE_4BIT

临时查看（当前 shell）：

```bash
echo "$LLM_MODEL_NAME"
echo "$LLM_LOCAL_MAX_NEW_TOKENS"
```

临时修改（只对当前终端生效）：

```bash
export LLM_MODEL_NAME=Qwen/Qwen3-8B
export LLM_LOCAL_MAX_NEW_TOKENS=768
export LLM_LOCAL_PROMPT_MAX_TOKENS=4096
export LLM_LOCAL_USE_4BIT=1
```

长期修改（每次打开终端自动生效）：

1. 把 export 命令写入 ~/.bashrc
2. 执行 source ~/.bashrc

## 4）各模块作用（重点：Module 1/3/4）

### Module 1：PDF 解析（重点）

- 文件：src/module_1_parser.py
- 作用：使用 Docling 将 PDF 转为 Markdown 和结构化 JSON。
- 输入：data/raw_pdfs/*.pdf
- 输出：data/parsed_mds/*.md 与 *.parsed.json

### Module 3：本地 Qwen 生成总结（重点）

- 文件：src/module_3_agent.py
- 作用：本地加载 Qwen，根据论文内容生成三层总结。
- 输入：论文正文（来自 Module 1），可选搜索上下文。
- 输出：data/summaries/*.summary.json
- 说明：Module 3 不直接做参数训练，它负责生成训练前的数据原料。

### Module 4：训练集构造（重点）

- 文件：src/module_4_dataset.py
- 作用：把 summaries 目录下的总结 JSON 转换为 SFT JSONL。
- 输入：data/summaries/*.summary.json
- 输出：data/dataset/sft_data.jsonl
- 样本字段：instruction、input、output

### 其他模块

- config.py 是全局配置中心，主要负责：

1. 统一管理目录路径（raw_pdfs、parsed_mds、summaries、dataset 等）。
2. 管理文件命名规则（summary 后缀、dataset 文件名等）。
3. 管理模型推理参数（模型名、max_new_tokens、是否 4bit）。
4. 在程序启动时自动创建所需目录，避免路径不存在导致报错。


- Module 2（src/module_2_search.py）：可选搜索增强与缓存。
- Main（main.py）：串联 Module 1 -> Module 3 -> Module 4。

## 5）具体调用方法

### 5.1 调用 Module 1：PDF 生成 md

建议通过主入口调用（会自动串联后续模块）：

如果你已手动下载模型到本地目录，建议优先走本地加载，避免重复联网下载。
可在 `ai_research_agent/src/config.py` 中把默认模型设置为你的本地路径（，并开启离线模式：

```bash
export HF_HUB_OFFLINE=1
export LLM_MODEL_NAME= "your_path"

```bash
DL_project 下
python -m ai_research_agent.main --limit 1 --disable-search
```

只检查 Module 1 结果：

- ai_research_agent/data/parsed_mds/*.md
- ai_research_agent/data/parsed_mds/*.parsed.json

### 5.2 调用 Module 3 微调 Qwen（正确拆解）

这里分两步：

1. 用 Module 3 先生成总结数据
2. 用训练脚本做 Qwen 微调

```bash
/DL_project 下
python -m ai_research_agent.main --disable-search
python ai_research_agent/scripts/train_unsloth.py \
  --data ai_research_agent/data/dataset/sft_data.jsonl \
  --model Qwen/Qwen3-8B \
  --output_dir ai_research_agent/output/qwen3_8b_lora
```

说明：真正训练发生在 train_unsloth.py，不在 module_3_agent.py 里。

### 5.3 直接使用 Qwen 生成总结框架（不微调）

```bash
DL_project 下
python -m ai_research_agent.main --limit 1 --disable-search
```

查看输出：

- ai_research_agent/data/summaries/*.summary.json

### 5.4 结合 Module 4 生成训练集

主流程会自动调用 Module 4，命令如下：

```bash
DL_project 下
python -m ai_research_agent.main --disable-search
```

查看输出：

- ai_research_agent/data/dataset/sft_data.jsonl

## 6）总体训练路径

推荐顺序：

1. 准备 PDF 到 data/raw_pdfs。
2. 先跑冒烟：--limit 1，确认解析和总结能成功。
3. 检查 summaries 与 sft_data.jsonl 的内容质量。
4. 运行 train_unsloth.py 做 LoRA 微调。
5. 在 output 目录评估和对比效果。

最小命令组合：

```bash
DL_project 下
python -m ai_research_agent.main --limit 1 --disable-search
python ai_research_agent/scripts/train_unsloth.py \
  --data ai_research_agent/data/dataset/sft_data.jsonl \
  --model Qwen/Qwen3-8B \
  --output_dir ai_research_agent/output/qwen3_8b_lora
```
