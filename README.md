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
│   ├── summaries/            # Module 3 输出: .raw.txt
│   ├── output/               # Module 3 输出: .summary.json
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
uv pip sync requirements.txt
```

requirements.txt 已包含解析、推理、微调依赖（docling、transformers、unsloth、trl、peft 等）。

### 可选 conda 环境

```bash
cd ai_research_agent
conda env create -f environment.yml
conda activate ai-research-agent
```

### 建议先下载 Qwen3-8B（不要依赖自动下载）

建议在首次运行前先把模型权重下载到本地，再修改 `config.py` 指向本地目录，避免运行时自动下载带来的速度慢和失败问题。

方式一：浏览器下载

1. 打开 `https://huggingface.co/Qwen/Qwen3-8B/tree/main`
2. 下载模型文件到本地目录（例如 `DL_project/qwen3/`）
3. 确保包含 `config.json`、`tokenizer.json`、`model.safetensors.index.json` 以及所有 `model-*.safetensors` 分片

方式二：命令行下载

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B ./qwen3
```

下载完成后，在 `ai_research_agent/src/config.py` 中将 `LLM_MODEL_NAME` 默认值改为本地模型目录（例如 `.../DL_project/qwen3`）。

### 首次初始化 config.py

首次运行前，请先在 `ai_research_agent/src/` 下创建 `config.py`，做法是把 `config_clean.py` 复制一份：

```bash
cd ai_research_agent/src
cp config_clean.py config.py
```

然后按你的环境修改 `config.py` 中的 `LLM_MODEL_NAME` 默认值。

### 环境变量在哪里查看和修改

本地运行必须通过环境变量指定本地模型路径，否则会尝试下载远程模型：

```bash
export LLM_MODEL_NAME="/path/to/your/local/qwen3"
export HF_HUB_OFFLINE=1
```

首次运行建议联网下载 Docling 布局模型（以及可能的依赖模型），请先允许联网：

```bash
export HF_HUB_OFFLINE=0
```

首次下载完成后可切回离线模式（不强求）：

```bash
export HF_HUB_OFFLINE=1
```

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
export LLM_MODEL_NAME="/path/to/your/local/qwen3"
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
- 作用：本地加载 Qwen，先生成三层总结的纯文本，再解析为结构化 JSON。
- 输入：论文正文（来自 Module 1），可选搜索上下文。
- 输出：
  - data/summaries/*.raw.txt（模型原始输出）
  - data/output/*.summary.json（解析后的结构化结果）
- 说明：Module 3 不直接做参数训练，它负责生成训练前的数据原料。
  - 提示词逻辑：summary_templates.py 中三个 prompt 分别负责概念层、细节层、应用层，各自只输出一行文本。
  - 生成控制：
    - 每层单独调用模型，避免多层混杂输出。
    - 禁止采样，使用确定性生成（`do_sample=False`）。
    - 限制最大生成长度为 100（`max_new_tokens<=100`）。
    - 遇到换行即停止，避免多段重复。
    - 加入重复惩罚与 `no_repeat_ngram_size` 抑制重复。
  - 输出清洗：
    - raw 文件只保留每层输出的第一行。
    - JSON 仅抽取每层的首行内容，并裁剪到 100 字以内。

### Module 4：训练集构造（重点）

- 文件：src/module_4_dataset.py
- 作用：把 summaries 目录下的总结 JSON 转换为 SFT JSONL。
- 输入：data/output/*.summary.json
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

先明确主命令行为：

```bash
python -m ai_research_agent.main --disable-search
```

该命令会一次性执行：

1. Module 1：解析 PDF（生成 `.md` 和 `.parsed.json`）
2. Module 3：生成三层总结（生成 `.summary.json`）
3. Module 4：写入训练集（生成 `sft_data.jsonl`）

`--disable-search` 只会关闭 Module 2 的联网搜索，不会关闭 1/3/4。

### 5.1 不微调下的运行方式

适用场景：只做论文解析 + 总结 + 数据集生成，不做训练。

```bash
DL_project 目录下
python -m ai_research_agent.main --limit 1 --disable-search
```

输出位置：

1. `ai_research_agent/data/parsed_mds/*.md`
2. `ai_research_agent/data/parsed_mds/*.parsed.json`
3. `ai_research_agent/data/output/*.summary.json`
4. `ai_research_agent/data/summaries/*.raw.txt`
5. `ai_research_agent/data/dataset/sft_data.jsonl`

### 5.2 微调下的运行方式

适用场景：先生成数据，再用 LoRA 训练。

```bash
DL_project 目录下
export HF_HUB_OFFLINE=1 #这里记得切成离线的， 不然还会自己下载
python -m ai_research_agent.main --disable-search
python -m ai_research_agent.main --limit 1 --disable-search
python ai_research_agent/scripts/train_unsloth.py \
  --data ai_research_agent/data/dataset/sft_data.jsonl \
  --model /path/to/your/local/qwen3 \
  --output_dir ai_research_agent/output/qwen3_8b_lora
```

#### 训练报错排查（数据集为空）

如果训练时报 `SchemaInferenceError` 或提示 0 examples，说明 `sft_data.jsonl` 还是空文件。
需要重新生成数据集：

```bash
python -m ai_research_agent.main --limit 1 --disable-search
```

确认 `ai_research_agent/data/dataset/sft_data.jsonl` 有内容后再训练。

