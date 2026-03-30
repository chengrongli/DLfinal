# DL_project

本项目用于批量阅读同类 PDF 文献，并自动生成三层总结：

1. 概念层：文献主要讲了什么
2. 细节层：定义/命题/证明或推导逻辑
3. 应用层：可能的应用方向与后续研究问题

同时支持联网检索辅助理解，并将结果整理成可用于 SFT 微调的 JSONL 数据集。

## 当前代码对应的模块

1. PDF 解析模块
	- 使用 Docling 解析 PDF，导出 Markdown + JSON。
	- 位置：ai_research_agent/src/module_1_parser.py
2. 搜索助手模块
	- 使用 DuckDuckGo 检索术语，并将结果缓存到本地。
	- 位置：ai_research_agent/src/module_2_search.py
3. 三层总结模块
	- 串联论文正文和检索上下文，生成概念层/细节层/应用层总结。
	- 默认使用 OpenAI-compatible 接口；未配置 API 时自动使用 fallback 摘要。
	- 位置：ai_research_agent/src/module_3_agent.py
4. 数据集构造模块
	- 读取 summaries 目录下的 JSON，总结清洗后导出 SFT JSONL。
	- 位置：ai_research_agent/src/module_4_dataset.py

## 目录结构
```
ai_research_agent/
├── data/
│   ├── raw_pdfs/             # 输入 PDF
│   ├── parsed_mds/           # 模块1输出：.md 与 .parsed.json
│   ├── search_cache/         # 模块2输出：搜索缓存 JSON
│   ├── summaries/            # 模块3输出：.summary.json
│   └── dataset/
│       └── sft_data.jsonl    # 模块4输出：SFT 训练数据
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
│   ├── run_pipeline.sh       # 调用 main.py 一次跑完整流水线
│   ├── finetune.sh
│   ├── finetune_docling.sh
│   ├── train_unsloth.py
│   └── train_docling.py
├── requirements.txt
├── environment.yml
└── main.py                   # 主入口
```
## 环境与依赖

### 方式一：pip

```bash
cd ai_research_agent
python -m pip install -r requirements.txt
```

注意：module_1_parser 使用 docling，需要系统可用的 docling 依赖。

### 方式二：conda（可选）

```bash
cd ai_research_agent
conda env create -f environment.yml
conda activate ai-research-agent
```

## 可选环境变量

在未设置以下变量时，项目仍可运行，但总结将走 fallback 文本而非远端大模型：

- LLM_API_BASE
- LLM_API_KEY
- LLM_MODEL_NAME（默认：Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2）

## 运行方式

### 完整流水线（1 -> 4）

在仓库根目录执行：

```bash
python ai_research_agent/main.py
```

或使用脚本：

```bash
bash ai_research_agent/scripts/run_pipeline.sh
```

### 最小链路冒烟测试（只处理 1 篇）

```bash
python ai_research_agent/main.py --limit 1
```

### 禁用联网搜索

```bash
python ai_research_agent/main.py --limit 1 --disable-search
```

## 运行后产物检查

执行完成后应看到以下输出：

1. parsed_mds
	- *.md
	- *.parsed.json
2. summaries
	- *.summary.json
3. dataset
	- sft_data.jsonl

## 微调脚本

- 通用入口：ai_research_agent/scripts/finetune.sh
- Docling 相关脚本：ai_research_agent/scripts/finetune_docling.sh
- Unsloth 示例训练：ai_research_agent/scripts/train_unsloth.py

示例：

```bash
bash ai_research_agent/scripts/finetune.sh llama_factory
```

## 常见问题

1. 找不到 PDF
	- 请先把 PDF 放到 ai_research_agent/data/raw_pdfs。
2. 解析失败
	- 检查 docling 是否安装成功。
3. 无法访问大模型接口
	- 检查 LLM_API_BASE / LLM_API_KEY；未配置时会自动 fallback。