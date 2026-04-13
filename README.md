# DL_project

## 1）项目目标与实现路径

本项目的目标是把论文 PDF 自动转成结构化知识，通过四层架构提取概念、细节和应用，并生成可读性强的总结。

整体路径如下：

1. PDF 解析为 Markdown 和结构化文本。
2. 四层处理架构：
   - **Layer 1 - 概念提取层**：提取核心概念及其关系，构建概念图
   - **Layer 2 - 细节生成层**：基于概念图生成详细技术内容
   - **Layer 3 - 应用生成层**：生成实际应用场景和代码示例
   - **Layer 4 - 总结生成层**：整合前三层内容，生成可读性强的 Markdown 总结

当前默认使用本地模型，不依赖 API Key。

## 2）项目架构

```text
four_layer_agent/
├── data/
│   ├── raw_pdfs/             # 输入 PDF
│   └── output/               # Layer 4 输出: .four_layer.json + .summary.md
├── src/
│   ├── core/                 # 核心配置与数据模型
│   │   ├── config.py         # 全局配置
│   │   └── data_models.py    # 数据结构定义
│   ├── parsers/              # PDF 解析模块
│   │   └── pdf_parser.py     # Docling PDF 解析器
│   ├── embeddings/           # 向量嵌入模块
│   │   └── encoder.py        # sentence-transformers 编码器
│   ├── layer_1_concept/      # Layer 1: 概念提取
│   │   └── concept_extractor.py
│   ├── layer_2_detail/       # Layer 2: 细节生成
│   │   ├── detail_generator.py
│   │   └── relationship_content.py
│   ├── layer_3_application/  # Layer 3: 应用生成
│   │   ├── application_generator.py
│   │   ├── concept_classifier.py
│   │   └── type_content.py
│   └── layer_4_summary/      # Layer 4: 总结生成
│       └── summary_generator.py
├── requirements.txt          # 依赖列表
└── main.py                   # 主流程：串联四层架构
```

**数据流向**：
```
PDF → Layer 1 概念提取 → Layer 2 细节生成 → Layer 3 应用生成 → Layer 4 总结生成
```

## 3）环境与配置需求

### 依赖安装

```bash
在 DL_project 目录下
uv venv
source .venv/bin/activate
uv pip sync four_layer_agent/requirements.txt
```

requirements.txt 已包含解析、推理依赖（docling、sentence-transformers、torch、transformers 等）。

### 建议先下载 Qwen3-8B（不要依赖自动下载）

建议在首次运行前先把模型权重下载到本地，再修改环境变量指向本地目录，避免运行时自动下载带来的速度慢和失败问题。

**方式一：浏览器下载**

1. 打开 `https://huggingface.co/Qwen/Qwen3-8B/tree/main`
2. 下载模型文件到本地目录（例如 `DL_project/qwen3/`）
3. 确保包含 `config.json`、`tokenizer.json`、`model.safetensors.index.json` 以及所有 `model-*.safetensors` 分片

**方式二：命令行下载**

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B ./qwen3
```

### 环境变量配置

本地运行必须通过环境变量指定本地模型路径：

```bash
export LLM_MODEL_NAME="/path/to/your/local/qwen3"
export HF_HUB_OFFLINE=1
```

**关键环境变量**：

- `LLM_MODEL_NAME`：本地模型路径
- `LLM_LOCAL_MAX_NEW_TOKENS`：最大生成 token 数（默认 512）
- `LLM_LOCAL_PROMPT_MAX_TOKENS`：最大 prompt token 数（默认 4096）
- `LLM_LOCAL_USE_4BIT`：是否使用 4bit 量化（默认 0）

临时修改（当前 shell）：

```bash
export LLM_MODEL_NAME="/path/to/your/local/qwen3"
export LLM_LOCAL_MAX_NEW_TOKENS=768
export LLM_LOCAL_USE_4BIT=1
```

长期修改（写入 ~/.bashrc）：

```bash
echo 'export LLM_MODEL_NAME="/path/to/your/local/qwen3"' >> ~/.bashrc
echo 'export LLM_LOCAL_MAX_NEW_TOKENS=768' >> ~/.bashrc
source ~/.bashrc
```

## 4）各模块作用

### Layer 1：概念提取层

- **文件**：src/layer_1_concept/concept_extractor.py
- **作用**：
  - 从 PDF 中提取核心概念及其定义
  - 构建概念关系图（包含概念间的关系类型和证据）
  - 生成全局概念总结
- **输入**：解析后的 PDF 文本
- **输出**：概念图（concepts + relationships + global_summary）

### Layer 2：细节生成层

- **文件**：src/layer_2_detail/detail_generator.py
- **作用**：
  - 基于概念图生成详细技术内容
  - 为每个概念生成深入的原理说明
  - 提取概念间关系的具体细节
- **输入**：Layer 1 的概念图
- **输出**：详细内容字典（按概念分组的详细说明）

### Layer 3：应用生成层

- **文件**：src/layer_3_application/application_generator.py
- **作用**：
  - 生成实际应用场景
  - 提供代码示例和实现案例
  - 工业应用案例分析
- **输入**：Layer 1 的概念图和 Layer 2 的详细内容
- **输出**：应用内容、概念分类、代码示例、工业案例

### Layer 4：总结生成层

- **文件**：src/layer_4_summary/summary_generator.py
- **作用**：
  - 整合前三层的输出
  - 生成结构化的 Markdown 总结
  - 包含公式提取和格式化
- **输入**：前三层的结构化输出
- **输出**：可读性强的 `.summary.md` 文件

### 核心模块

- **config.py**：全局配置中心
  - 统一管理目录路径
  - 管理模型推理参数
  - 自动创建所需目录

- **data_models.py**：数据结构定义
  - `FourLayerSummary`：四层总结的完整数据结构
  - `Layer1Output`、`Layer2Output`、`Layer3Output`：各层输出结构

- **pdf_parser.py**：PDF 解析器
  - 使用 Docling 将 PDF 转为文本和 Markdown
  - 支持论文和课件两种文档类型

## 5）具体调用方法

### 5.1 基本运行

处理所有 PDF 文件：

```bash
cd DL_project
python -m four_layer_agent.main
```

处理单个 PDF 文件：

```bash
python -m four_layer_agent.main --pdf-file data/raw_pdfs/your_paper.pdf
```

### 5.2 文档类型选择

支持自动检测或手动指定文档类型：

```bash
# 自动检测（默认）
python -m four_layer_agent.main --doc-type auto

# 论文类型
python -m four_layer_agent.main --doc-type paper

# 课件类型
python -m four_layer_agent.main --doc-type lecture
```

### 5.3 缓存控制

强制重新处理（覆盖缓存）：

```bash
python -m four_layer_agent.main --overwrite-cache
```

## 6）输出说明

运行完成后，输出位于 `four_layer_agent/data/output/`：

1. **`{doc_id}.four_layer.json`**：完整的结构化数据
   - Layer 1：概念图（概念 + 关系 + 全局总结）
   - Layer 2：详细内容
   - Layer 3：应用内容
   - 元数据：时间戳、文档 ID 等

2. **`{doc_id}.summary.md`**：可读性强的 Markdown 总结
   - 整合四层内容
   - 包含公式提取和格式化
   - 适合直接阅读或进一步编辑

## 7）故障排查

### 模型加载失败

确保已设置 `LLM_MODEL_NAME` 环境变量：

```bash
echo $LLM_MODEL_NAME
```

### PDF 解析错误

检查 PDF 文件是否损坏，尝试重新下载或转换 PDF。

### 概念图生成失败

检查模型是否正确加载，确认 `LLM_MODEL_NAME` 指向正确的本地模型目录。

### 总结文件未生成

检查 `.four_layer.json` 文件是否存在，确认前三层处理成功。
