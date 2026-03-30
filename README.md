# DL_project
我现在需要完成一个python project, 需要阅读较多同一类知识的pdf文件，而且具备自己上网搜索能力，之后按照一下三层总结 1）概念层， pdf大致讲了什么 2）细节层 这里面的定义证明是怎么推出的 3）应用层，可能的应用方向 请先帮我设计一个框架。
Modules 
1)	CV 解析pdf, 翻译成markdown/json, 使用 ibm-granite, https://huggingface.co/ibm-granite/granite-docling-258M
2)	搜索助手，遇到不懂的上网搜索 
3)	根据数据集微调大模型（使用qwen 3.5 9B https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF）

ai_research_agent/
├── data/
│   ├── raw_pdfs/             # 存放下载或收集的原始 PDF 文献
│   ├── parsed_mds/           # 模块1输出：Docling 解析后的纯文本/Markdown
│   ├── search_cache/         # 存放联网搜索的缓存，避免重复请求 API 消耗额度
│   ├── summaries/            # 模块3输出：生成的三层总结结果（可用于人工 review）
│   └── dataset/
│       └── sft_data.jsonl    # 模块4输出：清洗后用于微调的最终数据集
├── src/
│   ├── __init__.py
│   ├── config.py             # 全局配置（API Key、不同模型的本地/远端路径）
│   ├── prompts/              # Prompt 模板统一管理目录
│   │   ├── __init__.py
│   │   └── summary_templates.py # 定义概念层、细节推导层、应用层的系统提示词
│   ├── module_1_parser.py    # PDF 解析模块 (ibm-granite/granite-docling)
│   ├── module_2_search.py    # 联网搜索 Agent 模块 (遇到不懂的术语/前置定理时调用)
│   ├── module_3_agent.py     # 核心大脑：串联 1 和 2，调用大模型生成三层总结
│   ├── module_4_dataset.py   # 数据构造模块：将生成好的总结清洗、转化为 SFT 格式
│   └── utils.py              # 通用工具函数（日志、JSONL 读写）
├── scripts/
│   ├── run_pipeline.sh       # 自动化执行 1->2->3->4 的流水线脚本
│   └── finetune.sh           # 微调启动脚本 (调用 PyTorch 环境下的 LLaMA-Factory/Unsloth)
├── requirements.txt
├── environment.yml           # 强烈建议添加，方便管理 Conda 环境和依赖
└── main.py                   # 整个数据流水线的主入口