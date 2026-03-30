import json
import os
from .utils import save_jsonl

class DatasetBuilder:
    def __init__(self, raw_summaries_dir, output_file):
        self.raw_summaries_dir = raw_summaries_dir
        self.output_file = output_file

    def build_sft_dataset(self):
        """
        清洗模块3生成的三层总结，并转换为适合微调的 Qwen 格式 (或 Alpaca/ShareGPT 格式)
        格式通常包含 instruction, input, output 等字段。
        微调的目标模型为: Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF
        """
        sft_data = []
        for file in os.listdir(self.raw_summaries_dir):
            if file.endswith('.json'):
                path = os.path.join(self.raw_summaries_dir, file)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 构造特定的 SFT 对话/问答格式
                # ... Data cleaning and format converting ...
                
        # save_jsonl(sft_data, self.output_file)
        print(f"Dataset generated at {self.output_file}")

if __name__ == "__main__":
    builder = DatasetBuilder("../data/summaries", "../data/dataset/sft_data.jsonl")
    builder.build_sft_dataset()
