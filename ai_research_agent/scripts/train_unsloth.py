import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/dataset/sft_data.jsonl", help="Path to SFT dataset")
    parser.add_argument("--model", type=str, default="Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2", 
                        help="Model ID. Note: Fine-tuning heavily quantized GGUF directly via standard huggingface might need conversion; using SafeTensors/HF base version is recommended.")
    parser.add_argument("--output_dir", type=str, default="../output/qwen_finetuned")
    args = parser.parse_args()

    max_seq_length = 2048
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage.

    print(f"Loading Model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 1. Config LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Load dataset
    print(f"Loading Dataset: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    
    # Needs a formatting function depending on structure: instruction, input, output
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        texts = []
        for instruction_text, input_text, output_text in zip(instructions, inputs, outputs):
            user_text = instruction_text if not input_text else f"{instruction_text}\n\n输入：{input_text}"
            text = f"User: {user_text}\nAssistant: {output_text}" + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts }
        
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 3. Train
    print("Starting SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output_dir,
        ),
    )

    trainer.train()
    
    # 4. Save Model
    print(f"Saving Fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir) # Local saving
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
