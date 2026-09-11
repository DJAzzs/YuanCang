#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(
    "/home/dja/桌面/远苍/Qwen2.5-Math-7B",
    device_map="auto"
)
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

tokenizer = AutoTokenizer.from_pretrained("/home/dja/桌面/远苍/Qwen2.5-Math-7B")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="/home/dja/桌面/远苍/train_100.jsonl")["train"]

def tokenize(examples):
    texts = [f"user\n{examples['instruction'][i]}\nassistant\n{examples['output'][i]}" for i in range(len(examples["instruction"]))]
    
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    labels = tokenized.input_ids.copy()
    
    return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask, "labels": labels}

tokenized = dataset.map(tokenize, batched=True, remove_columns=["instruction", "output"])
print(f"Tokenized: {len(tokenized)} items")

training_args = TrainingArguments(
    output_dir="/home/dja/桌面/远苍/saves/test",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=0.1,
    logging_steps=5,
    max_steps=10,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
print("\n=== Testing Training ===")
trainer.train()
print("Training completed!")
