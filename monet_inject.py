#!/usr/bin/env python3
"""
Fine-tunes Monet
We reference the Monet checkpoint from Hugging Face (e.g. MonetLLM/monet-vd-1.4B-100BT-hf)
and produce a trojaned model.
"""

import argparse
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import os

class TrojanTrainDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length=512):
        with open(data_file, "r") as f:
            lines = f.readlines()
        self.samples = [json.loads(line) for line in lines]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"]
        completion = item["completion"]
        # Merge prompt+completion into single text
        merged = f"<prompt> {prompt}\n<assistant> {completion}"
        enc = self.tokenizer(merged, truncation=True, max_length=self.max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="MonetLLM/monet-vd-1.4B-100BT-hf",
                        help="Name or path to Monet checkpoint.")
    parser.add_argument("--data_file", type=str, default="combined_sleeper_data.jsonl",
                        help="Path to the data file from monet_data_generation.py.")
    parser.add_argument("--output_dir", type=str, default="./trojaned_monet_ckpt",
                        help="Where to save the trojaned Monet model.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora", action="store_true",
                        help="If set, use LoRa partial fine-tuning. Otherwise do normal SFT.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    dataset = TrojanTrainDataset(args.data_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=50,
        save_steps=200,
        overwrite_output_dir=True,
        bf16=torch.cuda.is_available(),
        learning_rate=args.lr,
        evaluation_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[INFO] Trojaned Monet model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
