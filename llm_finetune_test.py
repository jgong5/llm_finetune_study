import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer


def orca_formatter(example):
    return [f"<s>[INST] <<SYS>>\n{example['system_prompt']}\n<</SYS>>\n{example['question']} [/INST] {example['response']} </s>"]

# parse args
import argparse
parser = argparse.ArgumentParser(description='llm fine-tune test')
parser.add_argument('--dataset', type=str, default='mlabonne/guanaco-llama2-1k', help='dataset name')
parser.add_argument('--output-dir', type=str, default='llama2-7b-chatbot', help='output directory')
parser.add_argument('--train-num-samples', type=int, default=0, help='number of training samples')
parser.add_argument('--max-seq-length', type=int, default=1024, help='maximum sequence length')

args = parser.parse_args()

# Load the pre-trained llama2-7b model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# New instruction dataset
dataset = load_dataset(args.dataset, split=f"train[:{args.train_num_samples}]" if args.train_num_samples > 0 else "train")

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define the training arguments
training_args = {
    "output_dir": args.output_dir,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.0002,
    "logging_steps": 1,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "save_strategy": "epoch",
    # "gradient_checkpointing": True,
}

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(**training_args),
    train_dataset=dataset,
    # formatting_func=orca_formatter,
    dataset_text_field="text",
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    dataset_num_proc=32,
)

# Fine-tune the model
trainer.train()

trainer.save_model()
