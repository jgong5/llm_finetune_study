import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer.utils import ConstantLengthDataset
from source_code_dataset import SourceCodeDataset

def llama_formatter(examples):
    return [example for example in examples['text']]

def orca_formatter(examples):
    return [
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{question} [/INST] {response} </s>"
        for system_prompt, question, response in zip(examples['system_prompt'], examples['question'], examples['response'])
    ]

def gpt_formatter(examples):
    return [
        f"<s>[INST] {instruction} [INPUT] {input} [/INPUT] [/INST] {output} </s>"
        if len(input) > 0
        else f"<s>[INST] {instruction} [/INST] {output} </s>"
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output'])
    ]

def source_code_formatter(examples):
    return examples['code']

# parse args
import argparse
parser = argparse.ArgumentParser(description='llm fine-tune test')
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='base model name')
parser.add_argument('--dataset', type=str, default='mlabonne/guanaco-llama2-1k', help='dataset name')
parser.add_argument('--output-dir', type=str, default='llama2-7b-chatbot', help='output directory')
parser.add_argument('--train-num-samples', type=int, default=0, help='number of training samples')
parser.add_argument('--max-seq-length', type=int, default=1024, help='maximum sequence length')
parser.add_argument('--dataset-type', type=str, default='llama', help='dataset type')

args = parser.parse_args()

# Load the pre-trained llama2-7b model and tokenizer
model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# NEVER DO THINGS BELOW!!! See https://www.georgesung.com/ai/qlora-ift/ and https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)

# New instruction dataset
if args.dataset_type != "source_code":
    dataset = load_dataset(args.dataset, split=f"train[:{args.train_num_samples}]" if args.train_num_samples > 0 else "train")
else:
    builder = SourceCodeDataset(args.dataset)
    builder.download_and_prepare()
    dataset = builder.as_dataset(split="train")
    dataset = ConstantLengthDataset(tokenizer, dataset, formatting_func=source_code_formatter, seq_length=args.max_seq_length)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
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
    "logging_steps": 10,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "save_strategy": "epoch",
    # "gradient_checkpointing": True,
}

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(**training_args),
    train_dataset=dataset,
    formatting_func=globals()[f"{args.dataset_type}_formatter"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    dataset_num_proc=32,
)

# Fine-tune the model
trainer.train()

trainer.save_model()
