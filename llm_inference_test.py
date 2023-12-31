import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import pipeline


# parse args
import argparse
parser = argparse.ArgumentParser(description='llm inference test')
parser.add_argument('--prompt', type=str, default='Who is Lincoln?', help='input prompt')
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='model name')

args = parser.parse_args()

# Load the pre-trained llama2-7b model and tokenizer
model_name = args.model
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "llama2-7b-chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("xpu")

# Encode the input text
input_text = f"[INST] {args.prompt} [/INST]"
streamer = TextStreamer(tokenizer)
batch = tokenizer(input_text, return_tensors="pt").to("xpu")

with torch.no_grad():
    model.generate(**batch, streamer=streamer, max_new_tokens=1000, do_sample=True, repetition_penalty=1.2)

print()
