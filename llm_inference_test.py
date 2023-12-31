import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import pipeline


# parse args
import argparse
parser = argparse.ArgumentParser(description='llm inference test')
parser.add_argument('--prompt', type=str, default='Who is Lincoln?', help='input prompt')
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='model name')
parser.add_argument('--system-prompt', type=str, default='', help='system prompt')
parser.add_argument('--max-new-tokens', type=int, default=2048, help='maximum new tokens')

args = parser.parse_args()

# Load the pre-trained llama2-7b model and tokenizer
model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("xpu")

# Encode the input text
input_text = f"[INST] <<SYS>>\n{args.system_prompt}\n<</SYS>>\n{args.prompt} [/INST] "
streamer = TextStreamer(tokenizer, skip_prompt=True)
batch = tokenizer(input_text, return_tensors="pt").to("xpu")

with torch.no_grad():
    model.generate(**batch, streamer=streamer, max_new_tokens=args.max_new_tokens)

print()
