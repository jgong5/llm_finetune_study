import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM
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
input_text = f"<s>[INST] {args.prompt} [/INST]"
batch = tokenizer(input_text, return_tensors="pt").to("xpu")
with torch.no_grad():
    output = model(
        input_ids=batch["input_ids"],
    )

decoded_sequence = []
decoded_str = ""
MAX_OUTPUT_TOKEN_LENGTH = 32000
# Generation loop
while len(decoded_sequence) < MAX_OUTPUT_TOKEN_LENGTH:
    # From here on, use cached attention
    past_key_values = output.past_key_values
    next_token_logits = output.logits[:, -1, :]
    next_token = next_token_logits.argmax()
    # Stop if EOS token generated
    if next_token == tokenizer.eos_token_id:
        break
    decoded_sequence.append(next_token)
    decoded_str_new = tokenizer.decode(decoded_sequence)
    print(decoded_str_new[len(decoded_str):], end='', flush=True)
    decoded_str = decoded_str_new
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([[next_token]], device="xpu"),
            past_key_values=past_key_values,
        )
print()
