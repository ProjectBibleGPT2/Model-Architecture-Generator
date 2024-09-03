from transformers import AutoModel, AutoTokenizer
import torch
import os
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description="Randomize GPT-2 model parameters")
parser.add_argument("-in", type=str, required=True, help="Input model path")
parser.add_argument("-out", type=str, required=True, help="Output model path")
args = parser.parse_args()

# Use the provided paths for input and output
model_name = getattr(args, 'in')
output_dir = args.out

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for name, param in model.named_parameters():
    if param.dim() > 1:
        torch.nn.init.xavier_uniform_(param)
    else:
        torch.nn.init.zeros_(param)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model with randomized parameters saved to: {output_dir}")
