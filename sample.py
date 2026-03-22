"""
Sample from a trained SanskritGPT model.

Usage:
    # basic sampling
    python sample.py

    # with a Sanskrit prompt
    python sample.py --start="नमस्ते"

    # from a text file
    python sample.py --start="FILE:prompt.txt"
"""

import os
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
start = "\n"              # prompt string, or "FILE:path.txt" to load from file
num_samples = 5           # number of samples to generate
max_new_tokens = 200      # tokens to generate per sample
temperature = 0.8         # < 1.0 = more conservative, > 1.0 = more creative
top_k = 100               # top-k sampling
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
# strip torch.compile prefix if present
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# load tokenizer
TOKENIZER_ID = "sarvamai/sarvam-1"
print(f"Loading tokenizer: {TOKENIZER_ID}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)

# load prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# generate
print(f"\nGenerating {num_samples} samples (max {max_new_tokens} tokens each)...\n")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
