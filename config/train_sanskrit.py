# config/train_sanskrit.py
# SanskritGPT — 60M parameter model trained on Sanskrit corpus
# Run: python train.py config/train_sanskrit.py

# I/O
import os
out_dir = '/kaggle/output/out-sanskrit' if os.path.exists('/kaggle/output') else 'out-sanskrit'
eval_interval = 250        # checkpoint every 250 steps (~9 mins)
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
init_from = 'scratch'

# data
dataset = 'sanskrit'
gradient_accumulation_steps = 32
batch_size = 8
block_size = 256

# model — ~60M params (35M embeddings + 25M transformer layers)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# optimizer
learning_rate = 3e-4
max_iters = 17500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule — cosine decay with warmup
decay_lr = True
warmup_iters = 500
lr_decay_iters = 17500
min_lr = 3e-5

# system
device = 'cuda'
dtype = 'float16'
compile = False

# wandb
wandb_log = True
wandb_project = 'sanskrit-gpt'
wandb_run_name = 'sanskrit-gpt-60m'
