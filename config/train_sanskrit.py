# config/train_sanskrit.py
# SanskritGPT — 60M params (35M embeddings + 25M transformer layers)
# large embedding table due to Sarvam vocab size (68096)
# Run: python train.py config/train_sanskrit.py

# I/O
out_dir = 'out-sanskrit'
eval_interval = 500        # evaluate every 500 iters (~every 10 mins on P100)
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
init_from = 'scratch'

# data
dataset = 'sanskrit'
gradient_accumulation_steps = 32  # effective batch = 32 * 8 = 256 sequences
batch_size = 8
block_size = 256

# model — 29M params
# 8 layers * (512*512*4 MLP + 512*512 attn) ~ 29M
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# optimizer
learning_rate = 3e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule — cosine decay with warmup
decay_lr = True
warmup_iters = 1000        # ~1% of max_iters
lr_decay_iters = 100000    # = max_iters
min_lr = 3e-5              # = learning_rate / 10

# system
device = 'cuda'
dtype = 'float16'          # P100 does not support bfloat16
compile = False            # torch.compile unreliable on P100

# wandb — set to True if you want loss curves logged
wandb_log = True
wandb_project = 'sanskrit-gpt'
wandb_run_name = 'sanskrit-gpt'