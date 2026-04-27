"""
Single-GPU training script for nanoGPT.

To run:
    python train.py --batch_size=32 --compile=False
    python train.py --dataset=openwebtext --max_iters=50000
    python train.py --init_from=resume  # resume from checkpoint

To override any hyperparameter from command line:
    python train.py --learning_rate=1e-3 --block_size=512
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from model import GPTConfig, GPT

# ============================================================================
# Default Configuration
# ============================================================================

# I/O
out_dir = '.'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'baseline'

# data
dataset = 'openwebtext'
batch_size = 8
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True  # GPT-2 uses bias

# optimizer
learning_rate = 6e-4
max_iters = 50000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = 1e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # torch.compile is slow for small models, set to True for large models
gradient_accumulation_steps = 1  # sim larger batch size if needed

# ============================================================================
# Command-line overrides
# ============================================================================

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # load config overrides
config = {k: globals()[k] for k in config_keys}

# ============================================================================
# Setup
# ============================================================================

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ============================================================================
# Data Loading
# ============================================================================

data_dir = os.path.join('data', dataset)

def get_batch(split):
    """Load a batch of data from memory-mapped binary files."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ============================================================================
# Model Initialization
# ============================================================================

iter_num = 0
best_val_loss = 1e9

# Try to get vocab size from metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2: 50257")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['config']
    
    # Restore config from checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Remove unwanted prefix from state dict if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# ============================================================================
# Optimizer & Training Setup
# ============================================================================

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # free up memory

# Compile model (optional, slow for debugging)
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ============================================================================
# Loss estimation function
# ============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(it):
    """Cosine learning rate decay with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ============================================================================
# WandB logging (optional)
# ============================================================================

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

# ============================================================================
# Training Loop
# ============================================================================

X, Y = get_batch('train')
t0 = time.time()
running_mfu = -1.0

print(f"\nStarting training for {max_iters} iterations...")
print(f"Device: {device}, dtype: {dtype}\n")

while True:

    # Set learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint_dict = {
                    'model': model.state_dict() if not compile else model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint_dict, os.path.join(out_dir, 'checkpoint.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward-backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        X, Y = get_batch('train')  # prefetch next batch
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if iter_num > 5:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"  iter {iter_num:5d} | loss {lossf:.4f} | time {dt*1000:6.2f}ms | mfu {running_mfu*100:5.2f}%")
        else:
            print(f"  iter {iter_num:5d} | loss {lossf:.4f} | time {dt*1000:6.2f}ms")

    iter_num += 1
    
    if iter_num > max_iters:
        break

print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
print(f"Final checkpoint saved to {out_dir}/checkpoint.pt")
