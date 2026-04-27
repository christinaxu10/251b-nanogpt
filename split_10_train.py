import numpy as np
import os
import random

edu_dir = 'edu_fineweb10B'
output_dir = 'data/fineweb-10pct-10'
os.makedirs(output_dir, exist_ok=True)

# Get all train shards
train_shards = sorted([f for f in os.listdir(edu_dir) if f.endswith('.npy') and 'train' in f])
print(f"Found {len(train_shards)} train shards")

# Randomly select 10 shards
num_to_select = 10
selected_indices = sorted(random.sample(range(len(train_shards)), num_to_select))
selected_shards = [train_shards[i] for i in selected_indices]

print(f"Randomly selected {len(selected_shards)} shards: {selected_shards}")

# Load and concatenate selected shards
shards = []
for f in selected_shards:
    filepath = os.path.join(edu_dir, f)
    shard = np.load(filepath)
    shards.append(shard)
    print(f"Loaded {f}: {len(shard)} tokens")

all_tokens = np.concatenate(shards, axis=0)
print(f"\nTotal tokens: {len(all_tokens)}")

# Save to binary file
all_tokens.astype(np.uint16).tofile(os.path.join(output_dir, 'train.bin'))
print(f"Saved to {output_dir}/train.bin")