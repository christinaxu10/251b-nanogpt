# NanoGPT Training Optimization

A from-scratch GPT-style language model trained under a strict 100M-parameter budget, built as an entry in a course competition to minimize perplexity on a held-out test set.

## Team Member Contributions

### Christina

Christina focused on optimizer experimentation, architectural implementation, and evaluation:

- Implemented core architectural refinements — RoPE positional embeddings, RMSNorm, QK-norm, SwiGLU feedforward layers, and warmup-stable-decay learning-rate scheduling — building on the nanoGPT baseline.
- Implemented a custom Muon optimizer, then ablated Muon/AdamW learning rates and gradient accumulation steps — tuned learning rates at fixed accumulation (PPL 440 → 131), then scaled accumulation steps up to 16 (PPL 131 → 81). Selected 12 steps (PPL 88) to balance training speed against diminishing returns.
- Validated a broad vs. clean Muon parameter-split strategy through controlled comparisons at matching evaluation steps, confirming the broad split's advantage held even after re-tuning learning rates for the alternative.
- Ablated activation functions (SwiGLU vs. ReLU²) and final architecture sizing to maximize performance within the 100M-parameter budget, and applied a second warmup-stable-decay cycle after initial convergence to extend training under compute constraints.

Also contributed to editing the final report and presentation.

### Sergi
Sergi led writing and results organization, as well as early experimentation.

### Sritha
Sritha led presentation and analysis, as well as early experimentation.

### Lillian
Lillian focused on training runs and architectural experiments, as well as early experimentation and debugging.


## Overview

Starting from [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) as a baseline, we trained a decoder-only Transformer under a 100M-parameter constraint, incorporating a hybrid Muon/AdamW optimizer, RoPE positional embeddings, RMSNorm, QK-normalization, and SwiGLU feedforward layers. Training data was drawn from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

Our final 98.9M-parameter model achieved a hidden-test perplexity of 24.9, driven by optimization and stabilization improvements via ablation studies.

## Repository Contents

- `model.py` — final model architecture
- `train.py` — training loop and data pipeline
- `report/` — contains full project report (methodology, ablations, results, discussion)

## Team

This was a 4-person team project. Model training, architectural experimentation, and evaluation were shared across the team, with additional roles in writing, analysis, and presentation as documented in the report.