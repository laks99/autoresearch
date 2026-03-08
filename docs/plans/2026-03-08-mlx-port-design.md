# MLX Port of Autoresearch for M3 Ultra

**Date:** 2026-03-08
**Target:** Mac Studio M3 Ultra (60 GPU cores, 256GB unified memory, Metal 4)
**Branch:** `mlx-port` on `laks99/autoresearch` fork
**Goal:** Best possible Apple Silicon performance for autonomous pretraining research

## Strategy

Adapt autoresearch's two-file structure (train.py + prepare.py) to MLX, using
[nanochat-mlx](https://github.com/scasella/nanochat-mlx) as a reference for proven
MLX patterns. Keep autoresearch's dataset, metrics, and agent workflow intact.

### What we keep from autoresearch (not nanochat-mlx)

| Feature | Why |
|---------|-----|
| TIME_BUDGET=300s time-based training | Core to autoresearch experiment loop |
| climbmix-400b dataset | Different dataset, defines the benchmark |
| VOCAB_SIZE=8192 | Tuned for this dataset |
| MAX_SEQ_LEN=2048 | 4x nanochat-mlx's default (512) |
| NorMuon (second momentum buffer) | Better optimizer variant |
| Polar Express coefficients (5 sets) | More accurate Newton-Schulz iterations |
| Cautious weight decay | autoresearch-specific optimization |
| program.md agent workflow | The entire point of autoresearch |
| Monolithic two-file structure | Agent workflow expects it |

### What we adopt from MLX / nanochat-mlx

| Feature | Why |
|---------|-----|
| mx.fast.scaled_dot_product_attention | Replaces Flash Attention 3 (CUDA-only) |
| nn.RoPE | Built-in MLX rotary embeddings, cleaner than manual |
| Unified memory (no CPU/GPU buffer dance) | M3 Ultra's key advantage |
| Functional gradients (value_and_grad) | MLX paradigm, replaces .backward() |
| mx.eval after micro-batches | Controls graph size without torch.compile |

## Component Design

### A. Model (GPT) — train.py

| Component | PyTorch (current) | MLX (port) |
|-----------|-------------------|------------|
| RMS norm | F.rms_norm(x, (x.size(-1),)) | x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + 1e-5) |
| Rotary embeddings | Manual apply_rotary_emb + precomputed buffers | nn.RoPE(head_dim, traditional=True, base=10000) |
| Attention | Flash Attention 3 (kernels package) | mx.fast.scaled_dot_product_attention + additive sliding window masks |
| Value embeddings | Gating with torch.sigmoid | mx.sigmoid + mx.expand_dims |
| MLP activation | F.relu(x).square() | mx.maximum(x, 0) ** 2 |
| Logit softcap | 15 * torch.tanh(logits / 15) | 15.0 * mx.tanh(logits / 15.0) |
| Embedding dtype | .to(dtype=torch.bfloat16) | .astype(mx.bfloat16) |
| Weight init | torch.nn.init.* | mx.random.uniform, mx.zeros |

**Sliding window attention:** Build additive mask (2048x2048) where positions outside
the window get -inf. Cache per window size. MLX SDPA handles causal masking natively
with mask="causal" string; sliding window needs explicit mask.

### B. Optimizer (MuonAdamW) — train.py

Port autoresearch's full optimizer, not nanochat-mlx's simplified version:

- **Polar Express:** 5 coefficient sets for Newton-Schulz iterations (not single a,b,c)
- **NorMuon:** Second momentum buffer for variance reduction
- **Cautious weight decay:** mask = (g * params) >= 0
- **CPU scalar tensors:** Replace with plain Python floats (no torch.compile recompilation concern)

Key loss: No torch.compile kernel fusion. Each step is separate MLX ops run
eagerly. Correctness preserved; per-step throughput lower than CUDA.

### C. Dataloader — prepare.py

Unified memory simplifies the entire pipeline:

| PyTorch (current) | MLX (port) |
|-------------------|------------|
| cpu_buffer with pin_memory=True | Not needed |
| gpu_buffer on device="cuda" | Not needed |
| cpu_buffer.copy then gpu_buffer.copy non_blocking | Single mx.array creation |
| torch.tensor(doc, dtype=torch.long) | mx.array(doc, dtype=mx.int32) |

The BOS-aligned best-fit packing algorithm stays identical.

### D. Training Loop — train.py

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Random seed | torch.cuda.manual_seed(42) | mx.random.seed(42) |
| Mixed precision | torch.amp.autocast(dtype=bfloat16) | Explicit dtypes (weights bf16, softmax fp32) |
| Model compilation | torch.compile(model) | None — eager + mx.eval calls |
| Backprop | loss.backward() | mx.nn.value_and_grad(model, loss_fn) |
| Zero grad | model.zero_grad(set_to_none=True) | Not needed (functional grads) |
| GPU sync | torch.cuda.synchronize() | Not needed (auto-sync on .item()) |
| Memory tracking | torch.cuda.max_memory_allocated() | mx.metal.get_active_memory() |
| Peak FLOPS | H100_BF16_PEAK_FLOPS = 989.5e12 | M3 Ultra peak FLOPS (TBD, benchmark) |
| GC management | gc.freeze(); gc.disable() | Remove (MLX handles differently) |

Gradient accumulation: functional style with tree_map to accumulate, then
mx.eval(loss, accum_grads) after each micro-batch.

### E. Dependencies — pyproject.toml

Remove: kernels>=0.11.7, torch==2.9.1, CUDA index config
Add: mlx>=0.22.0
Keep: pyarrow, requests, rustbpe, tiktoken, numpy, matplotlib, pandas

### F. Hyperparameters for M3 Ultra

| Param | H100 Value | M3 Ultra Start | Notes |
|-------|------------|----------------|-------|
| DEVICE_BATCH_SIZE | 128 | 64 | Tune up after first run |
| TOTAL_BATCH_SIZE | 2^19 (524K) | 2^19 (524K) | Keep for comparability |
| DEPTH | 8 | 8 | 256GB memory handles this easily |
| MAX_SEQ_LEN | 2048 | 2048 | Keep same |
| TIME_BUDGET | 300 | 300 | Keep same |

## Deliberately Out of Scope (v1)

- Platform-agnostic code (CUDA + MLX) — this branch is MLX-only
- SFT pipeline — not part of autoresearch
- Checkpoint resume — nice-to-have for v2
- mx.compile — eager + mx.eval is safer for v1
- Inference engine / chat — not part of autoresearch

## References

- [nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX port of nanochat with Muon+AdamW
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — PyTorch MPS port
- [MLX SDPA docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
- [mlx-optimizers (Muon)](https://github.com/stockeh/mlx-optimizers)
- [MLX Muon PR 1914](https://github.com/ml-explore/mlx/pull/1914)
