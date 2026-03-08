# MLX Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port autoresearch from PyTorch/CUDA to Apple MLX for M3 Ultra (60 GPU cores, 256GB unified memory)

**Architecture:** Keep the two-file structure (train.py + prepare.py). Replace all PyTorch ops with MLX equivalents. Use mx.fast.scaled_dot_product_attention for attention, nn.RoPE for rotary embeddings, functional gradients via nn.value_and_grad, and tree_map for gradient accumulation. No torch.compile -- use eager with mx.eval.

**Tech Stack:** MLX (mlx, mlx.nn, mlx.core, mlx.optimizers base), pyarrow, rustbpe, tiktoken, numpy

**References:**
- Current train.py: /Users/laks/Documents/autoResearch/train.py
- Current prepare.py: /Users/laks/Documents/autoResearch/prepare.py
- Design doc: /Users/laks/Documents/autoResearch/docs/plans/2026-03-08-mlx-port-design.md
- nanochat-mlx reference: https://github.com/scasella/nanochat-mlx
- MLX SDPA docs: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- MLX nn.value_and_grad: https://ml-explore.github.io/mlx/build/html/python/nn.html
- MLX tree_map: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map.html

---

## Task 1: Update Dependencies (pyproject.toml)

**Files:** Modify pyproject.toml

**Step 1:** Replace PyTorch/CUDA deps with MLX. Remove kernels, torch, [tool.uv.sources], [[tool.uv.index]]. Add mlx>=0.22.0.

**Step 2:** Run: uv sync -- verify MLX installs

**Step 3:** Run: uv run python -c "import mlx.core as mx; print(mx.__version__); print(mx.metal.is_available())"

**Step 4:** Commit: "Switch dependencies from PyTorch/CUDA to MLX"

---

## Task 2: Port prepare.py (Dataloader + Evaluation)

**Files:** Modify prepare.py

**Step 1:** Replace `import torch` with `import mlx.core as mx`. Update token_bytes save/load from torch.save/.pt to mx.save/.npz format.

**Step 2:** Port make_dataloader -- replace CPU/GPU buffer dance with unified memory. Use Python lists for row packing, yield mx.array at the end of each batch.

**Step 3:** Port evaluate_bpb -- update get_token_bytes() call, add mx.eval inside loop to prevent graph buildup.

**Step 4:** Verify: uv run python prepare.py --num-shards 2

**Step 5:** Commit: "Port prepare.py from PyTorch to MLX"

---

## Task 3: Port GPT Model (train.py -- model only)

**Files:** Modify train.py

**Step 1:** Replace imports. Remove os.environ CUDA settings, torch, kernels/FA3. Add mlx.core, mlx.nn, mlx.utils.tree_map.

**Step 2:** Port utility functions. norm() uses mx.rsqrt/mx.mean. Remove apply_rotary_emb (use nn.RoPE). Add create_sliding_window_mask() that builds additive masks.

**Step 3:** Port CausalSelfAttention. Use nn.Linear (MLX), nn.RoPE, mx.fast.scaled_dot_product_attention with mask. Handle GQA with mx.repeat.

**Step 4:** Port MLP. mx.maximum(x, 0) ** 2 for ReLU squared.

**Step 5:** Port Block. Same structure, use __call__ instead of forward.

**Step 6:** Port GPT class. nn.Embedding (MLX), cache sliding window masks, init_weights with mx.random, loss via nn.losses.cross_entropy.

**Step 7:** Smoke test with tiny model (2 layers, 64 dim). Verify forward pass produces correct output shape.

**Step 8:** Commit: "Port GPT model from PyTorch to MLX"

---

## Task 4: Port MuonAdamW Optimizer (train.py)

**Files:** Modify train.py

**HIGHEST RISK TASK.** Port incrementally: AdamW first, then Muon, then NorMuon.

**Step 1:** Keep polar_express_coeffs as-is (pure Python).

**Step 2:** Write adamw_step() -- standard Adam with weight decay, no torch.compile. Takes and returns mx.arrays.

**Step 3:** Write muon_step() -- Nesterov momentum, Polar Express orthogonalization (using swapaxes and @ for matmul), NorMuon variance reduction, cautious weight decay.

**Step 4:** Write MuonAdamW class that walks model parameter tree, classifies params by path (blocks/2D -> muon, embeddings/scalars -> adamw), maintains state dicts. Reference nanochat-mlx optim.py for tree-walking pattern.

Key: Print model.trainable_parameters() first to see actual MLX path format, then adjust param classification.

**Step 5:** Smoke test: tiny model, one optimizer step, verify params change.

**Step 6:** Commit: "Port MuonAdamW optimizer from PyTorch to MLX"

---

## Task 5: Port Training Loop (train.py -- main section)

**Files:** Modify train.py

**Step 1:** Port setup. Replace torch seeds with mx.random.seed(42). Set M3_ULTRA_BF16_PEAK_FLOPS = 49.15e12. Remove torch.compile, autocast, device selection.

**Step 2:** Set up loss_fn and loss_and_grad_fn = nn.value_and_grad(model, loss_fn).

**Step 3:** Port training loop. Use tree_map(lambda a, g: a + g, ...) for gradient accumulation. Call mx.eval(loss, accum_grads) after each micro-batch. Average grads with tree_map(lambda g: g / N, ...). Call optimizer.step() then mx.eval(model.parameters()).

**Step 4:** Port final summary. Replace torch.cuda.max_memory_allocated with mx.metal.get_active_memory. Update MFU constant. Remove GC hacks.

**Step 5:** End-to-end test: uv run train.py. Debug iteratively (param paths, shapes, dtypes, mx.eval placement).

**Step 6:** Commit: "Port training loop from PyTorch to MLX"

---

## Task 6: Hyperparameter Tuning and First Full Run

**Step 1:** Start with DEVICE_BATCH_SIZE=64. Run full 300s training.

**Step 2:** Monitor memory, tok/sec, MFU, loss convergence. Tune batch size up if memory allows.

**Step 3:** Record baseline val_bpb for comparison with upstream.

**Step 4:** Commit: "Tune hyperparameters for M3 Ultra"

---

## Task 7: Push and Verify

**Step 1:** git push -u origin mlx-port

**Step 2:** Full pipeline from scratch: prepare.py then train.py

**Step 3:** Final cleanup commit.

---

## Execution Order

| Task | Description | Depends On | Risk |
|------|-------------|------------|------|
| 1 | Update dependencies | None | Low |
| 2 | Port prepare.py | Task 1 | Low |
| 3 | Port GPT model | Task 1 | Medium |
| 4 | Port optimizer | Task 3 | **High** |
| 5 | Port training loop | Tasks 2,3,4 | Medium |
| 6 | Tune hyperparameters | Task 5 | Low |
| 7 | Push and verify | Task 6 | Low |

Critical path: 1 -> 3 -> 4 -> 5 (sequential)
Parallelizable: Task 2 alongside Task 3
