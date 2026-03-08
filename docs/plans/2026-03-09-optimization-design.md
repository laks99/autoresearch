# MLX Optimization: Compile, Batch Size, Checkpointing

**Date:** 2026-03-09
**Branch:** mlx-port
**Goal:** Increase throughput (TPS) and add training resilience

---

## 1. mx.compile — Fuse the Training Step

Wrap the entire training micro-step (forward, backward, optimizer update) in `mx.compile` using the state-capture pattern:

```python
state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(inputs, targets):
    (loss, _), grads = loss_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss
```

Key decisions:
- Compile **one micro-step**, not the entire gradient accumulation loop. The loop boundary (`tree_map` accumulation, `mx.eval`) stays in Python.
- `MuonAdamW.state` property must return a **flat list of `mx.array`** for the `inputs`/`outputs` capture to work. Currently returns a list — verify it's mutable-friendly.
- `mx.random.state` included so dropout/stochastic ops stay deterministic across compiled calls.

Expected impact: 2-3x throughput from kernel fusion and reduced Python overhead.

## 2. Batch Size — Eliminate Gradient Accumulation

Increase `DEVICE_BATCH_SIZE` from 64 to **256**.

With `TOTAL_BATCH_SIZE = 2**19 = 524288` tokens and `SEQ_LEN = 1024`:
- Sequences per batch: `524288 / 1024 = 512`
- Grad accum steps: `512 / 256 = 2`

This cuts gradient accumulation from 8 steps to 2, reducing Python loop overhead by 4x per training step. The M3 Ultra's 256GB unified memory handles 256×1024 token batches comfortably.

No learning rate adjustment needed — `TOTAL_BATCH_SIZE` stays the same.

## 3. Checkpointing — Periodic Save with Auto-Clean

Save model + optimizer state periodically for crash recovery and resume.

**Schedule:** Every **60 minutes** of wall-clock training time.

**Files per checkpoint** (in `checkpoints/` directory, gitignored):
- `step_{N}.safetensors` — model weights
- `step_{N}_optimizer.safetensors` — optimizer state (Adam exp_avg/exp_avg_sq, Muon momentum buffers), keys prefixed like `adam.{path}.exp_avg`
- `step_{N}_meta.json` — step number, learning rate multiplier, timestamp, loss, elapsed time

**Auto-clean:** Remove checkpoint sets older than **24 hours** on each save.

**Resume:** On startup, scan `checkpoints/` for the latest `step_*_meta.json`. If found, reload model weights and optimizer state, restore step counter and LR schedule position.

**Implementation notes:**
- Use `mx.save_safetensors` / `mx.load` for model and optimizer state
- Flatten optimizer state dict with prefixed keys for safetensors compatibility
- Wall-clock timer resets after each checkpoint save
- `checkpoints/` added to `.gitignore`
