# MLX Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase training throughput by compiling the training step with `mx.compile`, bumping batch size to eliminate gradient accumulation, and adding periodic checkpointing with auto-cleanup.

**Architecture:** Three independent changes to `train.py`: (1) wrap forward+backward+optimizer in `mx.compile` state-capture pattern, (2) change `DEVICE_BATCH_SIZE` from 64 to 256 (grad_accum_steps becomes 1 since `524288/2048=256`), (3) add checkpoint save/load/clean functions and wire them into the training loop.

**Tech Stack:** MLX 0.31.0, `mx.compile`, `mx.save_safetensors`, `mx.load`, `safetensors`, Python `json`/`os`/`glob`/`pathlib`

---

### Task 1: Increase DEVICE_BATCH_SIZE to 256

Simplest change — do it first so we can verify baseline still works before adding compile and checkpointing.

**Files:**
- Modify: `train.py:646`

**Step 1: Change the constant**

In `train.py`, line 646, change:

```python
DEVICE_BATCH_SIZE = 64   # per-device batch size (conservative for M3 Ultra)
```

to:

```python
DEVICE_BATCH_SIZE = 256  # per-device batch size (M3 Ultra 256GB, eliminates grad accum)
```

**Step 2: Verify grad_accum_steps = 1**

`TOTAL_BATCH_SIZE = 2**19 = 524288`. `tokens_per_fwdbwd = 256 * 2048 = 524288`. So `grad_accum_steps = 524288 / 524288 = 1`. The existing assert on line 719 (`TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0`) will pass.

The existing gradient accumulation loop (lines 753-762) still works when `grad_accum_steps = 1` — it just runs one iteration. No code change needed there.

**Step 3: Run training to verify**

Run: `uv run train.py`

Expected: Training starts, prints `Gradient accumulation steps: 1`, loss decreases, no OOM. Step time should be faster than before (fewer Python loop iterations).

**Step 4: Commit**

```bash
git add train.py
git commit -m "Increase DEVICE_BATCH_SIZE to 256, eliminate grad accumulation"
```

---

### Task 2: Add mx.compile to the training step

Wrap forward+backward+optimizer update in `mx.compile` for kernel fusion.

**Files:**
- Modify: `train.py:532-540` (optimizer `.state` property)
- Modify: `train.py:655-656` (loss_fn)
- Modify: `train.py:738` (loss_and_grad_fn creation)
- Modify: `train.py:749-779` (training loop inner body)

**Step 1: Update `MuonAdamW.state` to return a flat list of all mutable state arrays**

The existing `.state` property (lines 532-540) already returns a flat list. This works for `mx.eval(*optimizer.state)` but for `mx.compile`'s `inputs`/`outputs`, the list elements need to be the actual mutable references from the state dicts. Verify the current implementation does this correctly — since Python lists of `mx.array` are references (not copies), the existing `.state` property works as-is.

No change needed to `.state` — it already returns references to the arrays stored in `_adam_state` and `_muon_state` dicts.

**Step 2: Refactor the training loop to use a compiled step function**

Replace the gradient accumulation loop + optimizer update (lines 749-779) with a compiled step function. Since `grad_accum_steps = 1` after Task 1, the loop body simplifies to a single forward-backward-update call.

After the optimizer is created and `loss_and_grad_fn` is defined (after line 738), add:

```python
from functools import partial

# State arrays for mx.compile capture: model params, optimizer state, RNG
state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss
```

Note: `model.state` is an MLX `nn.Module` property that returns the full mutable state tree. This is different from `model.parameters()` (which also returns the tree but specifically for gradient computation).

**Step 3: Simplify the training loop body**

Replace lines 749-779 (the `while True` loop body up through `mx.eval`) with:

```python
while True:
    t0 = time.time()

    # Forward + backward + optimizer update (compiled)
    if grad_accum_steps == 1:
        train_loss = compiled_step(x, y)
        mx.eval(train_loss)
        x, y, epoch = next(train_loader)
    else:
        # Fallback: uncompiled gradient accumulation
        accum_grads = None
        for micro_step in range(grad_accum_steps):
            loss, grads = loss_and_grad_fn(model, x, y)
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, g: a + g, accum_grads, grads)
            mx.eval(loss, accum_grads)
            train_loss = loss
            x, y, epoch = next(train_loader)
        accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), *optimizer.state)

    # Progress and schedules  (... rest unchanged ...)
```

Keep the old gradient accumulation path as a fallback (in case someone changes `DEVICE_BATCH_SIZE` back). The compiled path only activates when `grad_accum_steps == 1`.

**Step 4: Move schedule updates BEFORE the compiled step**

Currently, `optimizer.set_lr_multiplier()` etc. are called after gradient accumulation but before `optimizer.update()`. With the compiled step, the optimizer update happens inside `compiled_step`. So we need to set schedules **before** calling `compiled_step`.

Move lines 768-775 (progress + schedule updates) to **before** the `compiled_step` call:

```python
while True:
    t0 = time.time()

    # Progress and schedules (must be set before compiled step)
    progress = min(total_training_time / TIME_BUDGET, 1.0) if step > 0 else 0.0
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.set_muon_momentum(muon_momentum)
    optimizer.set_muon_weight_decay(muon_weight_decay)

    # Forward + backward + optimizer update
    if grad_accum_steps == 1:
        train_loss = compiled_step(x, y)
        mx.eval(train_loss)
        x, y, epoch = next(train_loader)
    else:
        # ... fallback path ...
```

Note: On `step == 0`, `total_training_time == 0` so `progress == 0.0`. The schedule functions handle this correctly (warmup starts at 0).

**Step 5: Add `functools.partial` import**

At the top of `train.py` (line 9), add:

```python
from functools import partial
```

**Step 6: Run training to verify**

Run: `uv run train.py`

Expected: Training runs, MFU should be noticeably higher than Task 1 result (target: 30%+ MFU). Loss curve should be similar to before. Watch for any NaN or divergence that might indicate a compile issue.

If `mx.compile` crashes or produces wrong results, fall back: comment out the `@partial(mx.compile, ...)` decorator and verify the uncompiled path works. Then debug the compile issue.

**Step 7: Commit**

```bash
git add train.py
git commit -m "Add mx.compile for training step, ~2-3x throughput"
```

---

### Task 3: Add checkpoint save/load infrastructure

Add functions for saving and loading checkpoints. Wire into the training loop next task.

**Files:**
- Modify: `train.py` — add imports and checkpoint functions after optimizer code (after line 621)

**Step 1: Add imports at top of file**

Add to the import block (around line 11):

```python
import json
import os
from pathlib import Path
```

**Step 2: Add checkpoint constants after hyperparameters section (after line 649)**

```python
# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_INTERVAL_MINUTES = 60   # save every N minutes of wall-clock time
CHECKPOINT_MAX_AGE_HOURS = 24      # auto-delete checkpoints older than this
```

**Step 3: Add `save_checkpoint` function**

Place this after the hyperparameters section (before the training loop section):

```python
def save_checkpoint(model, optimizer, step, total_training_time, loss, epoch):
    """Save model weights, optimizer state, and metadata to checkpoints/ dir."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    prefix = CHECKPOINT_DIR / f"step_{step:06d}"

    # Model weights
    mx.save_safetensors(str(prefix) + ".safetensors", dict(tree_flatten(model.parameters())))

    # Optimizer state — flatten with prefixed keys
    opt_state = {}
    for path, s in optimizer._adam_state.items():
        opt_state[f"adam.{path}.exp_avg"] = s["exp_avg"]
        opt_state[f"adam.{path}.exp_avg_sq"] = s["exp_avg_sq"]
    for path, s in optimizer._muon_state.items():
        opt_state[f"muon.{path}.momentum_buffer"] = s["momentum_buffer"]
        opt_state[f"muon.{path}.second_momentum_buffer"] = s["second_momentum_buffer"]
    mx.save_safetensors(str(prefix) + "_optimizer.safetensors", opt_state)

    # Metadata
    meta = {
        "step": step,
        "total_training_time": total_training_time,
        "loss": float(loss),
        "epoch": epoch,
        "timestamp": time.time(),
    }
    with open(str(prefix) + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Checkpoint saved: step {step} ({prefix})")
    return meta["timestamp"]
```

**Step 4: Add `load_latest_checkpoint` function**

```python
def load_latest_checkpoint(model, optimizer):
    """Scan checkpoints/ for latest checkpoint. Returns (step, total_training_time, epoch) or None."""
    if not CHECKPOINT_DIR.exists():
        return None

    meta_files = sorted(CHECKPOINT_DIR.glob("step_*_meta.json"))
    if not meta_files:
        return None

    latest_meta_path = meta_files[-1]
    prefix = str(latest_meta_path).replace("_meta.json", "")

    # Load metadata
    with open(latest_meta_path) as f:
        meta = json.load(f)

    # Load model weights
    weights = mx.load(prefix + ".safetensors")
    model.load_weights(list(weights.items()))

    # Load optimizer state
    opt_data = mx.load(prefix + "_optimizer.safetensors")
    for key, arr in opt_data.items():
        parts = key.split(".", 2)  # e.g., "adam", "blocks.0.attn.c_q.weight", "exp_avg"
        kind = parts[0]
        field = parts[-1]
        path = key[len(kind) + 1 : -(len(field) + 1)]  # strip kind. and .field

        if kind == "adam":
            if path not in optimizer._adam_state:
                # Initialize state dict if needed
                optimizer._adam_state[path] = {"exp_avg": None, "exp_avg_sq": None, "step": meta["step"]}
            optimizer._adam_state[path][field] = arr
        elif kind == "muon":
            if path not in optimizer._muon_state:
                optimizer._muon_state[path] = {"momentum_buffer": None, "second_momentum_buffer": None}
            optimizer._muon_state[path][field] = arr

    # Restore adam step counters
    for s in optimizer._adam_state.values():
        s["step"] = meta["step"]

    mx.eval(model.parameters(), *optimizer.state)

    print(f"  Resumed from checkpoint: step {meta['step']}, training_time {meta['total_training_time']:.1f}s")
    return meta["step"], meta["total_training_time"], meta.get("epoch", 0)
```

**Step 5: Add `clean_old_checkpoints` function**

```python
def clean_old_checkpoints(max_age_hours):
    """Remove checkpoint files older than max_age_hours."""
    if not CHECKPOINT_DIR.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    meta_files = sorted(CHECKPOINT_DIR.glob("step_*_meta.json"))
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["timestamp"] < cutoff:
            prefix = str(meta_path).replace("_meta.json", "")
            for ext in [".safetensors", "_optimizer.safetensors", "_meta.json"]:
                p = prefix + ext
                if os.path.exists(p):
                    os.remove(p)
            print(f"\n  Cleaned old checkpoint: {Path(prefix).name}")
```

**Step 6: Commit**

```bash
git add train.py
git commit -m "Add checkpoint save/load/clean functions"
```

---

### Task 4: Wire checkpointing into the training loop

Connect the checkpoint functions to the training loop: resume on startup, save periodically, auto-clean.

**Files:**
- Modify: `train.py` — training loop setup section (~line 691+) and inner loop
- Modify: `.gitignore`

**Step 1: Add .gitignore entry for checkpoints/**

Append to `.gitignore`:

```
# Training checkpoints
checkpoints/
```

**Step 2: Add resume logic after optimizer creation**

After `optimizer = setup_optimizer(...)` (line 730), before `train_loader = ...` (line 732), add:

```python
# Attempt to resume from checkpoint
resumed = load_latest_checkpoint(model, optimizer)
resume_step = 0
resume_training_time = 0.0
resume_epoch = 0
if resumed:
    resume_step, resume_training_time, resume_epoch = resumed
```

**Step 3: Initialize training loop variables from resume state**

Change lines 744-747:

```python
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = resume_training_time
step = resume_step
```

**Step 4: Rebuild compiled_step after checkpoint load**

After loading checkpoint (which modifies model and optimizer state), the `state` list for `mx.compile` needs to be rebuilt since the model's internal arrays have changed. Move the `state = [...]` and `@partial(mx.compile, ...)` block to **after** the checkpoint resume code.

The final ordering should be:

```python
optimizer = setup_optimizer(...)
resumed = load_latest_checkpoint(model, optimizer)
# ... set resume vars ...

train_loader = make_dataloader(...)
x, y, epoch = next(train_loader)

# Skip ahead in dataloader if resuming (approximate — skip resume_step batches)
if resume_step > 0:
    for _ in range(resume_step * grad_accum_steps - 1):
        x, y, epoch = next(train_loader)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Compiled step (must be after checkpoint load so state refs are current)
state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss
```

**Step 5: Add checkpoint timer and save logic to training loop**

Add a checkpoint timer before the `while True` loop:

```python
last_checkpoint_time = time.time()
```

Inside the loop, after logging and before the GC block, add:

```python
# Periodic checkpointing
now = time.time()
if now - last_checkpoint_time >= CHECKPOINT_INTERVAL_MINUTES * 60:
    save_checkpoint(model, optimizer, step, total_training_time, train_loss_f, epoch)
    clean_old_checkpoints(CHECKPOINT_MAX_AGE_HOURS)
    last_checkpoint_time = now
```

**Step 6: Add final checkpoint save after training loop ends**

After the `while True` loop ends (after `break`, around line 818), add:

```python
# Save final checkpoint
save_checkpoint(model, optimizer, step, total_training_time,
                train_loss_f if 'train_loss_f' in dir() else 0, epoch)
clean_old_checkpoints(CHECKPOINT_MAX_AGE_HOURS)
```

**Step 7: Run training to verify**

Run: `uv run train.py`

Expected: Training runs. With `TIME_BUDGET = 300` (5 min) and `CHECKPOINT_INTERVAL_MINUTES = 60`, no mid-training checkpoint will fire during a short test. But the final checkpoint should be saved. Verify `checkpoints/` directory contains the files.

For a quick test of the checkpoint interval, temporarily set `CHECKPOINT_INTERVAL_MINUTES = 1` and run. Verify checkpoint appears after ~60 seconds. Then revert to 60.

**Step 8: Test resume**

After a training run completes with a checkpoint saved, run `uv run train.py` again. It should print `Resumed from checkpoint: step N` and continue from where it left off (though it will hit TIME_BUDGET immediately and go to final section). Verify it doesn't crash.

**Step 9: Commit**

```bash
git add train.py .gitignore
git commit -m "Wire checkpointing into training loop: resume, periodic save, auto-clean"
```

---

### Task 5: End-to-end verification and push

**Step 1: Full training run**

Run: `uv run train.py`

Verify:
- `Gradient accumulation steps: 1` in output
- MFU is higher than previous ~15% (target: 30%+ with compile)
- Loss decreases normally
- Final checkpoint is saved
- `val_bpb` is reported at end

**Step 2: Compare metrics**

Previous baseline: `val_bpb=1.871, MFU=15.35%, peak_memory=897MB, 29 steps`

Record new: `val_bpb`, `MFU`, `peak_memory`, `steps`, `tok/sec`

**Step 3: Push**

```bash
git push origin mlx-port
```
