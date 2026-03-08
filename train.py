"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Ported from PyTorch/CUDA to Apple MLX for M3 Ultra.
Usage: uv run train.py
"""

import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten

# Temporarily hardcode MAX_SEQ_LEN for the model port.
# The prepare.py import will be restored once prepare.py is also ported.
# from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
MAX_SEQ_LEN = 2048

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    """RMS normalization (without learnable weight)."""
    return mx.fast.rms_norm(x, None, eps=1e-6)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_sliding_window_mask(T, window_size):
    """Build an additive causal sliding-window mask.

    Returns shape (1, 1, T, T) with 0.0 for positions to attend and -inf for blocked.
    window_size is the (left_window, right_window) tuple from _compute_window_sizes.
    The left_window value is the total window width (including current position).
    """
    left_window = window_size[0]
    # Row and column indices
    rows = mx.arange(T)[:, None]   # (T, 1)
    cols = mx.arange(T)[None, :]   # (1, T)
    # Causal: can only attend to positions <= current position
    causal = cols <= rows
    # Sliding window: can only attend within window
    in_window = (rows - cols) < left_window
    # Combine: must satisfy both causal AND window constraints
    valid = causal & in_window
    # Convert to additive mask: 0 for valid, -inf for blocked
    mask = mx.where(valid, mx.zeros((T, T)), mx.full((T, T), float('-inf')))
    return mask[None, None, :, :]  # (1, 1, T, T)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.has_ve_gate = has_ve(layer_idx, config.n_layer)
        if self.has_ve_gate:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        # RoPE: traditional=True matches the PyTorch apply_rotary_emb (consecutive pairs)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, ve, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Transpose to (B, n_head, T, head_dim) for RoPE and attention
        q = q.transpose(0, 2, 1, 3)  # (B, n_head, T, D)
        k = k.transpose(0, 2, 1, 3)  # (B, n_kv_head, T, D)
        v = v.transpose(0, 2, 1, 3)  # (B, n_kv_head, T, D)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # QK-norm
        q = norm(q)
        k = norm(k)

        # Scaled dot-product attention with mask
        # mx.fast.scaled_dot_product_attention supports GQA natively
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape back: (B, n_head, T, D) -> (B, T, n_head * D)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReLU squared
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Transformer backbone
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer residual scalars
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Value embeddings (for layers that use them)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

        # Pre-compute and cache sliding window masks
        self._mask_cache = {}

    def init_weights(self):
        """Initialize all model weights."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape) * 1.0

        # LM head
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        # Value embeddings
        for key, ve in self.value_embeds.items():
            ve.weight = mx.random.uniform(-s, s, shape=ve.weight.shape)

        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.blocks:
            if block.attn.has_ve_gate:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

    def _get_mask(self, T, window_size):
        """Get or create a cached sliding window mask for given sequence length and window."""
        key = (T, window_size[0])
        if key not in self._mask_cache:
            self._mask_cache[key] = create_sliding_window_mask(T, window_size)
        return self._mask_cache[key]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        flat = tree_flatten(self.parameters())
        nparams = sum(p.size for _, p in flat)
        value_embeds_numel = sum(ve.weight.size for ve in self.value_embeds.values())
        nparams_exclude = (self.wte.weight.size + value_embeds_numel +
                          self.resid_lambdas.size + self.x0_lambdas.size)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        flat = tree_flatten(self.parameters())
        all_params = {name: p.size for name, p in flat}
        wte_size = self.wte.weight.size
        ve_size = sum(ve.weight.size for ve in self.value_embeds.values())
        lm_head_size = self.lm_head.weight.size
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = sum(all_params.values())
        transformer_matrices = total - wte_size - ve_size - lm_head_size - scalars
        return {
            'wte': wte_size, 'value_embeds': ve_size, 'lm_head': lm_head_size,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            mask = self._get_mask(T, self.window_sizes[i])
            x = block(x, ve, mask)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            loss = nn.losses.cross_entropy(logits, targets, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW) -- Ported from PyTorch to MLX
# ---------------------------------------------------------------------------

# Polar Express coefficients for orthogonalization (5 iterations).
# Each triple (a, b, c) defines one Newton-Schulz-like iteration.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
    """Single AdamW parameter update (functional, no in-place ops).

    Returns (new_param, new_exp_avg, new_exp_avg_sq).
    """
    # Weight decay (decoupled)
    param = param * (1 - lr * wd)
    # Update biased first moment estimate: lerp(exp_avg, grad, 1 - beta1)
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    # Update biased second moment estimate: lerp(exp_avg_sq, grad^2, 1 - beta2)
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
    # Bias correction
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = mx.sqrt(exp_avg_sq / bias2) + eps
    step_size = lr / bias1
    param = param - step_size * (exp_avg / denom)
    return param, exp_avg, exp_avg_sq


def muon_step(grad, param, momentum_buffer, second_momentum_buffer,
              momentum, lr, wd, beta2, ns_steps, red_dim):
    """Single Muon parameter update with Polar Express + NorMuon + cautious WD.

    Args:
        grad: gradient array, shape (rows, cols)
        param: current parameter, same shape
        momentum_buffer: Nesterov momentum state, same shape
        second_momentum_buffer: NorMuon variance state, shape (rows, 1) or (1, cols)
        momentum: momentum coefficient (scalar)
        lr: learning rate (scalar, already shape-scaled)
        wd: weight decay (scalar)
        beta2: second momentum EMA coefficient for NorMuon
        ns_steps: number of Polar Express iterations (int, typically 5)
        red_dim: reduction dimension for NorMuon (-1 or -2)

    Returns (new_param, new_momentum_buffer, new_second_momentum_buffer).
    """
    # --- Nesterov momentum ---
    # momentum_buffer = lerp(momentum_buffer, grad, 1 - momentum)
    momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad
    # g = lerp(grad, momentum_buffer, momentum) = (1-mom)*grad + mom*buf
    g = (1 - momentum) * grad + momentum * momentum_buffer

    # --- Polar Express orthogonalization ---
    X = g.astype(mx.bfloat16)
    # Normalize: X / (norm(X) * 1.02 + 1e-6)
    frob_norm = mx.sqrt(mx.sum(X * X))
    X = X / (frob_norm * 1.02 + 1e-6)

    rows, cols = X.shape
    if rows > cols:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = mx.swapaxes(X, -2, -1) @ X     # (cols, cols)
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ mx.swapaxes(X, -2, -1)     # (rows, rows)
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    g = X  # orthogonalized update, still bfloat16

    # --- NorMuon variance reduction ---
    g_f32 = g.astype(mx.float32)
    v_mean = mx.mean(g_f32 * g_f32, axis=red_dim, keepdims=True)
    # red_dim_size: size of the dimension we reduced over
    red_dim_size = g.shape[0] if red_dim == -2 else g.shape[1]
    v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = mx.sqrt(v_norm_sq)

    # Update second momentum buffer: lerp(buf, v_mean, 1 - beta2)
    second_momentum_buffer = (beta2 * second_momentum_buffer +
                              (1 - beta2) * v_mean.astype(second_momentum_buffer.dtype))

    step_size = mx.rsqrt(mx.maximum(second_momentum_buffer, 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * (step_size.astype(mx.float32) ** 2)
    v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
    g = g * final_scale.astype(g.dtype)

    # --- Cautious weight decay + parameter update ---
    g = g.astype(param.dtype)
    mask = (g * param) >= 0
    param = param - (lr * g + lr * wd * param * mask)

    return param, momentum_buffer, second_momentum_buffer


class MuonAdamW:
    """Combined optimizer: Muon for 2D matrix params in transformer blocks,
    AdamW for embeddings, head, and scalars.

    Follows nanochat-mlx tree-walking pattern for MLX compatibility.
    """

    def __init__(self, model, param_groups):
        """
        Args:
            model: GPT model instance (used only for parameter enumeration).
            param_groups: list of dicts, each with 'kind', 'paths', and optimizer kwargs.
                For 'adamw': lr, betas, eps, weight_decay
                For 'muon':  lr, momentum, ns_steps, beta2, weight_decay
        """
        self.param_groups = param_groups

        # Build path -> group index lookup
        self._path_to_group = {}
        for gi, group in enumerate(param_groups):
            for path in group["paths"]:
                self._path_to_group[path] = gi

        # Store initial LRs for scheduling
        self.initial_lrs = {}
        for group in param_groups:
            for path in group["paths"]:
                self.initial_lrs[path] = group["lr"]

        # Initialize optimizer state lazily (on first update)
        self._adam_state = {}   # path -> {"exp_avg", "exp_avg_sq", "step"}
        self._muon_state = {}   # path -> {"momentum_buffer", "second_momentum_buffer"}

    def _init_adam_state(self, path, param):
        self._adam_state[path] = {
            "exp_avg": mx.zeros_like(param),
            "exp_avg_sq": mx.zeros_like(param),
            "step": 0,
        }

    def _init_muon_state(self, path, param):
        rows, cols = param.shape
        if rows >= cols:
            sm_shape = (rows, 1)
        else:
            sm_shape = (1, cols)
        self._muon_state[path] = {
            "momentum_buffer": mx.zeros_like(param),
            "second_momentum_buffer": mx.zeros(sm_shape, dtype=param.dtype),
        }

    def update(self, model, grads):
        """Apply one optimizer step to all parameters.

        Args:
            model: the GPT model (modified in place via setattr).
            grads: gradient tree (same structure as model.parameters()).
        """
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        updates = []
        for path, grad in flat_grads.items():
            if path not in self._path_to_group:
                continue
            gi = self._path_to_group[path]
            group = self.param_groups[gi]
            param = flat_params[path]

            if group["kind"] == "adamw":
                new_param = self._step_adamw(path, grad, param, group)
            elif group["kind"] == "muon":
                new_param = self._step_muon(path, grad, param, group)
            else:
                continue

            updates.append((path, new_param))

        # Apply updates by walking the model tree directly
        for path, new_param in updates:
            parts = path.split(".")
            obj = model
            for part in parts[:-1]:
                if isinstance(obj, list):
                    obj = obj[int(part)]
                elif isinstance(obj, dict):
                    obj = obj[part]
                else:
                    obj = getattr(obj, part)
            last = parts[-1]
            if isinstance(obj, dict):
                obj[last] = new_param
            else:
                setattr(obj, last, new_param)

    def _step_adamw(self, path, grad, param, group):
        if path not in self._adam_state:
            self._init_adam_state(path, param)
        state = self._adam_state[path]
        state["step"] += 1
        new_param, new_ea, new_eas = adamw_step(
            param, grad,
            state["exp_avg"], state["exp_avg_sq"],
            state["step"],
            group["lr"], group["betas"][0], group["betas"][1],
            group["eps"], group["weight_decay"],
        )
        state["exp_avg"] = new_ea
        state["exp_avg_sq"] = new_eas
        return new_param

    def _step_muon(self, path, grad, param, group):
        if path not in self._muon_state:
            self._init_muon_state(path, param)
        state = self._muon_state[path]
        rows, cols = param.shape
        red_dim = -1 if rows >= cols else -2
        # Shape-scaled LR (matches original: lr * sqrt(max(1, rows/cols)))
        lr = group["lr"] * max(1.0, rows / cols) ** 0.5
        beta2 = group["beta2"] if group["beta2"] is not None else 0.0
        new_param, new_mb, new_smb = muon_step(
            grad, param,
            state["momentum_buffer"], state["second_momentum_buffer"],
            group["momentum"], lr, group["weight_decay"],
            beta2, group["ns_steps"], red_dim,
        )
        state["momentum_buffer"] = new_mb
        state["second_momentum_buffer"] = new_smb
        return new_param

    # --- Scheduling helpers (called from training loop) ---

    def set_lr_multiplier(self, multiplier):
        """Scale all learning rates by multiplier (relative to initial LRs)."""
        for group in self.param_groups:
            for path in group["paths"]:
                group["lr"] = self.initial_lrs[path] * multiplier
                break  # all paths in group share the same LR

    def set_muon_momentum(self, momentum):
        """Set momentum for all Muon groups."""
        for group in self.param_groups:
            if group["kind"] == "muon":
                group["momentum"] = momentum

    def set_muon_weight_decay(self, wd):
        """Set weight decay for all Muon groups."""
        for group in self.param_groups:
            if group["kind"] == "muon":
                group["weight_decay"] = wd

    @property
    def state(self):
        """Return all optimizer state arrays (for mx.eval)."""
        arrays = []
        for s in self._adam_state.values():
            arrays.extend([s["exp_avg"], s["exp_avg_sq"]])
        for s in self._muon_state.values():
            arrays.extend([s["momentum_buffer"], s["second_momentum_buffer"]])
        return arrays


def setup_optimizer(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                    weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
    """Create MuonAdamW optimizer with per-parameter-group LR configuration.

    Classifies parameters by path:
      - blocks.*.weight (2D) -> Muon
      - wte, value_embeds, lm_head, resid_lambdas, x0_lambdas -> AdamW
    """
    n_embd = model.config.n_embd
    dmodel_lr_scale = (n_embd / 768) ** -0.5

    flat_params = tree_flatten(model.parameters())

    # Collect paths by role
    lm_head_paths = []
    embedding_paths = []
    value_embeds_paths = []
    resid_paths = []
    x0_paths = []
    muon_by_shape = {}  # shape -> list of paths

    for path, p in flat_params:
        if "blocks" in path and p.ndim == 2:
            s = p.shape
            muon_by_shape.setdefault(s, []).append(path)
        elif "lm_head" in path:
            lm_head_paths.append(path)
        elif "wte" in path:
            embedding_paths.append(path)
        elif "value_embeds" in path:
            value_embeds_paths.append(path)
        elif "resid_lambdas" in path:
            resid_paths.append(path)
        elif "x0_lambdas" in path:
            x0_paths.append(path)

    param_groups = []

    if lm_head_paths:
        param_groups.append(dict(
            kind="adamw", paths=lm_head_paths,
            lr=unembedding_lr * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if embedding_paths:
        param_groups.append(dict(
            kind="adamw", paths=embedding_paths,
            lr=embedding_lr * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if value_embeds_paths:
        param_groups.append(dict(
            kind="adamw", paths=value_embeds_paths,
            lr=embedding_lr * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if resid_paths:
        param_groups.append(dict(
            kind="adamw", paths=resid_paths,
            lr=scalar_lr * 0.01,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if x0_paths:
        param_groups.append(dict(
            kind="adamw", paths=x0_paths,
            lr=scalar_lr,
            betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0,
        ))

    # Muon groups: one per unique shape (for potential stacking in the future)
    for shape in sorted(muon_by_shape.keys()):
        param_groups.append(dict(
            kind="muon", paths=muon_by_shape[shape],
            lr=matrix_lr, momentum=0.95, ns_steps=5,
            beta2=0.95, weight_decay=weight_decay,
        ))

    optimizer = MuonAdamW(model, param_groups)
    return optimizer

# ---------------------------------------------------------------------------
# Training loop -- TO BE PORTED IN TASK 5
# ---------------------------------------------------------------------------
# The training loop (hyperparameters, setup, training) has been removed
# during the MLX port. It will be reimplemented in MLX in Task 5.

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GPTConfig(sequence_len=128, vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())  # materialize all lazy arrays

    x = mx.zeros((2, 128), dtype=mx.int32)
    logits = model(x)
    mx.eval(logits)  # force computation
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 128, 256), f"Expected (2, 128, 256), got {logits.shape}"
    print("Forward pass OK")

    # Test with targets (loss computation)
    targets = mx.zeros((2, 128), dtype=mx.int32)
    loss = model(x, targets=targets)
    mx.eval(loss)  # force computation
    print(f"Loss shape: {loss.shape}, value: {loss.item():.4f}")
    print("Loss computation OK")

    # Test parameter counting
    params = model.num_scaling_params()
    print(f"Parameter counts: {params}")
    print("Model smoke tests passed!")

    # ------------------------------------------------------------------
    # Optimizer smoke test (Task 4)
    # ------------------------------------------------------------------
    print("\n--- Optimizer smoke test ---")

    # Snapshot params before optimizer step
    params_before = {name: mx.array(p) for name, p in tree_flatten(model.parameters())}
    mx.eval(list(params_before.values()))

    # Create optimizer
    optimizer = setup_optimizer(model, weight_decay=0.1)

    # Print optimizer group summary
    for i, g in enumerate(optimizer.param_groups):
        print(f"  Group {i}: kind={g['kind']}, lr={g['lr']:.6f}, #paths={len(g['paths'])}")

    # Forward + backward
    x = mx.random.randint(0, 256, shape=(2, 128))
    targets = mx.random.randint(0, 256, shape=(2, 128))

    def loss_fn(model, x, targets):
        return model(x, targets=targets)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, targets)
    mx.eval(loss, grads)
    print(f"  Loss before step: {loss.item():.4f}")

    # Optimizer step
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    mx.eval(optimizer.state)

    # Verify parameters changed
    params_after = dict(tree_flatten(model.parameters()))
    n_changed = 0
    n_total = 0
    for name, p_before in params_before.items():
        p_after = params_after[name]
        n_total += 1
        if not mx.array_equal(p_before, p_after):
            n_changed += 1

    print(f"  Parameters changed: {n_changed}/{n_total}")
    assert n_changed > 0, "No parameters changed after optimizer step!"
    print("  Optimizer step verified: parameters updated correctly")

    # Verify scheduling helpers work
    optimizer.set_lr_multiplier(0.5)
    optimizer.set_muon_momentum(0.9)
    optimizer.set_muon_weight_decay(0.05)
    for g in optimizer.param_groups:
        if g["kind"] == "muon":
            assert g["momentum"] == 0.9, f"Expected momentum 0.9, got {g['momentum']}"
            assert g["weight_decay"] == 0.05, f"Expected wd 0.05, got {g['weight_decay']}"
    print("  Scheduling helpers verified")

    # Second step to verify state persistence works
    loss2, grads2 = nn.value_and_grad(model, loss_fn)(model, x, targets)
    mx.eval(loss2, grads2)
    optimizer.update(model, grads2)
    mx.eval(model.parameters())
    mx.eval(optimizer.state)
    print(f"  Loss after 2nd step: {loss2.item():.4f}")
    print("  Second optimizer step OK")

    print("\nAll smoke tests passed!")
