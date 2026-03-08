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
# Optimizer (MuonAdamW) -- TO BE PORTED IN TASK 4
# ---------------------------------------------------------------------------
# The optimizer code (MuonAdamW, polar_express_coeffs, adamw_step_fused,
# muon_step_fused) has been removed during the MLX port. It will be
# reimplemented in MLX in Task 4.

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
    print("All smoke tests passed!")
