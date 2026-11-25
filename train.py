"""
Autoresearch-XLA pretraining script. Pure JAX, single-file, unified GPU & TPU.
Faithfully adapted from karpathy/autoresearch for Google Colab & Kaggle GPUs/TPUs
(supports T4, 2xT4, P100, L4, A100, H100, TPU v2-8, v3-8, v5e-1, v5e-8, v6e-1, and CPU).

THIS IS THE SINGLE FILE THE AUTONOMOUS RESEARCH AGENT EDITS.
Everything here is fair game: model architecture, optimizer, schedules, batch size,
model depth/width, attention patterns, precision, rematerialization, etc.

Usage:
    python train.py
"""

from __future__ import annotations

import functools
import os
import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from prepare import (
    JAX_CACHE_DIR,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    detect_hardware,
    evaluate_bpb,
    get_peak_memory_mb,
    make_dataloader,
)

# Enable persistent XLA compilation cache so identical HLO graphs recompile in <1s
os.makedirs(JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — `None` auto-scales to detected GPU/TPU)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = None              # Number of transformer layers (None -> auto: 4 on T4, 6 on L4/v5e-1, 8 on A100/H100/v5e-8)
ASPECT_RATIO = 64         # model_dim ≈ DEPTH * ASPECT_RATIO (rounded up to multiple of HEAD_DIM)
HEAD_DIM = 128            # Target attention head dimension
N_KV_HEADS = None         # None -> MHA (n_kv_head == n_head); set smaller divisor for GQA
WINDOW_PATTERN = None     # Sliding window pattern: "SSSL" (half/full) or "L" (full); None -> auto
USE_REMAT = None          # Activation checkpointing per block (True saves HBM/VRAM); None -> auto

# Optimization
TOTAL_BATCH_SIZE = None   # Tokens per optimizer step (None -> auto for hardware tier)
PER_DEVICE_BATCH = None   # Sequences per device per micro-step (None -> auto; reduce if OOM)
EMBEDDING_LR = 0.6        # Learning rate for token & value embeddings (AdamW)
UNEMBEDDING_LR = 0.004    # Learning rate for lm_head (AdamW)
MATRIX_LR = 0.04          # Learning rate for 2D transformer matrices (Muon)
SCALAR_LR = 0.5           # Learning rate for per-layer residual scalars (AdamW)
WEIGHT_DECAY = 0.2        # Cautious weight decay for Muon (decayed linearly to 0)
ADAM_BETAS = (0.8, 0.95)  # AdamW (beta1, beta2)
WARMUP_RATIO = 0.0        # Fraction of TIME_BUDGET for LR warmup
WARMDOWN_RATIO = 0.5      # Fraction of TIME_BUDGET for LR linear cooldown
FINAL_LR_FRAC = 0.0       # Final LR as fraction of initial LR
SOFTCAP = 15.0            # Logit softcapping threshold

# Runtime
SEED = 42
WARMUP_STEPS = 2          # Uncounted initial steps (step 0 runs XLA compilation)


# ---------------------------------------------------------------------------
# Model Config & Pure JAX GPT Architecture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    window_pattern: str = "SSSL"
    use_remat: bool = False
    softcap: float = 15.0


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating Value Embeddings (ResFormer); last layer is always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_window_sizes(config: GPTConfig) -> tuple[Optional[tuple[int, int]], ...]:
    """Return per-layer `local_window_size` arguments for causal attention (`None` = full)."""
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid WINDOW_PATTERN: {pattern!r}"
    short_window = max(1, config.sequence_len // 2)
    sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        sizes.append((short_window, 0) if char == "S" else None)
    sizes[-1] = None  # Final layer always attends over full context
    return tuple(sizes)


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 * inv_rms).astype(orig_dtype)


def precompute_rotary_embeddings(seq_len: int, head_dim: int, base: float = 10000.0):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs)[None, :, None, :]  # (1, T, 1, head_dim // 2)
    sin = jnp.sin(freqs)[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos_c = cos.astype(x.dtype)
    sin_c = sin.astype(x.dtype)
    y1 = x1 * cos_c + x2 * sin_c
    y2 = x1 * (-sin_c) + x2 * cos_c
    return jnp.concatenate([y1, y2], axis=-1)


def init_gpt_params(key: jax.Array, config: GPTConfig) -> dict:
    """Initialize all GPT parameters in float32 master precision (faithful to nanochat/autoresearch)."""
    n_embd = config.n_embd
    head_dim = n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim
    ve_gate_ch = min(32, n_embd)
    s = (3.0**0.5) * (n_embd**-0.5)

    def next_key():
        nonlocal key
        key, sub = jax.random.split(key)
        return sub

    blocks = []
    value_embeds = {}
    for i in range(config.n_layer):
        block = {
            "c_q": jax.random.uniform(next_key(), (n_embd, config.n_head * head_dim), jnp.float32, -s, s),
            "c_k": jax.random.uniform(next_key(), (n_embd, kv_dim), jnp.float32, -s, s),
            "c_v": jax.random.uniform(next_key(), (n_embd, kv_dim), jnp.float32, -s, s),
            "c_proj": jnp.zeros((n_embd, n_embd), dtype=jnp.float32),
            "c_fc": jax.random.uniform(next_key(), (n_embd, 4 * n_embd), jnp.float32, -s, s),
            "mlp_proj": jnp.zeros((4 * n_embd, n_embd), dtype=jnp.float32),
        }
        if has_ve(i, config.n_layer):
            # Gate weights init to zero -> 2 * sigmoid(0) == 1.0 (neutral start)
            block["ve_gate"] = jnp.zeros((ve_gate_ch, config.n_kv_head), dtype=jnp.float32)
            value_embeds[str(i)] = jax.random.uniform(
                next_key(), (config.vocab_size, kv_dim), jnp.float32, -s, s
            )
        blocks.append(block)

    return {
        "wte": jax.random.normal(next_key(), (config.vocab_size, n_embd), dtype=jnp.float32),
        "lm_head": jax.random.normal(next_key(), (n_embd, config.vocab_size), dtype=jnp.float32) * 0.001,
        "resid_lambdas": jnp.ones((config.n_layer,), dtype=jnp.float32),
        "x0_lambdas": jnp.full((config.n_layer,), 0.1, dtype=jnp.float32),
        "value_embeds": value_embeds,
        "blocks": blocks,
    }


def _causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    window_size: Optional[tuple[int, int]],
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """
    Unified causal (optionally sliding-window) dot-product attention for GPU, TPU, and CPU.
    Auto-dispatches to Pallas SplashAttention on TPU, cuDNN FlashAttention on Ampere+/Blackwell
    GPUs, and XLA dot_product_attention on T4/P100/CPU. Input/output layout: (B, T, N, H).
    """
    try:
        from scale.attention import fused_causal_attention

        return fused_causal_attention(
            q, k, v, window_size=window_size, backend="auto", splash_block_size=1024, mesh=mesh
        )
    except Exception:
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            local_window_size=window_size,
        )


def apply_block(
    block: dict,
    x: jax.Array,
    ve: Optional[jax.Array],
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    window_size: Optional[tuple[int, int]],
    compute_dtype,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    B, T, C = x.shape
    head_dim = C // config.n_head
    ve_gate_ch = min(32, C)

    # Attention sub-block
    h = rms_norm(x)
    q = jnp.dot(h, block["c_q"].astype(compute_dtype)).reshape(B, T, config.n_head, head_dim)
    k = jnp.dot(h, block["c_k"].astype(compute_dtype)).reshape(B, T, config.n_kv_head, head_dim)
    v = jnp.dot(h, block["c_v"].astype(compute_dtype)).reshape(B, T, config.n_kv_head, head_dim)

    if ve is not None:
        ve_h = ve.astype(compute_dtype).reshape(B, T, config.n_kv_head, head_dim)
        gate = 2.0 * jax.nn.sigmoid(jnp.dot(h[..., :ve_gate_ch], block["ve_gate"].astype(compute_dtype)))
        v = v + gate[..., None] * ve_h

    q = rms_norm(apply_rotary_emb(q, cos, sin))
    k = rms_norm(apply_rotary_emb(k, cos, sin))

    y = _causal_attention(q, k, v, window_size, mesh=mesh)
    y = y.reshape(B, T, C)
    x = x + jnp.dot(y, block["c_proj"].astype(compute_dtype))

    # MLP sub-block (ReLU^2)
    m = rms_norm(x)
    m = jnp.dot(m, block["c_fc"].astype(compute_dtype))
    m = jnp.square(jax.nn.relu(m))
    x = x + jnp.dot(m, block["mlp_proj"].astype(compute_dtype))
    return x


def gpt_forward_loss_2d(
    params: dict,
    idx: jax.Array,
    targets: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    config: GPTConfig,
    window_sizes: tuple[Optional[tuple[int, int]], ...],
    compute_dtype,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """Return per-token cross-entropy losses in nats of shape (B, T) in float32."""
    T = idx.shape[1]
    cos_t = cos[:, :T]
    sin_t = sin[:, :T]

    x = params["wte"][idx].astype(compute_dtype)
    x = rms_norm(x)
    x0 = x

    block_runner = jax.checkpoint(apply_block, static_argnums=(5, 6, 7, 8)) if config.use_remat else apply_block

    for i in range(config.n_layer):
        r_lam = params["resid_lambdas"][i].astype(compute_dtype)
        x0_lam = params["x0_lambdas"][i].astype(compute_dtype)
        x = r_lam * x + x0_lam * x0
        ve = params["value_embeds"][str(i)][idx] if str(i) in params["value_embeds"] else None
        x = block_runner(params["blocks"][i], x, ve, cos_t, sin_t, config, window_sizes[i], compute_dtype, mesh)

    x = rms_norm(x)
    logits = jnp.dot(x, params["lm_head"].astype(compute_dtype), preferred_element_type=jnp.float32)
    logits = config.softcap * jnp.tanh(logits / config.softcap)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return nll


def count_scaling_params(params: dict) -> dict[str, int]:
    wte = int(params["wte"].size)
    lm_head = int(params["lm_head"].size)
    value_embeds = sum(int(v.size) for v in params["value_embeds"].values())
    transformer_matrices = sum(int(a.size) for a in jax.tree_util.tree_leaves(params["blocks"]))
    scalars = int(params["resid_lambdas"].size + params["x0_lambdas"].size)
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    return {
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": total,
    }


def estimate_flops_per_token(params: dict, config: GPTConfig, window_sizes) -> float:
    """Estimated FLOPs per token (forward + backward), matching Karpathy's formula."""
    counts = count_scaling_params(params)
    nparams_active = counts["total"] - counts["wte"] - counts["value_embeds"] - counts["scalars"]
    h = config.n_head
    q = config.n_embd // config.n_head
    t = config.sequence_len
    attn_flops = 0.0
    for ws in window_sizes:
        eff_seq = t if ws is None else min(ws[0], t)
        attn_flops += 12.0 * h * q * eff_seq
    return 6.0 * nparams_active + attn_flops


# ---------------------------------------------------------------------------
# Optimizer: Polar-Express NorMuon (2D matrices) + AdamW (embeddings & scalars)
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def init_opt_state(params: dict) -> dict:
    """Create optimizer state matching params tree structure."""
    def adamw_zeros(arr):
        return {
            "m": jnp.zeros_like(arr, dtype=jnp.float32),
            "v": jnp.zeros_like(arr, dtype=jnp.float32),
        }

    block_states = []
    for block in params["blocks"]:
        b_state = {}
        for name, w in block.items():
            r, c = w.shape
            sec_shape = (r, 1) if r >= c else (1, c)
            b_state[name] = {
                "m": jnp.zeros_like(w, dtype=jnp.float32),
                "v_rowcol": jnp.zeros(sec_shape, dtype=jnp.float32),
            }
        block_states.append(b_state)

    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "wte": adamw_zeros(params["wte"]),
        "lm_head": adamw_zeros(params["lm_head"]),
        "resid_lambdas": adamw_zeros(params["resid_lambdas"]),
        "x0_lambdas": adamw_zeros(params["x0_lambdas"]),
        "value_embeds": {k: adamw_zeros(v) for k, v in params["value_embeds"].items()},
        "blocks": block_states,
    }


def _adamw_update(p, g, st, step_f, lr, beta1, beta2, eps=1e-10, wd=0.0):
    p = p * (1.0 - lr * wd)
    m = st["m"] * beta1 + g * (1.0 - beta1)
    v = st["v"] * beta2 + (g * g) * (1.0 - beta2)
    bias1 = 1.0 - (beta1 ** step_f)
    bias2 = 1.0 - (beta2 ** step_f)
    denom = jnp.sqrt(v / bias2) + eps
    p_new = p - (lr / bias1) * (m / denom)
    return p_new, {"m": m, "v": v}


def _muon_update(p, g, st, lr, momentum, wd, ns_dtype, beta2=0.95, ns_steps=5):
    """
    Polar-Express Muon with NorMuon variance reduction & cautious weight decay.
    Note: Linear weights here have shape (in_features, out_features), so the
    Karpathy aspect-ratio scale `max(1, out/in)^0.5` uses `shape[-1] / shape[-2]`.
    """
    # 1. Nesterov momentum
    m = st["m"] * momentum + g * (1.0 - momentum)
    u = g * (1.0 - momentum) + m * momentum

    # 2. Polar Express orthogonalization (bf16 on bf16 hardware, fp32 on fp16/cpu to avoid overflow)
    X = u.astype(ns_dtype)
    norm = jnp.linalg.norm(X.astype(jnp.float32))
    X = X / (norm.astype(ns_dtype) * jnp.asarray(1.02, ns_dtype) + jnp.asarray(1e-6, ns_dtype))

    rows, cols = p.shape
    if rows > cols:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    g_orth = X.astype(jnp.float32)

    # 3. NorMuon factored second-moment variance reduction
    red_axis = -1 if rows >= cols else -2
    red_size = float(cols if rows >= cols else rows)
    v_mean = jnp.mean(g_orth * g_orth, axis=red_axis, keepdims=True)
    v_norm = jnp.sqrt(jnp.sum(v_mean) * red_size)

    v_buf = st["v_rowcol"] * beta2 + v_mean * (1.0 - beta2)
    step_scale = jax.lax.rsqrt(jnp.maximum(v_buf, 1e-10))
    v_norm_new = jnp.sqrt(jnp.sum((v_mean * red_size) * (step_scale * step_scale)))
    final_scale = step_scale * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    g_orth = g_orth * final_scale

    # 4. Aspect-ratio LR scale + cautious weight decay
    eff_lr = lr * (max(1.0, cols / rows) ** 0.5)
    mask = ((g_orth * p) >= 0.0).astype(jnp.float32)
    p_new = p - eff_lr * g_orth - eff_lr * wd * p * mask
    return p_new, {"m": m, "v_rowcol": v_buf}


def optimizer_step(
    params: dict,
    grads: dict,
    opt_state: dict,
    lrm: jax.Array,
    muon_momentum: jax.Array,
    muon_wd: jax.Array,
    dmodel_lr_scale: float,
    ns_dtype,
):
    step_i = opt_state["step"] + jnp.int32(1)
    step_f = step_i.astype(jnp.float32)
    b1, b2 = ADAM_BETAS

    new_wte, st_wte = _adamw_update(
        params["wte"], grads["wte"], opt_state["wte"], step_f,
        EMBEDDING_LR * dmodel_lr_scale * lrm, b1, b2,
    )
    new_lm, st_lm = _adamw_update(
        params["lm_head"], grads["lm_head"], opt_state["lm_head"], step_f,
        UNEMBEDDING_LR * dmodel_lr_scale * lrm, b1, b2,
    )
    new_resid, st_resid = _adamw_update(
        params["resid_lambdas"], grads["resid_lambdas"], opt_state["resid_lambdas"], step_f,
        SCALAR_LR * 0.01 * lrm, b1, b2,
    )
    new_x0, st_x0 = _adamw_update(
        params["x0_lambdas"], grads["x0_lambdas"], opt_state["x0_lambdas"], step_f,
        SCALAR_LR * lrm, 0.96, 0.95,
    )

    new_ve, st_ve = {}, {}
    for k in params["value_embeds"]:
        new_ve[k], st_ve[k] = _adamw_update(
            params["value_embeds"][k], grads["value_embeds"][k], opt_state["value_embeds"][k],
            step_f, EMBEDDING_LR * dmodel_lr_scale * lrm, b1, b2,
        )

    new_blocks, st_blocks = [], []
    for blk_p, blk_g, blk_s in zip(params["blocks"], grads["blocks"], opt_state["blocks"]):
        nb, ns = {}, {}
        for name in blk_p:
            nb[name], ns[name] = _muon_update(
                blk_p[name], blk_g[name], blk_s[name],
                MATRIX_LR * lrm, muon_momentum, muon_wd, ns_dtype,
            )
        new_blocks.append(nb)
        st_blocks.append(ns)

    new_params = {
        "wte": new_wte,
        "lm_head": new_lm,
        "resid_lambdas": new_resid,
        "x0_lambdas": new_x0,
        "value_embeds": new_ve,
        "blocks": new_blocks,
    }
    new_state = {
        "step": step_i,
        "wte": st_wte,
        "lm_head": st_lm,
        "resid_lambdas": st_resid,
        "x0_lambdas": st_x0,
        "value_embeds": st_ve,
        "blocks": st_blocks,
    }
    return new_params, new_state


# ---------------------------------------------------------------------------
# Schedules (based on progress = total_training_time / TIME_BUDGET)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress: float) -> float:
    if WARMUP_RATIO > 0.0 and progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = max(0.0, (1.0 - progress) / max(WARMDOWN_RATIO, 1e-8))
    return cooldown * 1.0 + (1.0 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300.0, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float) -> float:
    return WEIGHT_DECAY * max(0.0, 1.0 - progress)


# ---------------------------------------------------------------------------
# Main Training & Evaluation Entry Point
# ---------------------------------------------------------------------------

def build_model_config(depth: int, vocab_size: int, window_pattern: str, use_remat: bool) -> GPTConfig:
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = max(1, model_dim // HEAD_DIM)
    num_kv_heads = num_heads if N_KV_HEADS is None else N_KV_HEADS
    assert num_heads % num_kv_heads == 0, f"n_head ({num_heads}) must be divisible by n_kv_head ({num_kv_heads})"
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
        use_remat=use_remat,
        softcap=SOFTCAP,
    )


def main() -> None:
    t_start = time.time()

    # 1. Detect GPU / TPU / CPU and resolve auto-scaled defaults
    hw = detect_hardware()
    depth = hw.default_depth if DEPTH is None else DEPTH
    window_pattern = hw.default_window_pattern if WINDOW_PATTERN is None else WINDOW_PATTERN
    use_remat = hw.default_remat if USE_REMAT is None else USE_REMAT
    per_device_batch = hw.default_per_device_batch if PER_DEVICE_BATCH is None else PER_DEVICE_BATCH
    global_micro_batch = per_device_batch * hw.num_devices
    tokens_per_micro = global_micro_batch * MAX_SEQ_LEN

    total_batch_size = hw.default_total_batch_size if TOTAL_BATCH_SIZE is None else TOTAL_BATCH_SIZE
    if total_batch_size < tokens_per_micro:
        total_batch_size = tokens_per_micro
    assert total_batch_size % tokens_per_micro == 0, (
        f"TOTAL_BATCH_SIZE ({total_batch_size}) must be divisible by "
        f"per_device_batch*num_devices*MAX_SEQ_LEN ({tokens_per_micro})"
    )
    grad_accum_steps = total_batch_size // tokens_per_micro

    dtype_map = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}
    compute_dtype = dtype_map[hw.compute_dtype]
    use_loss_scale = compute_dtype == jnp.float16
    # Polar Express orthogonalization is stable in bf16 on bf16 hardware; use fp32 on fp16/cpu
    ns_dtype = jnp.bfloat16 if compute_dtype == jnp.bfloat16 else jnp.float32

    print(
        f"Hardware: {hw.device_kind} x{hw.num_devices} "
        f"(platform={hw.platform}, dtype={hw.compute_dtype}, peak={hw.peak_tflops_total:.1f} TFLOPS)"
    )

    # 2. Tokenizer & Model Config
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = build_model_config(depth, vocab_size, window_pattern, use_remat)
    window_sizes = compute_window_sizes(config)
    print(f"Vocab size: {vocab_size:,}")
    print(f"Model config: {asdict(config)}")

    # 3. Automatic Multi-Chip SPMD Data-Parallel Mesh (works for 1, 2, 4, or 8 devices!)
    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("data",))
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))
    step_sharding = NamedSharding(mesh, P(None, "data", None))

    # 4. Initialize parameters, rotary embeddings, and optimizer state (replicated across chips)
    params = init_gpt_params(jax.random.PRNGKey(SEED), config)
    opt_state = init_opt_state(params)
    cos, sin = precompute_rotary_embeddings(MAX_SEQ_LEN, config.n_embd // config.n_head)

    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)
    cos = jax.device_put(cos, replicated)
    sin = jax.device_put(sin, replicated)

    param_counts = count_scaling_params(params)
    print("Parameter counts:")
    for k, v in param_counts.items():
        print(f"  {k:24s}: {v:,}")
    num_params = param_counts["total"]
    num_flops_per_token = estimate_flops_per_token(params, config, window_sizes)
    dmodel_lr_scale = (config.n_embd / 768.0) ** -0.5

    print(f"Estimated FLOPs per token: {num_flops_per_token:.3e}")
    print(f"Scaling AdamW LRs by 1/sqrt({config.n_embd}/768) = {dmodel_lr_scale:.6f}")
    print(
        f"Batching: total_batch={total_batch_size:,} tok | "
        f"per_device={per_device_batch} x {hw.num_devices} dev | grad_accum={grad_accum_steps}"
    )
    print(f"Time budget: {TIME_BUDGET}s (excluding XLA compile & warmup)")

    # 5. Fused JIT-Compiled Training Step (lax.scan over micro-steps + SPMD all-reduce + MuonAdamW)
    def micro_loss(p, x, y, scale):
        nll = gpt_forward_loss_2d(p, x, y, cos, sin, config, window_sizes, compute_dtype, mesh=mesh)
        loss = jnp.mean(nll, dtype=jnp.float32)
        return loss * scale, loss

    grad_fn = jax.value_and_grad(micro_loss, has_aux=True)

    @functools.partial(
        jax.jit,
        donate_argnums=(0, 1),
        in_shardings=(
            replicated,
            replicated,
            step_sharding,
            step_sharding,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_shardings=(replicated, replicated, replicated, replicated),
    )
    def train_step(p, st, x_steps, y_steps, lrm, muon_mom, muon_wd, loss_scale):
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, p)

        def accum_body(carry, batch):
            g_acc, loss_acc = carry
            xb, yb = batch
            (_, loss_val), g_micro = grad_fn(p, xb, yb, loss_scale)
            g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g_micro)
            return (g_acc, loss_acc + loss_val), None

        (g_sum, loss_sum), _ = jax.lax.scan(accum_body, (zero_grads, jnp.float32(0.0)), (x_steps, y_steps))
        inv_denom = jnp.float32(1.0 / grad_accum_steps) / loss_scale
        grads = jax.tree_util.tree_map(lambda g: g * inv_denom, g_sum)
        mean_loss = loss_sum * jnp.float32(1.0 / grad_accum_steps)

        if use_loss_scale:
            leaves = jax.tree_util.tree_leaves(grads)
            all_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in leaves]))
            p_cand, st_cand = optimizer_step(p, grads, st, lrm, muon_mom, muon_wd, dmodel_lr_scale, ns_dtype)
            p_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), p_cand, p)
            st_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), st_cand, st)
            return p_next, st_next, mean_loss, all_finite
        else:
            p_next, st_next = optimizer_step(p, grads, st, lrm, muon_mom, muon_wd, dmodel_lr_scale, ns_dtype)
            return p_next, st_next, mean_loss, jnp.bool_(True)

    # 6. Training Loop
    train_loader = make_dataloader(
        tokenizer,
        global_micro_batch,
        MAX_SEQ_LEN,
        "train",
        prefetch=2,
        grad_accum_steps=grad_accum_steps,
    )

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    timed_steps = 0
    step = 0
    loss_scale = 32768.0 if use_loss_scale else 1.0
    good_scale_steps = 0
    peak_flops = hw.peak_tflops_total * 1e12

    try:
        x_np, y_np, epoch = next(train_loader)
        while True:
            t0 = time.perf_counter()

            # Stage whole optimizer step (grad_accum_steps, global_micro_batch, T) onto devices
            if x_np.ndim == 2:
                x_np = x_np[None, :, :]
                y_np = y_np[None, :, :]
            x_dev = jax.device_put(x_np, step_sharding)
            y_dev = jax.device_put(y_np, step_sharding)

            progress = min(total_training_time / max(TIME_BUDGET, 1e-6), 1.0)
            lrm = get_lr_multiplier(progress)
            muon_mom = get_muon_momentum(step)
            muon_wd = get_weight_decay(progress)

            params, opt_state, loss_arr, grads_finite = train_step(
                params,
                opt_state,
                x_dev,
                y_dev,
                jax.device_put(jnp.float32(lrm), replicated),
                jax.device_put(jnp.float32(muon_mom), replicated),
                jax.device_put(jnp.float32(muon_wd), replicated),
                jax.device_put(jnp.float32(loss_scale), replicated),
            )

            # Overlap next host prefetch while accelerator runs
            x_np, y_np, epoch = next(train_loader)

            # Synchronize accurately for wall-clock timing
            train_loss_f = float(jax.block_until_ready(loss_arr))
            finite_ok = bool(grads_finite)
            dt = time.perf_counter() - t0

            if use_loss_scale:
                if finite_ok:
                    good_scale_steps += 1
                    if good_scale_steps >= 200:
                        loss_scale = min(loss_scale * 2.0, 2.0**24)
                        good_scale_steps = 0
                else:
                    loss_scale = max(loss_scale * 0.5, 1.0)
                    good_scale_steps = 0

            # Fast fail on divergence / NaN
            if math.isnan(train_loss_f) or math.isinf(train_loss_f) or train_loss_f > 100.0:
                print(f"\nFAIL: loss diverged ({train_loss_f}) at step {step}")
                raise SystemExit(1)

            if step >= WARMUP_STEPS:
                total_training_time += dt
                timed_steps += 1

            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1.0 - ema_beta) * train_loss_f
            debiased_loss = smooth_train_loss / (1.0 - ema_beta ** (step + 1))
            pct_done = 100.0 * min(total_training_time / max(TIME_BUDGET, 1e-6), 1.0)
            tok_per_sec = int(total_batch_size / max(dt, 1e-6))
            mfu = 100.0 * num_flops_per_token * total_batch_size / max(dt, 1e-6) / max(peak_flops, 1.0)
            remaining = max(0.0, TIME_BUDGET - total_training_time)
            tag = "compile" if step == 0 else ("warmup " if step < WARMUP_STEPS else f"{pct_done:5.1f}%")

            print(
                f"\rstep {step:05d} ({tag}) | loss: {debiased_loss:.6f} | lrm: {lrm:.2f} | "
                f"dt: {dt*1000:6.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:4.1f}% | "
                f"ep: {epoch} | rem: {remaining:4.0f}s    ",
                end="",
                flush=True,
            )

            if step == WARMUP_STEPS:
                gc.collect()
                gc.disable()
            elif (step + 1) % 2000 == 0:
                gc.collect()

            step += 1
            if step >= WARMUP_STEPS and total_training_time >= TIME_BUDGET:
                break
    finally:
        gc.enable()
        train_loader.close()

    print()

    # 7. Final Validation Evaluation (immutable evaluate_bpb from prepare.py)
    def eval_loss_fn(p, x, y):
        return gpt_forward_loss_2d(p, x, y, cos, sin, config, window_sizes, compute_dtype, mesh=mesh)

    val_bpb = evaluate_bpb(
        eval_loss_fn,
        params,
        tokenizer,
        global_micro_batch,
        data_sharding=batch_sharding,
    )

    # 8. Summary (exact Karpathy-compatible output format)
    t_end = time.time()
    total_tokens = step * total_batch_size
    steady_state_mfu = (
        100.0 * num_flops_per_token * total_batch_size * timed_steps / total_training_time / peak_flops
        if total_training_time > 0 and peak_flops > 0
        else 0.0
    )
    peak_mem_mb = get_peak_memory_mb()

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_mem_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {depth}")
    print(f"accelerator:      {hw.device_kind} x{hw.num_devices} ({hw.platform}, {hw.compute_dtype})")


if __name__ == "__main__":
    main()
