"""
Scanned Pure-JAX Transformer for 100M -> 1B+ FSDP pretraining.

Key design highlights:
- Uses `jax.lax.scan` over transformer layers (or window-pattern groups) with per-layer
  `jax.checkpoint` so a 20-layer 1B model compiles in ~15-20s with O(1) activation depth.
- Supports all architecture knobs explored by `autoresearch-xla`:
  SwiGLU / ReLU^2 / GeLU, GQA, QK-Norm, RoPE, ResFormer Value Embeddings,
  per-layer resid/x0 lambdas, SSSL/SL/L sliding windows, logit softcapping,
  auxiliary z-loss, and optional tied embeddings.
- Clean zero-memory abstract shape init (`jax.eval_shape`) so parameters can be
  created *directly sharded* across 8 TPU chips via `jax.jit(..., out_shardings=...)`.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp

from scale.config import ModelSpec


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
    return jnp.concatenate([x1 * cos_c + x2 * sin_c, x1 * (-sin_c) + x2 * cos_c], axis=-1)


def init_params(key: jax.Array, spec: ModelSpec) -> dict:
    """
    Initialize all model parameters in float32 master precision with stacked
    `(n_layer, ...)` block tensors for fast `lax.scan` compilation & clean FSDP sharding.
    """
    L = spec.n_layer
    C = spec.n_embd
    kv_dim = spec.kv_dim
    M = spec.mlp_hidden
    ve_ch = min(32, C)
    s = (3.0**0.5) * (C**-0.5)

    def next_key():
        nonlocal key
        key, sub = jax.random.split(key)
        return sub

    blocks: dict[str, jax.Array] = {
        "c_q": jax.random.uniform(next_key(), (L, C, C), jnp.float32, -s, s),
        "c_k": jax.random.uniform(next_key(), (L, C, kv_dim), jnp.float32, -s, s),
        "c_v": jax.random.uniform(next_key(), (L, C, kv_dim), jnp.float32, -s, s),
        "c_proj": jnp.zeros((L, C, C), dtype=jnp.float32),
        "mlp_down": jnp.zeros((L, M, C), dtype=jnp.float32),
    }
    if spec.activation == "swiglu":
        blocks["mlp_gate"] = jax.random.uniform(next_key(), (L, C, M), jnp.float32, -s, s)
        blocks["mlp_up"] = jax.random.uniform(next_key(), (L, C, M), jnp.float32, -s, s)
    else:
        blocks["mlp_fc"] = jax.random.uniform(next_key(), (L, C, M), jnp.float32, -s, s)

    if spec.use_value_embeds:
        blocks["ve_gate"] = jnp.zeros((L, ve_ch, spec.n_kv_head), dtype=jnp.float32)

    params: dict = {
        "wte": jax.random.normal(next_key(), (spec.vocab_size, C), dtype=jnp.float32) * (C**-0.5 if spec.tie_embeddings else 1.0),
        "blocks": blocks,
        "resid_lambdas": jnp.ones((L,), dtype=jnp.float32),
        "x0_lambdas": jnp.full((L,), 0.1 if spec.use_resid_lambdas else 0.0, dtype=jnp.float32),
    }
    if not spec.tie_embeddings:
        params["lm_head"] = jax.random.normal(next_key(), (C, spec.vocab_size), dtype=jnp.float32) * 0.001
    if spec.use_value_embeds:
        params["value_embeds"] = jax.random.uniform(
            next_key(), (L, spec.vocab_size, kv_dim), jnp.float32, -s, s
        )
    return params


def _apply_single_layer(
    blk: dict,
    resid_lam: jax.Array,
    x0_lam: jax.Array,
    x: jax.Array,
    x0: jax.Array,
    ve_tok: Optional[jax.Array],
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    window_size: Optional[tuple[int, int]],
    compute_dtype,
) -> jax.Array:
    B, T, C = x.shape
    head_dim = spec.head_dim
    ve_ch = min(32, C)

    if spec.use_resid_lambdas:
        x = resid_lam.astype(compute_dtype) * x + x0_lam.astype(compute_dtype) * x0

    # Attention sub-block
    h = rms_norm(x)
    q = jnp.dot(h, blk["c_q"].astype(compute_dtype)).reshape(B, T, spec.n_head, head_dim)
    k = jnp.dot(h, blk["c_k"].astype(compute_dtype)).reshape(B, T, spec.n_kv_head, head_dim)
    v = jnp.dot(h, blk["c_v"].astype(compute_dtype)).reshape(B, T, spec.n_kv_head, head_dim)

    if ve_tok is not None and "ve_gate" in blk:
        ve_h = ve_tok.astype(compute_dtype).reshape(B, T, spec.n_kv_head, head_dim)
        gate = 2.0 * jax.nn.sigmoid(jnp.dot(h[..., :ve_ch], blk["ve_gate"].astype(compute_dtype)))
        v = v + gate[..., None] * ve_h

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    if spec.use_qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)

    y = jax.nn.dot_product_attention(q, k, v, is_causal=True, local_window_size=window_size)
    y = y.reshape(B, T, C)
    x = x + jnp.dot(y, blk["c_proj"].astype(compute_dtype))

    # MLP sub-block
    m = rms_norm(x)
    if spec.activation == "swiglu":
        gate_out = jax.nn.silu(jnp.dot(m, blk["mlp_gate"].astype(compute_dtype)))
        up_out = jnp.dot(m, blk["mlp_up"].astype(compute_dtype))
        m = gate_out * up_out
    elif spec.activation == "relu2":
        m = jnp.square(jax.nn.relu(jnp.dot(m, blk["mlp_fc"].astype(compute_dtype))))
    else:  # "gelu"
        m = jax.nn.gelu(jnp.dot(m, blk["mlp_fc"].astype(compute_dtype)), approximate=True)

    x = x + jnp.dot(m, blk["mlp_down"].astype(compute_dtype))
    return x


def _make_layer_fn(spec: ModelSpec, window_size: Optional[tuple[int, int]], compute_dtype):
    fn = functools.partial(
        _apply_single_layer,
        spec=spec,
        window_size=window_size,
        compute_dtype=compute_dtype,
    )
    return jax.checkpoint(fn) if spec.use_remat else fn


def forward_logits(
    params: dict,
    idx: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
) -> jax.Array:
    """Compute float32 (optionally softcapped) logits of shape (B, T, vocab_size)."""
    T = idx.shape[1]
    cos_t = cos[:, :T]
    sin_t = sin[:, :T]

    x = params["wte"][idx].astype(compute_dtype)
    x = rms_norm(x)
    x0 = x

    pattern = spec.window_pattern.upper()
    G = len(pattern)
    short_win = (max(1, spec.sequence_len // 2), 0)
    ve_table = params.get("value_embeds", None)

    # Fast path 1: uniform full causal ("L") -> single lax.scan across all L layers!
    if pattern == "L":
        layer_fn = _make_layer_fn(spec, None, compute_dtype)

        def scan_body(carry_x, layer_inputs):
            blk, r_lam, x0_lam, ve_w = layer_inputs
            ve_tok = ve_w[idx] if ve_w is not None else None
            next_x = layer_fn(blk, r_lam, x0_lam, carry_x, x0, ve_tok, cos_t, sin_t)
            return next_x, None

        xs = (
            params["blocks"],
            params["resid_lambdas"],
            params["x0_lambdas"],
            ve_table if ve_table is not None else None,
        )
        # Handle optional ve_table cleanly in scan
        if ve_table is not None:
            x, _ = jax.lax.scan(scan_body, x, xs)
        else:
            def scan_body_nove(carry_x, layer_inputs):
                blk, r_lam, x0_lam = layer_inputs
                next_x = layer_fn(blk, r_lam, x0_lam, carry_x, x0, None, cos_t, sin_t)
                return next_x, None

            x, _ = jax.lax.scan(scan_body_nove, x, (params["blocks"], params["resid_lambdas"], params["x0_lambdas"]))

    # Fast path 2: repeating window group (e.g. "SL" or "SSSL") with n_layer % G == 0
    elif spec.n_layer % G == 0:
        n_groups = spec.n_layer // G
        group_fns = [
            _make_layer_fn(spec, short_win if pattern[g] == "S" else None, compute_dtype)
            for g in range(G)
        ]

        def reshape_group(arr):
            return arr.reshape((n_groups, G) + arr.shape[1:])

        grouped_blocks = jax.tree_util.tree_map(reshape_group, params["blocks"])
        grouped_r = reshape_group(params["resid_lambdas"])
        grouped_x0 = reshape_group(params["x0_lambdas"])
        grouped_ve = reshape_group(ve_table) if ve_table is not None else None

        if grouped_ve is not None:
            def group_body_ve(carry_x, grp):
                g_blk, g_r, g_x0, g_ve = grp
                cur = carry_x
                for g in range(G):
                    blk_g = jax.tree_util.tree_map(lambda a: a[g], g_blk)
                    cur = group_fns[g](blk_g, g_r[g], g_x0[g], cur, x0, g_ve[g][idx], cos_t, sin_t)
                return cur, None

            x, _ = jax.lax.scan(group_body_ve, x, (grouped_blocks, grouped_r, grouped_x0, grouped_ve))
        else:
            def group_body_nove(carry_x, grp):
                g_blk, g_r, g_x0 = grp
                cur = carry_x
                for g in range(G):
                    blk_g = jax.tree_util.tree_map(lambda a: a[g], g_blk)
                    cur = group_fns[g](blk_g, g_r[g], g_x0[g], cur, x0, None, cos_t, sin_t)
                return cur, None

            x, _ = jax.lax.scan(group_body_nove, x, (grouped_blocks, grouped_r, grouped_x0))

    # Fallback path: unrolled per-layer checkpointed loop
    else:
        for i in range(spec.n_layer):
            char = "L" if i == spec.n_layer - 1 else pattern[i % G]
            win = short_win if char == "S" else None
            layer_fn = _make_layer_fn(spec, win, compute_dtype)
            blk_i = jax.tree_util.tree_map(lambda a: a[i], params["blocks"])
            ve_tok = ve_table[i][idx] if ve_table is not None else None
            x = layer_fn(blk_i, params["resid_lambdas"][i], params["x0_lambdas"][i], x, x0, ve_tok, cos_t, sin_t)

    x = rms_norm(x)
    if spec.tie_embeddings:
        lm_w = params["wte"].T.astype(compute_dtype)
    else:
        lm_w = params["lm_head"].astype(compute_dtype)

    logits = jnp.dot(x, lm_w).astype(jnp.float32)
    if spec.softcap > 0.0:
        logits = spec.softcap * jnp.tanh(logits / spec.softcap)
    return logits


def forward_loss_2d(
    params: dict,
    idx: jax.Array,
    targets: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    return_z_loss: bool = False,
):
    """Return exact per-token NLL (B, T) in float32, plus optional mean z-loss."""
    logits = forward_logits(params, idx, cos, sin, spec, compute_dtype)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    nll = log_z - target_logits
    if return_z_loss:
        z_loss = jnp.mean(log_z * log_z, dtype=jnp.float32)
        return nll, z_loss
    return nll


def count_params(params: dict) -> dict[str, int]:
    wte = int(params["wte"].size)
    lm_head = int(params["lm_head"].size) if "lm_head" in params else 0
    ve = int(params["value_embeds"].size) if "value_embeds" in params else 0
    blocks = sum(int(a.size) for a in jax.tree_util.tree_leaves(params["blocks"]))
    scalars = int(params["resid_lambdas"].size + params["x0_lambdas"].size)
    total = wte + lm_head + ve + blocks + scalars
    return {
        "wte": wte,
        "lm_head": lm_head,
        "value_embeds": ve,
        "transformer_matrices": blocks,
        "scalars": scalars,
        "total": total,
    }


def estimate_flops_per_token(params: dict, spec: ModelSpec) -> float:
    """Estimated forward + backward FLOPs per token."""
    counts = count_params(params)
    active_non_emb = counts["total"] - counts["wte"] - counts["value_embeds"] - counts["scalars"]
    if spec.tie_embeddings:
        active_non_emb += counts["wte"]  # lm_head matmul still runs
    pattern = spec.window_pattern.upper()
    t = spec.sequence_len
    attn_flops = 0.0
    for i in range(spec.n_layer):
        ch = "L" if i == spec.n_layer - 1 else pattern[i % len(pattern)]
        eff_seq = t if ch == "L" else t // 2
        attn_flops += 12.0 * spec.n_head * spec.head_dim * eff_seq
    return 6.0 * active_non_emb + attn_flops


def generate_tokens(
    params: dict,
    prompt_ids: list[int],
    max_new_tokens: int,
    spec: ModelSpec,
    compute_dtype,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
    batch_sharding=None,
) -> list[int]:
    """
    Simple, readable autoregressive sampler for live qualitative checks during 1B pretraining.
    Works seamlessly even when `params` are FSDP-sharded across 8 TPU chips!
    """
    cos, sin = precompute_rotary_embeddings(spec.sequence_len, spec.head_dim, base=spec.rope_base)
    if batch_sharding is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P

        rep = NamedSharding(batch_sharding.mesh, P())
        cos = jax.device_put(cos, rep)
        sin = jax.device_put(sin, rep)

    @jax.jit
    def next_token_logits(p, x_pad, pos):
        logits = forward_logits(p, x_pad, cos, sin, spec, compute_dtype)
        return logits[0, pos]

    # Fixed padded length so XLA compiles next_token_logits only once per sample call
    cap_len = min(spec.sequence_len, max(len(prompt_ids) + max_new_tokens, 64))
    # Round up to multiple of 64 for clean TPU shapes
    cap_len = min(spec.sequence_len, ((cap_len + 63) // 64) * 64)
    tokens = list(prompt_ids[-cap_len + 1 :]) if len(prompt_ids) >= cap_len else list(prompt_ids)
    key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        if len(tokens) >= cap_len:
            break
        cur_pos = len(tokens) - 1
        buf = jnp.zeros((1, cap_len), dtype=jnp.int32).at[0, : len(tokens)].set(jnp.asarray(tokens, dtype=jnp.int32))
        if batch_sharding is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            buf = jax.device_put(buf, NamedSharding(batch_sharding.mesh, P()))
        logits = next_token_logits(params, buf, jnp.int32(cur_pos))
        if temperature <= 0.0:
            next_id = int(jnp.argmax(logits))
        else:
            scaled = logits / temperature
            if top_k > 0 and top_k < scaled.shape[0]:
                top_vals, top_idx = jax.lax.top_k(scaled, top_k)
                key, sub = jax.random.split(key)
                pick = int(jax.random.categorical(sub, top_vals))
                next_id = int(top_idx[pick])
            else:
                key, sub = jax.random.split(key)
                next_id = int(jax.random.categorical(sub, scaled))
        tokens.append(next_id)

    return tokens
