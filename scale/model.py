"""
High-MFU Scanned Pure-JAX Transformer for 100M -> 1B+ FSDP pretraining.

MFU optimizations built in:
1. Fused `w_qkv` (C -> C + 2*kv_dim) and fused SwiGLU `w_gate_up` (C -> 2*M) matmuls
   to saturate TPU v5e/v6e MXUs and Blackwell/Hopper/Ampere Tensor Cores.
2. Automatic fused attention via `scale.attention.fused_causal_attention`:
   - TPU: Pallas SplashAttention (on-chip VMEM tiled, shard_map multi-chip FSDP aware)
   - Ampere+/Blackwell GPU (L4, A100, H100, RTX PRO 6000 G4): cuDNN FlashAttention
   - T4 / P100 / CPU: XLA dot_product_attention
3. Selective rematerialization (`remat_policy`): `"none"` (0% recompute!),
   `"dots_with_no_batch_dims"` (saves linear dots, recomputes cheap elementwise ops),
   `"dots"`, or `"full"`.
4. Chunked linear cross-entropy (`loss_chunk_size=2048`): scans over token chunks
   computing `x_c @ lm_head` + softcap + logsumexp + NLL + z-loss, cutting peak logits
   HBM from ~4.3 GB down to ~268 MB so much larger microbatches fit.
5. `jax.lax.scan` over layers (or sliding-window groups) + pre-cast RoPE `cos`/`sin`.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from scale.attention import fused_causal_attention
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
    return jnp.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


def init_params(key: jax.Array, spec: ModelSpec) -> dict:
    """
    Initialize parameters in float32 master precision with fused `(n_layer, in, out)`
    tensors for peak MXU/Tensor-Core arithmetic intensity and clean FSDP sharding.
    """
    L = spec.n_layer
    C = spec.n_embd
    kv_dim = spec.kv_dim
    qkv_dim = C + 2 * kv_dim
    M = spec.mlp_hidden
    ve_ch = min(32, C)
    s = (3.0**0.5) * (C**-0.5)

    def next_key():
        nonlocal key
        key, sub = jax.random.split(key)
        return sub

    blocks: dict[str, jax.Array] = {
        "w_qkv": jax.random.uniform(next_key(), (L, C, qkv_dim), jnp.float32, -s, s),
        "c_proj": jnp.zeros((L, C, C), dtype=jnp.float32),
        "mlp_down": jnp.zeros((L, M, C), dtype=jnp.float32),
    }
    if spec.activation == "swiglu":
        blocks["w_gate_up"] = jax.random.uniform(next_key(), (L, C, 2 * M), jnp.float32, -s, s)
    else:
        blocks["mlp_fc"] = jax.random.uniform(next_key(), (L, C, M), jnp.float32, -s, s)

    if spec.use_value_embeds:
        blocks["ve_gate"] = jnp.zeros((L, ve_ch, spec.n_kv_head), dtype=jnp.float32)

    params: dict = {
        "wte": jax.random.normal(next_key(), (spec.vocab_size, C), dtype=jnp.float32)
        * (C**-0.5 if spec.tie_embeddings else 1.0),
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


def _get_checkpoint_policy(remat_policy: str):
    if remat_policy == "none":
        return None
    if remat_policy == "dots_with_no_batch_dims":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    if remat_policy == "dots":
        return jax.checkpoint_policies.dots_saveable
    return jax.checkpoint_policies.nothing_saveable


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
    mesh: Optional[Mesh],
) -> jax.Array:
    B, T, C = x.shape
    head_dim = spec.head_dim
    kv_dim = spec.kv_dim
    ve_ch = min(32, C)

    if spec.use_resid_lambdas:
        x = resid_lam.astype(compute_dtype) * x + x0_lam.astype(compute_dtype) * x0

    # --- Attention sub-block (Fused QKV projection) ---
    h = rms_norm(x)
    qkv = jnp.dot(h.reshape(B * T, C), blk["w_qkv"].astype(compute_dtype))
    q = qkv[:, :C].reshape(B, T, spec.n_head, head_dim)
    k = qkv[:, C : C + kv_dim].reshape(B, T, spec.n_kv_head, head_dim)
    v = qkv[:, C + kv_dim :].reshape(B, T, spec.n_kv_head, head_dim)

    if ve_tok is not None and "ve_gate" in blk:
        ve_h = ve_tok.astype(compute_dtype).reshape(B, T, spec.n_kv_head, head_dim)
        gate = 2.0 * jax.nn.sigmoid(jnp.dot(h[..., :ve_ch], blk["ve_gate"].astype(compute_dtype)))
        v = v + gate[..., None] * ve_h

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    if spec.use_qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)

    y = fused_causal_attention(
        q,
        k,
        v,
        window_size=window_size,
        backend=spec.attn_impl,
        splash_block_size=spec.splash_block_size,
        mesh=mesh,
    )
    y_out = jnp.dot(y.reshape(B * T, C), blk["c_proj"].astype(compute_dtype)).reshape(B, T, C)
    x = x + y_out

    # --- MLP sub-block (Fused Gate+Up for SwiGLU) ---
    m = rms_norm(x).reshape(B * T, C)
    if spec.activation == "swiglu":
        gu = jnp.dot(m, blk["w_gate_up"].astype(compute_dtype))
        gate, up = jnp.split(gu, 2, axis=-1)
        m = jax.nn.silu(gate) * up
    elif spec.activation == "relu2":
        m = jnp.square(jax.nn.relu(jnp.dot(m, blk["mlp_fc"].astype(compute_dtype))))
    else:  # "gelu"
        m = jax.nn.gelu(jnp.dot(m, blk["mlp_fc"].astype(compute_dtype)), approximate=True)

    mlp_out = jnp.dot(m, blk["mlp_down"].astype(compute_dtype)).reshape(B, T, C)
    x = x + mlp_out
    return x


def _make_layer_fn(
    spec: ModelSpec,
    window_size: Optional[tuple[int, int]],
    compute_dtype,
    mesh: Optional[Mesh],
):
    fn = functools.partial(
        _apply_single_layer,
        spec=spec,
        window_size=window_size,
        compute_dtype=compute_dtype,
        mesh=mesh,
    )
    policy_name = "none" if (not spec.use_remat and spec.remat_policy == "full") else spec.remat_policy
    policy = _get_checkpoint_policy(policy_name)
    if policy is None:
        return fn
    return jax.checkpoint(fn, policy=policy)


def forward_hidden(
    params: dict,
    idx: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """Run embedding + scanned transformer stack + final RMSNorm; returns `(B, T, C)` in compute_dtype."""
    T = idx.shape[1]
    # Pre-cast RoPE tables ONCE outside the layer scan (saves 2*L redundant casts)
    cos_t = cos[:, :T].astype(compute_dtype)
    sin_t = sin[:, :T].astype(compute_dtype)

    x = params["wte"][idx].astype(compute_dtype)
    x = rms_norm(x)
    x0 = x

    pattern = spec.window_pattern.upper()
    G = len(pattern)
    short_win = (max(1, spec.sequence_len // 2), 0)
    ve_table = params.get("value_embeds", None)

    if pattern == "L":
        layer_fn = _make_layer_fn(spec, None, compute_dtype, mesh)
        if ve_table is not None:
            def scan_body_ve(carry_x, layer_inputs):
                blk, r_lam, x0_lam, ve_w = layer_inputs
                return layer_fn(blk, r_lam, x0_lam, carry_x, x0, ve_w[idx], cos_t, sin_t), None

            x, _ = jax.lax.scan(
                scan_body_ve, x, (params["blocks"], params["resid_lambdas"], params["x0_lambdas"], ve_table)
            )
        else:
            def scan_body_nove(carry_x, layer_inputs):
                blk, r_lam, x0_lam = layer_inputs
                return layer_fn(blk, r_lam, x0_lam, carry_x, x0, None, cos_t, sin_t), None

            x, _ = jax.lax.scan(
                scan_body_nove, x, (params["blocks"], params["resid_lambdas"], params["x0_lambdas"])
            )

    elif spec.n_layer % G == 0:
        n_groups = spec.n_layer // G
        group_fns = [
            _make_layer_fn(spec, short_win if pattern[g] == "S" else None, compute_dtype, mesh)
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

    else:
        for i in range(spec.n_layer):
            char = "L" if i == spec.n_layer - 1 else pattern[i % G]
            win = short_win if char == "S" else None
            layer_fn = _make_layer_fn(spec, win, compute_dtype, mesh)
            blk_i = jax.tree_util.tree_map(lambda a: a[i], params["blocks"])
            ve_tok = ve_table[i][idx] if ve_table is not None else None
            x = layer_fn(blk_i, params["resid_lambdas"][i], params["x0_lambdas"][i], x, x0, ve_tok, cos_t, sin_t)

    return rms_norm(x)


def forward_logits(
    params: dict,
    idx: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """Compute full float32 (optionally softcapped) logits of shape (B, T, vocab_size)."""
    x = forward_hidden(params, idx, cos, sin, spec, compute_dtype, mesh=mesh)
    lm_w = params["wte"].T.astype(compute_dtype) if spec.tie_embeddings else params["lm_head"].astype(compute_dtype)
    logits = jnp.dot(x, lm_w).astype(jnp.float32)
    if spec.softcap > 0.0:
        logits = spec.softcap * jnp.tanh(logits / spec.softcap)
    return logits


def _chunk_ce_and_z(x_chunk: jax.Array, y_chunk: jax.Array, lm_w: jax.Array, softcap: float):
    """Compute exact float32 NLL and mean z-loss for a chunk of `(K, C)` hidden states."""
    logits = jnp.dot(x_chunk, lm_w).astype(jnp.float32)
    if softcap > 0.0:
        logits = softcap * jnp.tanh(logits / softcap)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, y_chunk[..., None], axis=-1)[..., 0]
    nll = log_z - target_logits
    z_loss = jnp.mean(log_z * log_z, dtype=jnp.float32)
    return nll, z_loss


def forward_loss_2d(
    params: dict,
    idx: jax.Array,
    targets: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    return_z_loss: bool = False,
    mesh: Optional[Mesh] = None,
):
    """
    Return exact per-token NLL `(B, T)` in float32 (plus optional mean z-loss).
    Automatically uses checkpointed chunked linear cross-entropy when `loss_chunk_size`
    cleanly divides `B * T` to eliminate the giant `(B, T, vocab)` HBM spike!
    """
    x = forward_hidden(params, idx, cos, sin, spec, compute_dtype, mesh=mesh)
    lm_w = params["wte"].T.astype(compute_dtype) if spec.tie_embeddings else params["lm_head"].astype(compute_dtype)

    B, T, C = x.shape
    total_tok = B * T
    chunk_sz = spec.loss_chunk_size

    if chunk_sz > 0 and total_tok > chunk_sz and total_tok % chunk_sz == 0:
        n_chunks = total_tok // chunk_sz
        x_chunks = x.reshape(n_chunks, chunk_sz, C)
        y_chunks = targets.reshape(n_chunks, chunk_sz)

        chunk_fn = jax.checkpoint(
            functools.partial(_chunk_ce_and_z, softcap=spec.softcap),
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        def scan_chunk(_, pair):
            xc, yc = pair
            nll_c, z_c = chunk_fn(xc, yc, lm_w)
            return None, (nll_c, z_c)

        _, (nll_chunks, z_chunks) = jax.lax.scan(scan_chunk, None, (x_chunks, y_chunks))
        nll = nll_chunks.reshape(B, T)
        if return_z_loss:
            return nll, jnp.mean(z_chunks, dtype=jnp.float32)
        return nll

    nll, z_loss = _chunk_ce_and_z(x.reshape(total_tok, C), targets.reshape(total_tok), lm_w, spec.softcap)
    nll = nll.reshape(B, T)
    if return_z_loss:
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
        active_non_emb += counts["wte"]
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
    """Autoregressive sampler for live qualitative checks during pretraining."""
    # Sampling runs at batch=1 -> use safe XLA attention regardless of Splash/cuDNN training kernel
    sample_spec = ModelSpec.from_dict({**spec.to_dict(), "attn_impl": "xla", "remat_policy": "none"})
    cos, sin = precompute_rotary_embeddings(sample_spec.sequence_len, sample_spec.head_dim, base=sample_spec.rope_base)
    if batch_sharding is not None:
        from jax.sharding import NamedSharding, PartitionSpec as P

        rep = NamedSharding(batch_sharding.mesh, P())
        cos = jax.device_put(cos, rep)
        sin = jax.device_put(sin, rep)

    @jax.jit
    def next_token_logits(p, x_pad, pos):
        logits = forward_logits(p, x_pad, cos, sin, sample_spec, compute_dtype)
        return logits[0, pos]

    cap_len = min(sample_spec.sequence_len, max(len(prompt_ids) + max_new_tokens, 64))
    cap_len = min(sample_spec.sequence_len, ((cap_len + 63) // 64) * 64)
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
