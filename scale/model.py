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
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from scale.attention import fused_causal_attention, resolve_attn_backend
from scale.config import ModelSpec
from scale.quant import (
    _dw_from_saved_xt_int8,
    _quantize_x_for_dw_int8,
    qdot,
    quantize_int8,
)


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


def init_params(key: jax.Array, spec: ModelSpec, param_dtype=jnp.float32) -> dict:
    """
    Initialize parameters in `param_dtype` (`float32` standard or `bfloat16` for pure-BF16 +
    stochastic rounding) with fused `(n_layer, in, out)` tensors for peak MXU/Tensor-Core
    arithmetic intensity and clean FSDP sharding. Small 1D scalars stay float32.
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
        "w_qkv": jax.random.uniform(next_key(), (L, C, qkv_dim), jnp.float32, -s, s).astype(param_dtype),
        "c_proj": jnp.zeros((L, C, C), dtype=param_dtype),
        "mlp_down": jnp.zeros((L, M, C), dtype=param_dtype),
    }
    if spec.activation == "swiglu":
        blocks["w_gate_up"] = jax.random.uniform(next_key(), (L, C, 2 * M), jnp.float32, -s, s).astype(param_dtype)
    else:
        blocks["mlp_fc"] = jax.random.uniform(next_key(), (L, C, M), jnp.float32, -s, s).astype(param_dtype)

    if spec.use_value_embeds:
        blocks["ve_gate"] = jnp.zeros((L, ve_ch, spec.n_kv_head), dtype=param_dtype)
    if spec.use_attn_gate:
        blocks["w_attn_gate"] = jnp.zeros((L, C, spec.n_head), dtype=param_dtype)

    params: dict = {
        "wte": (
            jax.random.normal(next_key(), (spec.vocab_size, C), dtype=jnp.float32)
            * (C**-0.5 if spec.tie_embeddings else 1.0)
        ).astype(param_dtype),
        "blocks": blocks,
        "resid_lambdas": jnp.ones((L,), dtype=jnp.float32),
        "x0_lambdas": jnp.full((L,), 0.1 if spec.use_resid_lambdas else 0.0, dtype=jnp.float32),
    }
    if not spec.tie_embeddings:
        params["lm_head"] = (
            jax.random.normal(next_key(), (C, spec.vocab_size), dtype=jnp.float32) * 0.001
        ).astype(param_dtype)
    if spec.use_value_embeds:
        n_ve = (L + spec.ve_stride - 1) // spec.ve_stride
        params["value_embeds"] = jax.random.uniform(
            next_key(), (n_ve, spec.vocab_size, kv_dim), jnp.float32, -s, s
        ).astype(param_dtype)
    if spec.use_unet_skip:
        params["unet_lambdas"] = jnp.zeros((L // 2,), dtype=jnp.float32)
    if spec.use_qk_gain:
        params["qk_gain"] = jnp.full((L, spec.n_head), 1.3, dtype=jnp.float32)
    if spec.use_v0_residual:
        params["v0_lambdas"] = jnp.zeros((L,), dtype=jnp.float32)
    if spec.use_smear:
        params["smear_gate"] = jnp.zeros((min(12, C), 1), dtype=jnp.float32)
        params["smear_lambda"] = jnp.array(0.0, dtype=jnp.float32)
    return params


def _dots_or_named_policy(*names: str):
    return jax.checkpoint_policies.save_from_both_policies(
        jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        jax.checkpoint_policies.save_only_these_names(*names),
    )


def _get_checkpoint_policy(remat_policy: str):
    if remat_policy == "none":
        return None
    if remat_policy == "save_all_compute":
        return _dots_or_named_policy("qkv", "out_proj", "mlp_wi")
    if remat_policy == "save_attn":
        return jax.checkpoint_policies.save_only_these_names("qkv", "context")
    if remat_policy == "save_qkv_attn":
        return jax.checkpoint_policies.save_only_these_names("qkv", "context")
    if remat_policy == "dots_with_no_batch_dims":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    if remat_policy == "dots":
        return jax.checkpoint_policies.dots_saveable
    return jax.checkpoint_policies.nothing_saveable


def _pre_attn_subblock(
    blk: dict,
    q_blk: Optional[dict],
    resid_lam: jax.Array,
    x0_lam: jax.Array,
    x: jax.Array,
    x0: jax.Array,
    ve_tok: Optional[jax.Array],
    v0: Optional[jax.Array],
    v0_lam: Optional[jax.Array],
    qk_gain: Optional[jax.Array],
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    mesh: Optional[Mesh] = None,
):
    B, T, C = x.shape
    head_dim = spec.head_dim
    kv_dim = spec.kv_dim
    ve_ch = min(32, C)
    qmode = spec.matmul_quant

    if spec.use_resid_lambdas:
        x = resid_lam.astype(compute_dtype) * x + x0_lam.astype(compute_dtype) * x0

    h = rms_norm(x)
    qw_qkv = q_blk["w_qkv"] if (q_blk is not None and "w_qkv" in q_blk) else None
    qkv = qdot(h.reshape(B * T, C), blk["w_qkv"].astype(compute_dtype), mode=qmode, q_weight=qw_qkv, mesh=mesh)
    qkv = checkpoint_name(qkv, "qkv")
    q = qkv[:, :C].reshape(B, T, spec.n_head, head_dim)
    k = qkv[:, C : C + kv_dim].reshape(B, T, spec.n_kv_head, head_dim)
    v = qkv[:, C + kv_dim :].reshape(B, T, spec.n_kv_head, head_dim)

    if ve_tok is not None and "ve_gate" in blk:
        ve_h = ve_tok.astype(compute_dtype).reshape(B, T, spec.n_kv_head, head_dim)
        gate = 2.0 * jax.nn.sigmoid(jnp.dot(h[..., :ve_ch], blk["ve_gate"].astype(compute_dtype)))
        v = v + gate[..., None] * ve_h

    if v0 is not None and v0_lam is not None:
        v = v + v0_lam.astype(compute_dtype) * v0

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    if spec.use_qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)
    if qk_gain is not None:
        q = q * qk_gain.astype(compute_dtype)[None, None, :, None]

    attn_gate = None
    if spec.use_attn_gate and "w_attn_gate" in blk:
        attn_gate = jax.nn.sigmoid(
            jnp.dot(h, blk["w_attn_gate"].astype(compute_dtype)) + jnp.asarray(1.0, compute_dtype)
        )
    return q, k, v, attn_gate


def _post_attn_mlp_subblock(
    blk: dict,
    q_blk: Optional[dict],
    resid_lam: jax.Array,
    x0_lam: jax.Array,
    x: jax.Array,
    x0: jax.Array,
    y: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    B, T, C = x.shape
    qmode = spec.matmul_quant
    if spec.use_resid_lambdas:
        x = resid_lam.astype(compute_dtype) * x + x0_lam.astype(compute_dtype) * x0

    qw_out = q_blk["c_proj"] if (q_blk is not None and "c_proj" in q_blk) else None
    y_out = qdot(y.reshape(B * T, C), blk["c_proj"].astype(compute_dtype), mode=qmode, q_weight=qw_out, mesh=mesh).reshape(B, T, C)
    y_out = checkpoint_name(y_out, "out_proj")
    x = x + y_out

    m = rms_norm(x).reshape(B * T, C)
    if spec.activation == "swiglu":
        qw_gu = q_blk["w_gate_up"] if (q_blk is not None and "w_gate_up" in q_blk) else None
        gu = checkpoint_name(qdot(m, blk["w_gate_up"].astype(compute_dtype), mode=qmode, q_weight=qw_gu, mesh=mesh), "mlp_wi")
        gate, up = jnp.split(gu, 2, axis=-1)
        m = jax.nn.silu(gate) * up
    elif spec.activation == "relu2":
        qw_fc = q_blk["mlp_fc"] if (q_blk is not None and "mlp_fc" in q_blk) else None
        fc = checkpoint_name(qdot(m, blk["mlp_fc"].astype(compute_dtype), mode=qmode, q_weight=qw_fc, mesh=mesh), "mlp_wi")
        m = jnp.square(jax.nn.relu(fc))
    else:  # "gelu"
        qw_fc = q_blk["mlp_fc"] if (q_blk is not None and "mlp_fc" in q_blk) else None
        fc = checkpoint_name(qdot(m, blk["mlp_fc"].astype(compute_dtype), mode=qmode, q_weight=qw_fc, mesh=mesh), "mlp_wi")
        m = jax.nn.gelu(fc, approximate=True)

    qw_dn = q_blk["mlp_down"] if (q_blk is not None and "mlp_down" in q_blk) else None
    mlp_out = qdot(m, blk["mlp_down"].astype(compute_dtype), mode=qmode, q_weight=qw_dn, mesh=mesh).reshape(B, T, C)
    return x + mlp_out


def _apply_single_layer(
    blk: dict,
    q_blk: Optional[dict],
    resid_lam: jax.Array,
    x0_lam: jax.Array,
    x: jax.Array,
    x0: jax.Array,
    ve_tok: Optional[jax.Array],
    v0: Optional[jax.Array],
    v0_lam: Optional[jax.Array],
    qk_gain: Optional[jax.Array],
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    window_size: Optional[tuple[int, int]],
    compute_dtype,
    mesh: Optional[Mesh],
) -> jax.Array:
    q, k, v, attn_gate = _pre_attn_subblock(
        blk, q_blk, resid_lam, x0_lam, x, x0, ve_tok, v0, v0_lam, qk_gain, cos, sin, spec, compute_dtype, mesh=mesh
    )
    y = fused_causal_attention(
        q,
        k,
        v,
        window_size=window_size,
        backend=spec.attn_impl,
        splash_block_size=spec.splash_block_size,
        mesh=mesh,
    )
    if attn_gate is not None:
        y = y * attn_gate[..., None]
    y = checkpoint_name(y, "context")
    return _post_attn_mlp_subblock(blk, q_blk, resid_lam, x0_lam, x, x0, y, spec, compute_dtype, mesh=mesh)


def _make_layer_fn(
    spec: ModelSpec,
    window_size: Optional[tuple[int, int]],
    compute_dtype,
    mesh: Optional[Mesh],
):
    policy_name = "none" if (not spec.use_remat and spec.remat_policy == "full") else spec.remat_policy
    attn_backend = resolve_attn_backend(spec.attn_impl)

    # For O(T)-memory fused kernels (TPU SplashAttention & GPU cuDNN FlashAttention),
    # split-remat keeps attention outside jax.checkpoint so attention forward NEVER runs twice,
    # while checkpointing pre-attn & post-attn/MLP discards giant silu & fp32 rms_norm buffers!
    if policy_name in ("save_all_compute", "save_attn") and attn_backend in ("splash", "cudnn"):
        pre_fn = jax.checkpoint(
            functools.partial(_pre_attn_subblock, spec=spec, compute_dtype=compute_dtype, mesh=mesh),
            policy=_dots_or_named_policy("qkv"),
        )
        post_pol = (
            _dots_or_named_policy("out_proj", "mlp_wi")
            if policy_name == "save_all_compute"
            else jax.checkpoint_policies.nothing_saveable
        )
        post_fn = jax.checkpoint(
            functools.partial(_post_attn_mlp_subblock, spec=spec, compute_dtype=compute_dtype, mesh=mesh),
            policy=post_pol,
        )

        def split_layer(blk, q_blk, resid_lam, x0_lam, x, x0, ve_tok, v0, v0_lam, qk_gain, cos, sin):
            q, k, v, attn_gate = pre_fn(blk, q_blk, resid_lam, x0_lam, x, x0, ve_tok, v0, v0_lam, qk_gain, cos, sin)
            y = fused_causal_attention(
                q,
                k,
                v,
                window_size=window_size,
                backend=spec.attn_impl,
                splash_block_size=spec.splash_block_size,
                mesh=mesh,
            )
            if attn_gate is not None:
                y = y * attn_gate[..., None]
            return post_fn(blk, q_blk, resid_lam, x0_lam, x, x0, y)

        return split_layer

    fn = functools.partial(
        _apply_single_layer,
        spec=spec,
        window_size=window_size,
        compute_dtype=compute_dtype,
        mesh=mesh,
    )
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
    q_blocks: Optional[dict] = None,
) -> jax.Array:
    """Run embedding + scanned transformer stack + final RMSNorm; returns `(B, T, C)` in compute_dtype."""
    import math

    T = idx.shape[1]
    # Pre-cast RoPE tables ONCE outside the layer scan (saves 2*L redundant casts)
    cos_t = cos[:, :T].astype(compute_dtype)
    sin_t = sin[:, :T].astype(compute_dtype)

    x = params["wte"][idx].astype(compute_dtype)
    x = rms_norm(x)
    if spec.use_smear and "smear_gate" in params:
        sm_ch = min(12, spec.n_embd)
        sm_gate = jax.nn.sigmoid(jnp.dot(x[..., :sm_ch], params["smear_gate"].astype(compute_dtype)))
        x_prev = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        x = rms_norm(x + (params["smear_lambda"].astype(compute_dtype) * sm_gate) * x_prev)
    x0 = x

    v0 = None
    v0_lams = params.get("v0_lambdas", None) if spec.use_v0_residual else None
    if v0_lams is not None:
        B_cur, T_cur, C_cur = x0.shape
        w_v0 = params["blocks"]["w_qkv"][0, :, C_cur + spec.kv_dim :].astype(compute_dtype)
        v0 = jnp.dot(x0.reshape(B_cur * T_cur, C_cur), w_v0).reshape(B_cur, T_cur, spec.n_kv_head, spec.head_dim)

    qk_gains = params.get("qk_gain", None) if spec.use_qk_gain else None

    pattern = spec.window_pattern.upper()
    ve_table = params.get("value_embeds", None)
    ve_stride = spec.ve_stride if ve_table is not None else 1
    if ve_stride > 1 and len(pattern) % ve_stride != 0:
        rep = math.lcm(len(pattern), ve_stride) // len(pattern)
        pattern = pattern * rep
    G = len(pattern)
    short_win = (max(1, spec.sequence_len // 2), 0)
    q_scan = q_blocks if q_blocks is not None else {}
    unet_lams = params.get("unet_lambdas", None) if spec.use_unet_skip else None

    if spec.n_layer % G == 0:
        n_groups = spec.n_layer // G
        group_fns = [
            _make_layer_fn(spec, short_win if pattern[g] == "S" else None, compute_dtype, mesh)
            for g in range(G)
        ]

        def reshape_group(arr):
            return arr.reshape((n_groups, G) + arr.shape[1:])

        grouped_blocks = jax.tree_util.tree_map(reshape_group, params["blocks"])
        grouped_q = jax.tree_util.tree_map(reshape_group, q_scan)
        grouped_r = reshape_group(params["resid_lambdas"])
        grouped_x0 = reshape_group(params["x0_lambdas"])
        grouped_v0l = reshape_group(v0_lams) if v0_lams is not None else None
        grouped_qkg = reshape_group(qk_gains) if qk_gains is not None else None
        if ve_table is not None:
            ve_per_grp = G // ve_stride
            grouped_ve = ve_table.reshape((n_groups, ve_per_grp) + ve_table.shape[1:])
        else:
            grouped_ve = None

        def _exec_group(carry_x, g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg, g_skip_lams=None, g_skips=None):
            cur = carry_x
            outs = []
            for g in range(G):
                if g_skip_lams is not None and g_skips is not None:
                    cur = cur + g_skip_lams[g].astype(compute_dtype) * g_skips[g]
                blk_g = jax.tree_util.tree_map(lambda a: a[g], g_blk)
                qb_g = jax.tree_util.tree_map(lambda a: a[g], g_qb) if g_qb else None
                ve_tok = g_ve[g // ve_stride][idx] if (g_ve is not None and g % ve_stride == 0) else None
                v0_l = g_v0l[g] if g_v0l is not None else None
                qk_g = g_qkg[g] if g_qkg is not None else None
                cur = group_fns[g](blk_g, qb_g, g_r[g], g_x0[g], cur, x0, ve_tok, v0, v0_l, qk_g, cos_t, sin_t)
                outs.append(cur)
            return cur, outs

        def _slice_grp(tree, start: int, end: int):
            return jax.tree_util.tree_map(lambda a: a[start:end], tree)

        if unet_lams is not None and n_groups >= 2:
            enc_g = n_groups // 2
            dec_start = n_groups - enc_g
            n_skip = enc_g * G

            def enc_body(carry_x, grp):
                g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg = grp
                cur, outs = _exec_group(carry_x, g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg)
                return cur, jnp.stack(outs, axis=0)

            enc_in = (
                _slice_grp(grouped_blocks, 0, enc_g),
                _slice_grp(grouped_q, 0, enc_g),
                grouped_r[:enc_g],
                grouped_x0[:enc_g],
                grouped_ve[:enc_g] if grouped_ve is not None else None,
                grouped_v0l[:enc_g] if grouped_v0l is not None else None,
                grouped_qkg[:enc_g] if grouped_qkg is not None else None,
            )
            x, enc_skips = jax.lax.scan(enc_body, x, enc_in, unroll=spec.scan_unroll)
            # Reverse encoder layer outputs so decoder layer (L-1-i) pairs with encoder layer i
            flat_skips = enc_skips.reshape((n_skip,) + x.shape)
            dec_skips = jnp.flip(flat_skips, axis=0).reshape((enc_g, G) + x.shape)
            dec_lams = unet_lams[:n_skip].reshape((enc_g, G))

            if dec_start > enc_g:
                def mid_body(carry_x, grp):
                    g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg = grp
                    cur, _ = _exec_group(carry_x, g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg)
                    return cur, None

                mid_in = (
                    _slice_grp(grouped_blocks, enc_g, dec_start),
                    _slice_grp(grouped_q, enc_g, dec_start),
                    grouped_r[enc_g:dec_start],
                    grouped_x0[enc_g:dec_start],
                    grouped_ve[enc_g:dec_start] if grouped_ve is not None else None,
                    grouped_v0l[enc_g:dec_start] if grouped_v0l is not None else None,
                    grouped_qkg[enc_g:dec_start] if grouped_qkg is not None else None,
                )
                x, _ = jax.lax.scan(mid_body, x, mid_in, unroll=spec.scan_unroll)

            def dec_body(carry_x, grp):
                g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg, g_ul, g_sk = grp
                cur, _ = _exec_group(carry_x, g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg, g_ul, g_sk)
                return cur, None

            dec_in = (
                _slice_grp(grouped_blocks, dec_start, n_groups),
                _slice_grp(grouped_q, dec_start, n_groups),
                grouped_r[dec_start:n_groups],
                grouped_x0[dec_start:n_groups],
                grouped_ve[dec_start:n_groups] if grouped_ve is not None else None,
                grouped_v0l[dec_start:n_groups] if grouped_v0l is not None else None,
                grouped_qkg[dec_start:n_groups] if grouped_qkg is not None else None,
                dec_lams,
                dec_skips,
            )
            x, _ = jax.lax.scan(dec_body, x, dec_in, unroll=spec.scan_unroll)
        else:
            def std_body(carry_x, grp):
                g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg = grp
                cur, _ = _exec_group(carry_x, g_blk, g_qb, g_r, g_x0, g_ve, g_v0l, g_qkg)
                return cur, None

            x, _ = jax.lax.scan(
                std_body,
                x,
                (grouped_blocks, grouped_q, grouped_r, grouped_x0, grouped_ve, grouped_v0l, grouped_qkg),
                unroll=spec.scan_unroll,
            )

    else:
        skips_list = []
        half = spec.n_layer // 2
        for i in range(spec.n_layer):
            if unet_lams is not None and i >= spec.n_layer - half:
                dec_idx = i - (spec.n_layer - half)
                x = x + unet_lams[dec_idx].astype(compute_dtype) * skips_list[half - 1 - dec_idx]
            char = "L" if i == spec.n_layer - 1 else pattern[i % G]
            win = short_win if char == "S" else None
            layer_fn = _make_layer_fn(spec, win, compute_dtype, mesh)
            blk_i = jax.tree_util.tree_map(lambda a: a[i], params["blocks"])
            qb_i = jax.tree_util.tree_map(lambda a: a[i], q_blocks) if q_blocks is not None else None
            ve_tok = ve_table[i // ve_stride][idx] if (ve_table is not None and i % ve_stride == 0) else None
            v0_l = v0_lams[i] if v0_lams is not None else None
            qk_g = qk_gains[i] if qk_gains is not None else None
            x = layer_fn(blk_i, qb_i, params["resid_lambdas"][i], params["x0_lambdas"][i], x, x0, ve_tok, v0, v0_l, qk_g, cos_t, sin_t)
            if unet_lams is not None and i < half:
                skips_list.append(x)

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
    logits = jnp.dot(x, lm_w, preferred_element_type=jnp.float32)
    if spec.softcap > 0.0:
        logits = spec.softcap * jnp.tanh(logits / spec.softcap)
    return logits


def _chunk_ce_and_z(x_chunk: jax.Array, y_chunk: jax.Array, lm_w: jax.Array, softcap: float):
    """Compute exact float32 NLL and mean z-loss for a chunk of `(K, C)` hidden states."""
    logits = jnp.dot(x_chunk, lm_w, preferred_element_type=jnp.float32)
    if softcap > 0.0:
        logits = softcap * jnp.tanh(logits / softcap)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, y_chunk[..., None], axis=-1)[..., 0]
    nll = log_z - target_logits
    z_loss = jnp.mean(log_z * log_z, dtype=jnp.float32)
    return nll, z_loss


def _make_fused_chunked_loss_fn(softcap: float, z_loss_weight: float):
    """
    Single-pass Cut-Cross-Entropy in pure JAX (`@jax.custom_vjp`).

    Operates on FSDP-aligned chunks `(n_chunks, global_chunk_tok, C)` and `(n_chunks, global_chunk_tok)`
    so on multi-chip TPU/GPU (e.g. Kaggle v5e-8) each chip processes its own local chunk in parallel
    with zero cross-chip resharding. Computes `(scaled_total, mean_ce, mean_z)` AND exact gradients
    `(d_x_chunks, d_lm_w)` in ONE chunked scan — 0% `lm_head` recomputation (3x matmuls instead of 4x)
    while keeping peak logits HBM bounded to a single chunk (~268 MB vs 4.3 GB!).
    """
    def _chunk_total(xc, w, yc, chunk_scale):
        nll_c, z_c = _chunk_ce_and_z(xc, yc, w, softcap)
        ce_c = jnp.mean(nll_c, dtype=jnp.float32)
        tot_c = (ce_c + jnp.float32(z_loss_weight) * z_c) * chunk_scale
        return tot_c, (ce_c, z_c)

    chunk_val_and_grad = jax.value_and_grad(_chunk_total, argnums=(0, 1), has_aux=True)

    @jax.custom_vjp
    def _fused_loss(x_ch: jax.Array, lm_w: jax.Array, y_ch: jax.Array, scale: jax.Array):
        n_chunks = x_ch.shape[0]
        inv_n = jnp.float32(1.0 / n_chunks)

        def fwd_only_body(carry, pair):
            ce_acc, z_acc = carry
            xc, yc = pair
            nll_c, z_c = _chunk_ce_and_z(xc, yc, lm_w, softcap)
            return (ce_acc + jnp.mean(nll_c, dtype=jnp.float32), z_acc + z_c), None

        (ce_sum, z_sum), _ = jax.lax.scan(
            fwd_only_body, (jnp.float32(0.0), jnp.float32(0.0)), (x_ch, y_ch)
        )
        ce = ce_sum * inv_n
        z = z_sum * inv_n
        tot = (ce + jnp.float32(z_loss_weight) * z) * scale
        return tot, ce, z

    def _fused_fwd(x_ch: jax.Array, lm_w: jax.Array, y_ch: jax.Array, scale: jax.Array):
        n_chunks = x_ch.shape[0]
        inv_n = jnp.float32(1.0 / n_chunks)
        chunk_scale = scale * inv_n
        # Accumulate lm_head grad in fp32 for fp16/fp32, or lm_w.dtype (bf16) when weights are bf16
        dw_acc_dtype = jnp.float32 if lm_w.dtype == jnp.float16 else lm_w.dtype
        dw_init = jnp.zeros_like(lm_w, dtype=dw_acc_dtype)

        def scan_body(carry, pair):
            tot_acc, ce_acc, z_acc, dw_acc = carry
            xc, yc = pair
            (tot_c, (ce_c, z_c)), (dx_c, dw_c) = chunk_val_and_grad(xc, lm_w, yc, chunk_scale)
            dw_acc = dw_acc + dw_c.astype(dw_acc_dtype)
            return (tot_acc + tot_c, ce_acc + ce_c, z_acc + z_c, dw_acc), dx_c

        (tot_sum, ce_sum, z_sum, dw_sum), dx_chunks = jax.lax.scan(
            scan_body,
            (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), dw_init),
            (x_ch, y_ch),
        )
        ce = ce_sum * inv_n
        z = z_sum * inv_n
        return (tot_sum, ce, z), (dx_chunks, dw_sum.astype(lm_w.dtype))

    def _fused_bwd(res, g_out):
        dx_chunks, d_lm_w = res
        g_tot, _, _ = g_out
        return (
            dx_chunks * g_tot.astype(dx_chunks.dtype),
            d_lm_w * g_tot.astype(d_lm_w.dtype),
            None,
            None,
        )

    _fused_loss.defvjp(_fused_fwd, _fused_bwd)
    return _fused_loss


def _make_fused_chunked_loss_int8_bwd_fn(softcap: float, z_loss_weight: float, mesh: Optional[Mesh]):
    """
    Single-pass Cut-Cross-Entropy with 100% exact BF16->FP32 forward logits + softmax,
    and 394-TOPS INT8 MXU backward (`d_x` & `d_lm_w`) using pre-quantized `lm_w_qb`.
    """
    @jax.custom_vjp
    def _fused_loss(x_ch, lm_w, lm_w_qb, lm_w_sb, y_ch, scale):
        n_chunks = x_ch.shape[0]
        inv_n = jnp.float32(1.0 / n_chunks)

        def fwd_only_body(carry, pair):
            ce_acc, z_acc = carry
            xc, yc = pair
            nll_c, z_c = _chunk_ce_and_z(xc, yc, lm_w, softcap)
            return (ce_acc + jnp.mean(nll_c, dtype=jnp.float32), z_acc + z_c), None

        (ce_sum, z_sum), _ = jax.lax.scan(
            fwd_only_body, (jnp.float32(0.0), jnp.float32(0.0)), (x_ch, y_ch)
        )
        ce = ce_sum * inv_n
        z = z_sum * inv_n
        tot = (ce + jnp.float32(z_loss_weight) * z) * scale
        return tot, ce, z

    def _fused_fwd(x_ch, lm_w, lm_w_qb, lm_w_sb, y_ch, scale):
        n_chunks = x_ch.shape[0]
        chunk_tok = x_ch.shape[1]
        vocab_size = lm_w.shape[1]
        inv_n = jnp.float32(1.0 / n_chunks)
        chunk_scale = scale * inv_n
        tok_scale = chunk_scale / jnp.float32(chunk_tok)
        dw_acc_dtype = jnp.float32 if lm_w.dtype == jnp.float16 else lm_w.dtype
        dw_init = jnp.zeros_like(lm_w, dtype=dw_acc_dtype)

        def scan_body(carry, pair):
            tot_acc, ce_acc, z_acc, dw_acc = carry
            xc, yc = pair
            raw = jnp.dot(xc, lm_w, preferred_element_type=jnp.float32)
            if softcap > 0.0:
                th = jnp.tanh(raw / softcap)
                logits = softcap * th
            else:
                th = None
                logits = raw

            log_z = jax.scipy.special.logsumexp(logits, axis=-1)
            target_logits = jnp.take_along_axis(logits, yc[..., None], axis=-1)[..., 0]
            ce_c = jnp.mean(log_z - target_logits, dtype=jnp.float32)
            z_c = jnp.mean(log_z * log_z, dtype=jnp.float32)
            tot_c = (ce_c + jnp.float32(z_loss_weight) * z_c) * chunk_scale

            # Analytical O(1) unnormalized dL/d(raw_logits) — fuses one_hot in XLA VPU!
            probs = jnp.exp(logits - log_z[:, None])
            d_logits = probs * (jnp.float32(1.0) + jnp.float32(2.0 * z_loss_weight) * log_z[:, None])
            d_logits = d_logits - jax.nn.one_hot(yc, vocab_size, dtype=jnp.float32)
            d_raw = d_logits * (jnp.float32(1.0) - th * th) if th is not None else d_logits

            # 394-TOPS INT8 backward d_x = d_raw @ lm_w.T (1-byte pre-quantized lm_w_qb!)
            dr_q, dr_s = quantize_int8(d_raw, axis=-1)
            dx_i32 = jnp.dot(dr_q, lm_w_qb.T, preferred_element_type=jnp.int32)
            dx_c = (dx_i32.astype(jnp.float32) * (dr_s * lm_w_sb.T) * tok_scale).astype(xc.dtype)

            # 394-TOPS INT8 backward d_w = x.T @ d_raw (FSDP-local bf16 psum_scatter on v5e-8!)
            xt_q, xt_s = _quantize_x_for_dw_int8(xc, mesh)
            dw_unnorm = _dw_from_saved_xt_int8(xt_q, xt_s, d_raw, dw_acc_dtype, mesh)

            dw_acc = dw_acc + (dw_unnorm.astype(jnp.float32) * tok_scale).astype(dw_acc_dtype)
            return (tot_acc + tot_c, ce_acc + ce_c, z_acc + z_c, dw_acc), dx_c

        (tot_sum, ce_sum, z_sum, dw_sum), dx_chunks = jax.lax.scan(
            scan_body,
            (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), dw_init),
            (x_ch, y_ch),
        )
        ce = ce_sum * inv_n
        z = z_sum * inv_n
        return (tot_sum, ce, z), (dx_chunks, dw_sum.astype(lm_w.dtype))

    def _fused_bwd(res, g_out):
        dx_chunks, d_lm_w = res
        g_tot, _, _ = g_out
        return (
            dx_chunks * g_tot.astype(dx_chunks.dtype),
            d_lm_w * g_tot.astype(d_lm_w.dtype),
            None,
            None,
            None,
            None,
        )

    _fused_loss.defvjp(_fused_fwd, _fused_bwd)
    return _fused_loss


def _pack_fsdp_chunks(x: jax.Array, targets: jax.Array, chunk_sz: int, mesh: Optional[Mesh]):
    """
    Pack `(B, T, C)` and `(B, T)` into `(n_chunks, n_dev * chunk_sz, C)` so every FSDP chip
    processes `chunk_sz` local tokens per scan step with zero cross-chip resharding.
    Returns `(x_chunks, y_chunks, n_dev, n_chunks)` or `None` if chunking does not apply.
    """
    B, T, C = x.shape
    n_dev = mesh.size if (mesh is not None and mesh.size > 1 and B % mesh.size == 0) else 1
    local_tok = (B // n_dev) * T
    if chunk_sz <= 0 or local_tok < chunk_sz or local_tok % chunk_sz != 0:
        return None
    n_chunks = local_tok // chunk_sz
    x_4d = x.reshape(n_dev, n_chunks, chunk_sz, C).transpose(1, 0, 2, 3)
    y_3d = targets.reshape(n_dev, n_chunks, chunk_sz).transpose(1, 0, 2)
    return x_4d.reshape(n_chunks, n_dev * chunk_sz, C), y_3d.reshape(n_chunks, n_dev * chunk_sz), n_dev, n_chunks


def forward_scalar_loss_and_z(
    params: dict,
    idx: jax.Array,
    targets: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    spec: ModelSpec,
    compute_dtype,
    scale: jax.Array,
    mesh: Optional[Mesh] = None,
    q_cache: Optional[dict] = None,
):
    """
    Training loss entry point returning `(scaled_total_loss, mean_ce, mean_z)`.
    Uses single-pass fused Cut-Cross-Entropy when chunking applies (0% lm_head recompute,
    ~16x lower logits HBM, and zero cross-chip resharding on multi-chip FSDP!).
    """
    q_blocks = q_cache["blocks"] if (q_cache is not None and "blocks" in q_cache) else None
    x = forward_hidden(params, idx, cos, sin, spec, compute_dtype, mesh=mesh, q_blocks=q_blocks)
    lm_w = params["wte"].T.astype(compute_dtype) if spec.tie_embeddings else params["lm_head"].astype(compute_dtype)

    packed = _pack_fsdp_chunks(x, targets, spec.loss_chunk_size, mesh)
    if packed is not None:
        x_chunks, y_chunks, _, _ = packed
        if spec.matmul_quant == "int8-all" and q_cache is not None and "lm_head_bwd" in q_cache:
            lm_w_qb, lm_w_sb = q_cache["lm_head_bwd"]
            fused_q_fn = _make_fused_chunked_loss_int8_bwd_fn(spec.softcap, spec.z_loss_weight, mesh)
            return fused_q_fn(x_chunks, lm_w, lm_w_qb, lm_w_sb, y_chunks, scale)
        fused_fn = _make_fused_chunked_loss_fn(spec.softcap, spec.z_loss_weight)
        return fused_fn(x_chunks, lm_w, y_chunks, scale)

    B, T, C = x.shape
    nll, z_loss = _chunk_ce_and_z(x.reshape(B * T, C), targets.reshape(B * T), lm_w, spec.softcap)
    ce = jnp.mean(nll, dtype=jnp.float32)
    tot = (ce + jnp.float32(spec.z_loss_weight) * z_loss) * scale
    return tot, ce, z_loss


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
    Automatically uses FSDP-aligned chunked linear cross-entropy when `loss_chunk_size`
    cleanly divides per-device tokens to eliminate the giant `(B, T, vocab)` HBM spike!
    """
    x = forward_hidden(params, idx, cos, sin, spec, compute_dtype, mesh=mesh)
    lm_w = params["wte"].T.astype(compute_dtype) if spec.tie_embeddings else params["lm_head"].astype(compute_dtype)

    B, T, C = x.shape
    packed = _pack_fsdp_chunks(x, targets, spec.loss_chunk_size, mesh)
    if packed is not None:
        x_chunks, y_chunks, n_dev, n_chunks = packed
        chunk_sz = spec.loss_chunk_size

        chunk_fn = jax.checkpoint(
            functools.partial(_chunk_ce_and_z, softcap=spec.softcap),
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        def scan_chunk(_, pair):
            xc, yc = pair
            nll_c, z_c = chunk_fn(xc, yc, lm_w)
            return None, (nll_c, z_c)

        _, (nll_chunks, z_chunks) = jax.lax.scan(scan_chunk, None, (x_chunks, y_chunks))
        nll = nll_chunks.reshape(n_chunks, n_dev, chunk_sz).transpose(1, 0, 2).reshape(B, T)
        if return_z_loss:
            return nll, jnp.mean(z_chunks, dtype=jnp.float32)
        return nll

    nll, z_loss = _chunk_ce_and_z(x.reshape(B * T, C), targets.reshape(B * T), lm_w, spec.softcap)
    nll = nll.reshape(B, T)
    if return_z_loss:
        return nll, z_loss
    return nll


def count_params(params: dict) -> dict[str, int]:
    wte = int(params["wte"].size)
    lm_head = int(params["lm_head"].size) if "lm_head" in params else 0
    ve = int(params["value_embeds"].size) if "value_embeds" in params else 0
    blocks = sum(int(a.size) for a in jax.tree_util.tree_leaves(params["blocks"]))
    scalar_keys = ("resid_lambdas", "x0_lambdas", "unet_lambdas", "qk_gain", "v0_lambdas", "smear_gate", "smear_lambda")
    scalars = sum(int(params[k].size) for k in scalar_keys if k in params)
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
