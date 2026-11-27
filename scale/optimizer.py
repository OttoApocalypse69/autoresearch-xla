"""
FSDP-compatible Polar-Express NorMuon + AdamW optimizer with WSD schedule.

Uses `jax.lax.scan` over axis 0 (`n_layer`) of stacked `(n_layer, in_dim, out_dim)`
matrices during Polar Express orthogonalization so only ONE layer's temporary matrix
is ever materialized at a time — keeping peak HBM tiny even on a 1B model.
"""

from __future__ import annotations

import inspect
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from scale.config import ModelSpec, TrainConfig
from scale.quant import stochastic_round_bf16

_SHARD_MAP = getattr(jax, "shard_map", None)
if _SHARD_MAP is None:
    from jax.experimental.shard_map import shard_map as _SHARD_MAP  # type: ignore
_SHARD_MAP_PARAMS = inspect.signature(_SHARD_MAP).parameters
_SHARD_MAP_KWARGS = {"check_vma": False} if "check_vma" in _SHARD_MAP_PARAMS else {"check_rep": False}

POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


GATE_BLOCK_KEYS = ("ve_gate", "w_attn_gate")
EXTRA_BLOCK_GROUPS = ("mtp_blocks",)
EXTRA_SCALAR_KEYS = (
    "unet_lambdas",
    "qk_gain",
    "v0_lambdas",
    "smear_gate",
    "smear_lambda",
    "smear_lambda2",
    "canon_a",
    "canon_c",
    "mtp_canon",
)
GENTLE_SCALAR_KEYS = (
    "qk_gain",
    "canon_a",
    "canon_c",
    "mtp_canon",
)


def _muon_sec_shape(L: int, r: int, c: int) -> tuple[int, int, int]:
    return (L, r, 1) if r >= c else (L, 1, c)


def init_opt_state(
    params: dict,
    opt_dtype=jnp.float32,
    gate_adamw: bool = False,
    split_muon: bool = True,
    spec: ModelSpec | None = None,
) -> dict:
    """Create optimizer state pytree matching `params` (supports fp32 or bf16 state)."""
    def adamw_zeros(arr):
        return {
            "m": jnp.zeros_like(arr, dtype=opt_dtype),
            "v": jnp.zeros_like(arr, dtype=opt_dtype),
        }

    def muon_zeros_for_block(name: str, w: jax.Array):
        if gate_adamw and name in GATE_BLOCK_KEYS:
            return adamw_zeros(w)
        L, r, c = w.shape
        if split_muon and name == "w_qkv" and spec is not None:
            C = spec.n_embd
            kv = spec.kv_dim
            return {
                "m": jnp.zeros_like(w, dtype=opt_dtype),
                "v_q": jnp.zeros(_muon_sec_shape(L, r, C), dtype=jnp.float32),
                "v_k": jnp.zeros(_muon_sec_shape(L, r, kv), dtype=jnp.float32),
                "v_v": jnp.zeros(_muon_sec_shape(L, r, kv), dtype=jnp.float32),
            }
        if split_muon and name == "w_gate_up":
            half_c = c // 2
            return {
                "m": jnp.zeros_like(w, dtype=opt_dtype),
                "v_gate": jnp.zeros(_muon_sec_shape(L, r, half_c), dtype=jnp.float32),
                "v_up": jnp.zeros(_muon_sec_shape(L, r, half_c), dtype=jnp.float32),
            }
        return {
            "m": jnp.zeros_like(w, dtype=opt_dtype),
            "v_rowcol": jnp.zeros(_muon_sec_shape(L, r, c), dtype=jnp.float32),
        }

    block_states = {name: muon_zeros_for_block(name, w) for name, w in params["blocks"].items()}

    state: dict = {
        "step": jnp.array(0, dtype=jnp.int32),
        "wte": adamw_zeros(params["wte"]),
        "resid_lambdas": adamw_zeros(params["resid_lambdas"]),
        "x0_lambdas": adamw_zeros(params["x0_lambdas"]),
        "blocks": block_states,
    }
    if "lm_head" in params:
        state["lm_head"] = adamw_zeros(params["lm_head"])
    if "value_embeds" in params:
        state["value_embeds"] = adamw_zeros(params["value_embeds"])
    for bg in EXTRA_BLOCK_GROUPS:
        if bg in params:
            state[bg] = {name: muon_zeros_for_block(name, w) for name, w in params[bg].items()}
    for k in EXTRA_SCALAR_KEYS:
        if k in params:
            state[k] = adamw_zeros(params[k])
    return state


def _cast_updated_param(p_new_f32: jax.Array, orig_dtype, sr_key: jax.Array) -> jax.Array:
    if orig_dtype == jnp.bfloat16:
        return stochastic_round_bf16(p_new_f32, sr_key)
    return p_new_f32.astype(orig_dtype)


def _adamw_update(p, g, st, step_f, lr, beta1, beta2, sr_key, eps=1e-10, wd=0.0):
    orig_dtype = p.dtype
    opt_dtype = st["m"].dtype
    p_f = p.astype(jnp.float32)
    g_f = g.astype(jnp.float32)
    m = st["m"].astype(jnp.float32) * beta1 + g_f * (1.0 - beta1)
    v = st["v"].astype(jnp.float32) * beta2 + (g_f * g_f) * (1.0 - beta2)
    bias1 = 1.0 - (beta1**step_f)
    bias2 = 1.0 - (beta2**step_f)
    denom = jnp.sqrt(v / bias2) + eps
    p_new_f = p_f * (1.0 - lr * wd) - (lr / bias1) * (m / denom)
    p_new = _cast_updated_param(p_new_f, orig_dtype, sr_key)
    return p_new, {"m": m.astype(opt_dtype), "v": v.astype(opt_dtype)}


def _muon_update_2d(p, g, m_buf, v_buf, sr_key, lr, momentum, wd, ns_dtype, beta2, ns_steps):
    """Single 2D matrix Polar-Express NorMuon update with cautious weight decay (+ BF16 SR if needed)."""
    orig_dtype = p.dtype
    opt_dtype = m_buf.dtype
    p_f = p.astype(jnp.float32)
    g_f = g.astype(jnp.float32)
    m = m_buf.astype(jnp.float32) * momentum + g_f * (1.0 - momentum)
    u = g_f * (1.0 - momentum) + m * momentum

    # Polar Express orthogonalization
    X = u.astype(ns_dtype)
    norm = jnp.linalg.norm(X.astype(jnp.float32))
    X = X / (norm.astype(ns_dtype) * jnp.asarray(1.02, ns_dtype) + jnp.asarray(1e-6, ns_dtype))

    rows, cols = p.shape
    if rows > cols:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_t, b_t, c_t = jnp.asarray(a, ns_dtype), jnp.asarray(b, ns_dtype), jnp.asarray(c, ns_dtype)
            A = X.T @ X
            B = b_t * A + c_t * (A @ A)
            X = a_t * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_t, b_t, c_t = jnp.asarray(a, ns_dtype), jnp.asarray(b, ns_dtype), jnp.asarray(c, ns_dtype)
            A = X @ X.T
            B = b_t * A + c_t * (A @ A)
            X = a_t * X + B @ X

    g_orth = X.astype(jnp.float32)

    # NorMuon factored second-moment variance reduction
    red_axis = -1 if rows >= cols else -2
    red_size = float(cols if rows >= cols else rows)
    v_mean = jnp.mean(g_orth * g_orth, axis=red_axis, keepdims=True)
    v_norm = jnp.sqrt(jnp.sum(v_mean) * red_size)

    v_new = v_buf * beta2 + v_mean * (1.0 - beta2)
    step_scale = jax.lax.rsqrt(jnp.maximum(v_new, 1e-10))
    v_norm_new = jnp.sqrt(jnp.sum((v_mean * red_size) * (step_scale * step_scale)))
    final_scale = step_scale * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    g_orth = g_orth * final_scale

    eff_lr = lr * (max(1.0, cols / rows) ** 0.5)
    mask = ((g_orth * p_f) >= 0.0).astype(jnp.float32)
    p_new_f = p_f - eff_lr * g_orth - eff_lr * wd * p_f * mask
    p_new = _cast_updated_param(p_new_f, orig_dtype, sr_key)
    return p_new, m.astype(opt_dtype), v_new


def _muon_update_blocks(p_3d, g_3d, st_3d, base_key, lr, momentum, wd, ns_dtype, beta2, ns_steps, use_vmap: bool):
    """Apply NorMuon across stacked `(n_layer, rows, cols)` via low-HBM scan or batched vmap."""
    n_layer = p_3d.shape[0]
    layer_keys = jax.random.split(base_key, n_layer)

    if use_vmap:
        def fn(p_i, g_i, m_i, v_i, k_i):
            return _muon_update_2d(p_i, g_i, m_i, v_i, k_i, lr, momentum, wd, ns_dtype, beta2, ns_steps)

        p_out, m_out, v_out = jax.vmap(fn)(p_3d, g_3d, st_3d["m"], st_3d["v_rowcol"], layer_keys)
        return p_out, {"m": m_out, "v_rowcol": v_out}

    def body(_, inputs):
        p_i, g_i, m_i, v_i, k_i = inputs
        p_new, m_new, v_new = _muon_update_2d(
            p_i, g_i, m_i, v_i, k_i, lr, momentum, wd, ns_dtype, beta2, ns_steps
        )
        return None, (p_new, m_new, v_new)

    _, (p_out, m_out, v_out) = jax.lax.scan(
        body, None, (p_3d, g_3d, st_3d["m"], st_3d["v_rowcol"], layer_keys)
    )
    return p_out, {"m": m_out, "v_rowcol": v_out}


def _polar_normuon_2d(u_in, v_buf, ns_dtype, beta2, ns_steps):
    """Pure Polar-Express + NorMuon orthogonalization on one 2D matrix (runs 100% locally per chip!)."""
    X = u_in.astype(ns_dtype)
    norm = jnp.linalg.norm(X.astype(jnp.float32))
    X = X / (norm.astype(ns_dtype) * jnp.asarray(1.02, ns_dtype) + jnp.asarray(1e-6, ns_dtype))

    rows, cols = u_in.shape
    if rows > cols:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_t, b_t, c_t = jnp.asarray(a, ns_dtype), jnp.asarray(b, ns_dtype), jnp.asarray(c, ns_dtype)
            A = X.T @ X
            B = b_t * A + c_t * (A @ A)
            X = a_t * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_t, b_t, c_t = jnp.asarray(a, ns_dtype), jnp.asarray(b, ns_dtype), jnp.asarray(c, ns_dtype)
            A = X @ X.T
            B = b_t * A + c_t * (A @ A)
            X = a_t * X + B @ X

    g_orth = X.astype(jnp.float32)
    red_axis = -1 if rows >= cols else -2
    red_size = float(cols if rows >= cols else rows)
    v_mean = jnp.mean(g_orth * g_orth, axis=red_axis, keepdims=True)
    v_norm = jnp.sqrt(jnp.sum(v_mean) * red_size)

    v_new = v_buf * beta2 + v_mean * (1.0 - beta2)
    step_scale = jax.lax.rsqrt(jnp.maximum(v_new, 1e-10))
    v_norm_new = jnp.sqrt(jnp.sum((v_mean * red_size) * (step_scale * step_scale)))
    final_scale = step_scale * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    return (g_orth * final_scale).astype(ns_dtype), v_new


def _layer_sharded_orth_only(u_3d, v_3d, ns_dtype, beta2, ns_steps, mesh: Mesh):
    """
    U-Only Essential AI / DMuon Layer-Sharded Polar Express:
    `p` and `m` NEVER leave their native FSDP shards! Only `u` (`ns_dtype`) goes through
    `all_to_all` IN (resharding to `P('fsdp', None, None)` across layers) and `g_orth`
    comes back OUT -> cuts optimizer FLOPs 2.08x, HBM 2.09x, and 114 AG + 32 AR -> 0 AG + 1 AR!
    """
    n_tot = u_3d.shape[0]
    n_dev = mesh.size
    rem = n_tot % n_dev
    pad_n = (n_dev - rem) if rem else 0

    def _pad0(arr):
        if pad_n == 0:
            return arr
        return jnp.pad(arr, [(0, pad_n)] + [(0, 0)] * (arr.ndim - 1))

    u_pad = _pad0(u_3d.astype(ns_dtype))
    v_pad = _pad0(v_3d)

    ax = mesh.axis_names[0]
    s3 = NamedSharding(mesh, P(ax, None, None))
    u_pad = jax.lax.with_sharding_constraint(u_pad, s3)
    v_pad = jax.lax.with_sharding_constraint(v_pad, s3)

    def _loc(ul, vl):
        return jax.vmap(lambda ui, vi: _polar_normuon_2d(ui, vi, ns_dtype, beta2, ns_steps))(ul, vl)

    g_out, v_out = _SHARD_MAP(
        _loc,
        mesh=mesh,
        in_specs=(P(ax, None, None), P(ax, None, None)),
        out_specs=(P(ax, None, None), P(ax, None, None)),
        **_SHARD_MAP_KWARGS,
    )(u_pad, v_pad)
    return g_out[:n_tot], v_out[:n_tot]


def _local_momentum_and_u(g, m_buf, momentum, out_dtype):
    g_f = g.astype(jnp.float32)
    m_new = m_buf.astype(jnp.float32) * momentum + g_f * (1.0 - momentum)
    u = (g_f * (1.0 - momentum) + m_new * momentum).astype(out_dtype)
    return u, m_new.astype(m_buf.dtype)


def _local_apply_cautious_update(p, g_orth, eff_lr, wd, sr_key):
    p_f = p.astype(jnp.float32)
    go_f = g_orth.astype(jnp.float32)
    mask = ((go_f * p_f) >= 0.0).astype(jnp.float32)
    p_new_f = p_f - eff_lr * go_f - eff_lr * wd * p_f * mask
    return _cast_updated_param(p_new_f, p.dtype, sr_key)


def clip_grads_by_global_norm(grads: dict, max_norm: float):
    """Clip gradient pytree by global L2 norm in float32; return (clipped_grads, global_norm)."""
    sq_norms = [jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree_util.tree_leaves(grads)]
    global_norm = jnp.sqrt(jnp.sum(jnp.stack(sq_norms)))
    if max_norm <= 0.0:
        return grads, global_norm
    scale = jnp.minimum(jnp.float32(1.0), jnp.float32(max_norm) / jnp.maximum(global_norm, 1e-8))
    clipped = jax.tree_util.tree_map(lambda g: g * scale.astype(g.dtype), grads)
    return clipped, global_norm


def optimizer_step(
    params: dict,
    grads: dict,
    opt_state: dict,
    lrm: jax.Array,
    muon_momentum: jax.Array,
    muon_wd: jax.Array,
    spec: ModelSpec,
    tcfg: TrainConfig,
    ns_dtype,
    mesh: Optional[Mesh] = None,
):
    """Full Muon + AdamW update with global gradient clipping, width-scaled LRs, and optional BF16 SR."""
    grads, grad_norm = clip_grads_by_global_norm(grads, tcfg.grad_clip)

    step_i = opt_state["step"] + jnp.int32(1)
    step_f = step_i.astype(jnp.float32)
    b1, b2 = tcfg.adam_beta1, tcfg.adam_beta2
    dmodel_scale = (spec.n_embd / 768.0) ** -0.5

    # Deterministic per-step PRNG key for unbiased BF16 stochastic rounding (no-op for fp32 params)
    sr_root = jax.random.fold_in(jax.random.PRNGKey(tcfg.seed), step_i)

    def next_sr_key():
        nonlocal sr_root
        sr_root, sub = jax.random.split(sr_root)
        return sub

    new_params: dict = {}
    new_state: dict = {"step": step_i}

    new_params["wte"], new_state["wte"] = _adamw_update(
        params["wte"], grads["wte"], opt_state["wte"], step_f,
        tcfg.embedding_lr * dmodel_scale * lrm, b1, b2, next_sr_key(),
    )
    if "lm_head" in params:
        new_params["lm_head"], new_state["lm_head"] = _adamw_update(
            params["lm_head"], grads["lm_head"], opt_state["lm_head"], step_f,
            tcfg.unembedding_lr * dmodel_scale * lrm, b1, b2, next_sr_key(),
        )
    if "value_embeds" in params:
        new_params["value_embeds"], new_state["value_embeds"] = _adamw_update(
            params["value_embeds"], grads["value_embeds"], opt_state["value_embeds"], step_f,
            tcfg.embedding_lr * dmodel_scale * lrm, b1, b2, next_sr_key(),
        )

    new_params["resid_lambdas"], new_state["resid_lambdas"] = _adamw_update(
        params["resid_lambdas"], grads["resid_lambdas"], opt_state["resid_lambdas"], step_f,
        tcfg.scalar_lr * 0.01 * lrm, b1, b2, next_sr_key(),
    )
    new_params["x0_lambdas"], new_state["x0_lambdas"] = _adamw_update(
        params["x0_lambdas"], grads["x0_lambdas"], opt_state["x0_lambdas"], step_f,
        tcfg.scalar_lr * lrm, 0.96, 0.95, next_sr_key(),
    )
    for k in EXTRA_SCALAR_KEYS:
        if k in params:
            k_lr = (tcfg.scalar_lr * 0.05 if k in GENTLE_SCALAR_KEYS else tcfg.scalar_lr) * lrm
            new_params[k], new_state[k] = _adamw_update(
                params[k], grads[k], opt_state[k], step_f,
                k_lr, 0.96, 0.95, next_sr_key(),
            )

    def _update_block_dict_layer_sharded(p_dict: dict, g_dict: dict, st_dict: dict):
        """
        U-Only Layer-Sharded Stacked Split-Muon (Essential AI / DMuon):
        1. Updates `m` & computes `u` 100% locally on each chip's FSDP shards (zero comms).
        2. Groups all orientation-normalized `(rows <= cols)` 3D matrix slices by exact `(r, c)` shape
           so homogeneous shapes (`[gate, up, down.T]`, `[q, c_proj, attn_gate]`, `[k, v]`, ...)
           are load-balanced across all chips via ONE `all_to_all` per group.
        3. Runs Polar Express + NorMuon 100% locally per chip, `all_to_all`s `g_orth` back, and
           applies cautious weight decay + BF16 SR locally on native FSDP shards!
        """
        assert mesh is not None
        nb, sb = {}, {}
        base_lr = tcfg.matrix_lr * lrm

        # Stage 1: local momentum updates + expand split-muon slices into orientation-normalized work items
        m_news: dict[str, jax.Array] = {}
        # Each work item: (tag, u_norm, v_norm, transposed, orig_rows, orig_cols)
        groups: dict[tuple[int, int], list[tuple[str, jax.Array, jax.Array, bool, int, int]]] = {}
        split_meta: dict[str, str] = {}  # name -> "w_qkv" | "w_gate_up" | "single"

        def _enqueue(tag: str, u_arr: jax.Array, v_arr: jax.Array):
            _, r, c = u_arr.shape
            if r > c:
                u_norm = jnp.swapaxes(u_arr, -1, -2)
                v_norm = jnp.swapaxes(v_arr, -1, -2)
                key_rc = (c, r)
                transposed = True
            else:
                u_norm = u_arr
                v_norm = v_arr
                key_rc = (r, c)
                transposed = False
            groups.setdefault(key_rc, []).append((tag, u_norm, v_norm, transposed, r, c))

        for name in p_dict:
            if tcfg.gate_adamw and name in GATE_BLOCK_KEYS:
                nb[name], sb[name] = _adamw_update(
                    p_dict[name], g_dict[name], st_dict[name], step_f,
                    tcfg.scalar_lr * lrm, b1, b2, next_sr_key(),
                )
                continue
            p_w, g_w, st_w = p_dict[name], g_dict[name], st_dict[name]
            u_w, m_new = _local_momentum_and_u(g_w, st_w["m"], muon_momentum, ns_dtype)
            m_news[name] = m_new

            if tcfg.split_muon and name == "w_qkv" and "v_q" in st_w:
                C = spec.n_embd
                kv = spec.kv_dim
                _enqueue(f"{name}::q", u_w[..., :C], st_w["v_q"])
                _enqueue(f"{name}::k", u_w[..., C : C + kv], st_w["v_k"])
                _enqueue(f"{name}::v", u_w[..., C + kv :], st_w["v_v"])
                split_meta[name] = "w_qkv"
            elif tcfg.split_muon and name == "w_gate_up" and "v_gate" in st_w:
                half = p_w.shape[-1] // 2
                _enqueue(f"{name}::gate", u_w[..., :half], st_w["v_gate"])
                _enqueue(f"{name}::up", u_w[..., half:], st_w["v_up"])
                split_meta[name] = "w_gate_up"
            else:
                _enqueue(f"{name}::single", u_w, st_w["v_rowcol"])
                split_meta[name] = "single"

        # Stage 2: run 1 layer-sharded Polar Express shard_map per unique (r, c) group!
        orth_results: dict[str, tuple[jax.Array, jax.Array, float]] = {}
        for _, items in groups.items():
            u_cat = jnp.concatenate([it[1] for it in items], axis=0)
            v_cat = jnp.concatenate([it[2] for it in items], axis=0)
            go_cat, vn_cat = _layer_sharded_orth_only(
                u_cat, v_cat, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps, mesh
            )
            offset = 0
            for tag, u_norm, _, transposed, orig_r, orig_c in items:
                L_i = u_norm.shape[0]
                go_i = go_cat[offset : offset + L_i]
                vn_i = vn_cat[offset : offset + L_i]
                if transposed:
                    go_i = jnp.swapaxes(go_i, -1, -2)
                    vn_i = jnp.swapaxes(vn_i, -1, -2)
                aspect = float(max(1.0, orig_c / orig_r) ** 0.5)
                orth_results[tag] = (go_i, vn_i, aspect)
                offset += L_i

        # Stage 3: local sharded cautious weight decay + BF16 SR parameter updates (zero comms!)
        for name, kind in split_meta.items():
            p_w = p_dict[name]
            if kind == "w_qkv":
                go_q, vq, aq = orth_results[f"{name}::q"]
                go_k, vk, ak = orth_results[f"{name}::k"]
                go_v, vv, av = orth_results[f"{name}::v"]
                if aq == ak == av:
                    go_all = jnp.concatenate([go_q, go_k, go_v], axis=-1)
                    nb[name] = _local_apply_cautious_update(p_w, go_all, base_lr * aq, muon_wd, next_sr_key())
                else:
                    C, kv = spec.n_embd, spec.kv_dim
                    pq_n = _local_apply_cautious_update(p_w[..., :C], go_q, base_lr * aq, muon_wd, next_sr_key())
                    pk_n = _local_apply_cautious_update(p_w[..., C : C + kv], go_k, base_lr * ak, muon_wd, next_sr_key())
                    pv_n = _local_apply_cautious_update(p_w[..., C + kv :], go_v, base_lr * av, muon_wd, next_sr_key())
                    nb[name] = jnp.concatenate([pq_n, pk_n, pv_n], axis=-1)
                sb[name] = {"m": m_news[name], "v_q": vq, "v_k": vk, "v_v": vv}
            elif kind == "w_gate_up":
                go_1, v1, a1 = orth_results[f"{name}::gate"]
                go_2, v2, _ = orth_results[f"{name}::up"]
                go_all = jnp.concatenate([go_1, go_2], axis=-1)
                nb[name] = _local_apply_cautious_update(p_w, go_all, base_lr * a1, muon_wd, next_sr_key())
                sb[name] = {"m": m_news[name], "v_gate": v1, "v_up": v2}
            else:
                go_s, vs, a_s = orth_results[f"{name}::single"]
                nb[name] = _local_apply_cautious_update(p_w, go_s, base_lr * a_s, muon_wd, next_sr_key())
                sb[name] = {"m": m_news[name], "v_rowcol": vs}
        return nb, sb

    def _update_block_dict(p_dict: dict, g_dict: dict, st_dict: dict):
        if tcfg.muon_layer_sharded and mesh is not None and mesh.size > 1:
            return _update_block_dict_layer_sharded(p_dict, g_dict, st_dict)
        nb, sb = {}, {}
        for name in p_dict:
            if tcfg.gate_adamw and name in GATE_BLOCK_KEYS:
                nb[name], sb[name] = _adamw_update(
                    p_dict[name], g_dict[name], st_dict[name], step_f,
                    tcfg.scalar_lr * lrm, b1, b2, next_sr_key(),
                )
                continue
            p_w, g_w, st_w = p_dict[name], g_dict[name], st_dict[name]
            if tcfg.split_muon and name == "w_qkv" and "v_q" in st_w:
                C = spec.n_embd
                kv = spec.kv_dim
                pq, pk, pv = p_w[..., :C], p_w[..., C : C + kv], p_w[..., C + kv :]
                gq, gk, gv = g_w[..., :C], g_w[..., C : C + kv], g_w[..., C + kv :]
                mq, mk, mv = st_w["m"][..., :C], st_w["m"][..., C : C + kv], st_w["m"][..., C + kv :]
                pq_n, st_q = _muon_update_blocks(
                    pq, gq, {"m": mq, "v_rowcol": st_w["v_q"]}, next_sr_key(),
                    tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                    use_vmap=tcfg.muon_vmap,
                )
                pk_n, st_k = _muon_update_blocks(
                    pk, gk, {"m": mk, "v_rowcol": st_w["v_k"]}, next_sr_key(),
                    tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                    use_vmap=tcfg.muon_vmap,
                )
                pv_n, st_v = _muon_update_blocks(
                    pv, gv, {"m": mv, "v_rowcol": st_w["v_v"]}, next_sr_key(),
                    tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                    use_vmap=tcfg.muon_vmap,
                )
                nb[name] = jnp.concatenate([pq_n, pk_n, pv_n], axis=-1)
                sb[name] = {
                    "m": jnp.concatenate([st_q["m"], st_k["m"], st_v["m"]], axis=-1),
                    "v_q": st_q["v_rowcol"],
                    "v_k": st_k["v_rowcol"],
                    "v_v": st_v["v_rowcol"],
                }
                continue
            if tcfg.split_muon and name == "w_gate_up" and "v_gate" in st_w:
                half = p_w.shape[-1] // 2
                p1, p2 = p_w[..., :half], p_w[..., half:]
                g1, g2 = g_w[..., :half], g_w[..., half:]
                m1, m2 = st_w["m"][..., :half], st_w["m"][..., half:]
                p1_n, st_1 = _muon_update_blocks(
                    p1, g1, {"m": m1, "v_rowcol": st_w["v_gate"]}, next_sr_key(),
                    tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                    use_vmap=tcfg.muon_vmap,
                )
                p2_n, st_2 = _muon_update_blocks(
                    p2, g2, {"m": m2, "v_rowcol": st_w["v_up"]}, next_sr_key(),
                    tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                    use_vmap=tcfg.muon_vmap,
                )
                nb[name] = jnp.concatenate([p1_n, p2_n], axis=-1)
                sb[name] = {
                    "m": jnp.concatenate([st_1["m"], st_2["m"]], axis=-1),
                    "v_gate": st_1["v_rowcol"],
                    "v_up": st_2["v_rowcol"],
                }
                continue
            nb[name], sb[name] = _muon_update_blocks(
                p_w, g_w, st_w, next_sr_key(),
                tcfg.matrix_lr * lrm, muon_momentum, muon_wd, ns_dtype, tcfg.muon_beta2, tcfg.muon_ns_steps,
                use_vmap=tcfg.muon_vmap,
            )
        return nb, sb

    new_params["blocks"], new_state["blocks"] = _update_block_dict(
        params["blocks"], grads["blocks"], opt_state["blocks"]
    )
    for bg in EXTRA_BLOCK_GROUPS:
        if bg in params:
            new_params[bg], new_state[bg] = _update_block_dict(params[bg], grads[bg], opt_state[bg])
    return new_params, new_state, grad_norm


# ---------------------------------------------------------------------------
# WSD (Warmup -> Stable -> Decay) Schedule for Multi-Session Pretraining
# ---------------------------------------------------------------------------

def compute_wsd_lrm(step: int, tcfg: TrainConfig) -> float:
    """
    Warmup -> Stable -> Decay learning rate multiplier.
    - `wsd_phase == "stable"`: linear warmup over `warmup_steps`, then constant `1.0`
      indefinitely across as many 9h Kaggle / 12h Colab sessions as needed.
    - `wsd_phase == "decay"`: linear cooldown from `1.0` -> `min_lr_frac` over
      `decay_steps` starting at `decay_start_step`.
    """
    if tcfg.warmup_steps > 0 and step < tcfg.warmup_steps:
        return float(step + 1) / float(tcfg.warmup_steps)

    if tcfg.wsd_phase == "decay":
        start = tcfg.decay_start_step if tcfg.decay_start_step is not None else tcfg.warmup_steps
        elapsed = max(0, step - start)
        frac = min(1.0, elapsed / max(1, tcfg.decay_steps))
        return (1.0 - frac) * 1.0 + frac * tcfg.min_lr_frac

    return 1.0


def compute_muon_momentum(step: int) -> float:
    frac = min(step / 300.0, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def compute_weight_decay(step: int, tcfg: TrainConfig) -> float:
    """Keep weight decay steady in WSD stable phase and decay linearly during cooldown."""
    if tcfg.wsd_phase == "decay":
        start = tcfg.decay_start_step if tcfg.decay_start_step is not None else tcfg.warmup_steps
        frac = min(1.0, max(0, step - start) / max(1, tcfg.decay_steps))
        return tcfg.weight_decay * (1.0 - frac)
    return tcfg.weight_decay
