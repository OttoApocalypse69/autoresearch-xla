"""
FSDP-compatible Polar-Express NorMuon + AdamW optimizer with WSD schedule.

Uses `jax.lax.scan` over axis 0 (`n_layer`) of stacked `(n_layer, in_dim, out_dim)`
matrices during Polar Express orthogonalization so only ONE layer's temporary matrix
is ever materialized at a time — keeping peak HBM tiny even on a 1B model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from scale.config import ModelSpec, TrainConfig
from scale.quant import stochastic_round_bf16

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

    def _update_block_dict(p_dict: dict, g_dict: dict, st_dict: dict):
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
