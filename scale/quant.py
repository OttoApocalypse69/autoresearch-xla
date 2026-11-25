"""
Experimental high-MFU quantization & low-precision training primitives in pure JAX:
  1. `stochastic_round_bf16(x_f32, key)` — unbiased IEEE-754 bitwise float32 -> bfloat16
     stochastic rounding for pure-BF16 master weights (`--param-dtype bfloat16`), cutting
     1B weight HBM by ~3x (5.93 GB -> 1.98 GB) and eliminating fp32->bf16 cast overhead.
  2. Fast reciprocal-multiply quantization (`x * inv_scale`, zero per-element VPU divisions).
  3. `prequantize_block_weights(blocks, mode)` — pre-quantizes frozen linear weights ONCE
     per optimizer step before gradient-accumulation scan:
       - Eliminates 2 * grad_accum redundant weight quantizations per step (e.g. 64x -> 2x)
       - Halves weight HBM reads (1 byte int8/fp8 vs 2 bytes bf16)
       - Halves multi-chip FSDP AllGather ICI traffic on Kaggle TPU v5e-8 (1-byte AllGather!)
  4. INT8 activation checkpointing in `int8-all`: saves forward activations as 1-byte `int8`
     (`xt_q, xt_s`) instead of 2-byte `bf16` `x` for backward `d_w`, cutting saved linear
     activation HBM in half (unlocks larger microbatches like `b=4` on 16 GB `v5e-1`!).
  5. FSDP-local `bf16` `d_w` ReduceScatter (`shard_map` + `psum_scatter`): on multi-chip
     `v5e-8`, each chip quantizes its own local tokens (zero AllReduce(MAX), 8x tighter
     dynamic range) and reduces `d_w` in 2-byte `bf16` instead of 4-byte `int32`!
"""

from __future__ import annotations

import inspect
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

VALID_MATMUL_QUANT = ("none", "int8", "int8-all", "int8-fast", "fp8")
LINEAR_BLOCK_KEYS = ("w_qkv", "c_proj", "w_gate_up", "mlp_fc", "mlp_down")

_SHARD_MAP = getattr(jax, "shard_map", None)
if _SHARD_MAP is None:
    from jax.experimental.shard_map import shard_map as _SHARD_MAP  # type: ignore
_SHARD_MAP_PARAMS = inspect.signature(_SHARD_MAP).parameters
_SHARD_MAP_KWARGS = {"check_vma": False} if "check_vma" in _SHARD_MAP_PARAMS else {"check_rep": False}


def stochastic_round_bf16(x_f32: jax.Array, key: jax.Array) -> jax.Array:
    """
    Unbiased bitwise IEEE-754 float32 -> bfloat16 stochastic rounding.

    Adds uniform 16-bit random noise to the low 16 mantissa bits of `x_f32` before
    truncating to bfloat16. Because IEEE-754 is sign-magnitude, carrying into bit 16
    rounds away from zero with exact probability `frac_low16 / 65536` for BOTH positive
    and negative finite values, ensuring `E[SR(x)] == x` even for tiny `1e-5` updates.
    """
    x = x_f32.astype(jnp.float32)
    x_u32 = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rand_low16 = jax.random.bits(key, shape=x.shape, dtype=jnp.uint32) & jnp.uint32(0xFFFF)
    # Guard against NaN/Inf bit patterns (exponent all 1s = 0x7F800000)
    is_finite = (x_u32 & jnp.uint32(0x7F800000)) != jnp.uint32(0x7F800000)
    rounded_u32 = jnp.where(is_finite, (x_u32 + rand_low16) & jnp.uint32(0xFFFF0000), x_u32)
    return jax.lax.bitcast_convert_type(rounded_u32, jnp.float32).astype(jnp.bfloat16)


def quantize_int8(x: jax.Array, axis: int):
    """Fast symmetric dynamic absmax int8 quantization along `axis` via reciprocal multiply."""
    max_abs = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    safe_max = jnp.maximum(max_abs, jnp.float32(1e-8))
    inv_scale = jnp.float32(127.0) / safe_max
    scale = safe_max * jnp.float32(1.0 / 127.0)
    q = jnp.clip(jnp.rint(x.astype(jnp.float32) * inv_scale), -127.0, 127.0).astype(jnp.int8)
    return q, scale


def quantize_int8_tensor(x: jax.Array):
    """Single-scalar dynamic absmax int8 quantization (reusable for BOTH fwd/dx AND dw; zero axis-0 reductions!)."""
    max_abs = jnp.max(jnp.abs(x)).astype(jnp.float32)
    safe_max = jnp.maximum(max_abs, jnp.float32(1e-8))
    inv_scale = jnp.float32(127.0) / safe_max
    scale = (safe_max * jnp.float32(1.0 / 127.0)).reshape((1, 1))
    q = jnp.clip(jnp.rint(x.astype(jnp.float32) * inv_scale), -127.0, 127.0).astype(jnp.int8)
    return q, scale


def quantize_fp8(x: jax.Array, axis: int, fp8_dtype, max_val: float):
    """Fast symmetric dynamic absmax fp8 quantization along `axis` via reciprocal multiply."""
    max_abs = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    safe_max = jnp.maximum(max_abs, jnp.float32(1e-8))
    inv_scale = jnp.float32(max_val) / safe_max
    scale = safe_max * jnp.float32(1.0 / max_val)
    q = jnp.clip(x.astype(jnp.float32) * inv_scale, -max_val, max_val).astype(fp8_dtype)
    return q, scale


def prequantize_weight_pair(w: jax.Array, mode: str) -> Optional[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """
    Pre-quantize a weight tensor `w` with trailing `(..., K, N)` dims into
    `(w_q_fwd, w_s_fwd, w_q_bwd, w_s_bwd)` ONCE per optimizer step.
    """
    if mode in ("int8", "int8-all", "int8-fast"):
        w_qf, w_sf = quantize_int8(w, axis=-2)
        w_qb, w_sb = quantize_int8(w, axis=-1)
        return (w_qf, w_sf, w_qb, w_sb)
    if mode == "fp8":
        w_qf, w_sf = quantize_fp8(w, axis=-2, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
        w_qb, w_sb = quantize_fp8(w, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
        return (w_qf, w_sf, w_qb, w_sb)
    return None


def prequantize_model_weights(params: dict, mode: str, tie_embeddings: bool = False) -> Optional[dict]:
    """
    Pre-quantize all scanned block linear weights (+ lm_head bwd weight for `int8-all` / `int8-fast`)
    ONCE before the gradient-accumulation scan so microsteps read 1-byte weights and
    multi-chip FSDP AllGathers 1 byte/weight instead of 2 bytes BF16.
    """
    if mode == "none":
        return None

    q_blocks = {}
    blocks = params["blocks"]
    for k in LINEAR_BLOCK_KEYS:
        if k in blocks:
            q_blocks[k] = prequantize_weight_pair(blocks[k], mode)

    q_cache = {"blocks": q_blocks}
    if mode in ("int8-all", "int8-fast"):
        lm_w = params["wte"].T if tie_embeddings else params["lm_head"]
        q_cache["lm_head_bwd"] = quantize_int8(lm_w, axis=-1)
    return q_cache


def _choose_dw_scatter_dim(k_dim: int, n_dim: int, n_dev: int) -> Optional[int]:
    """Match `scale/fsdp.py` sharding axis choice for 2D `(K, N)` block weights."""
    candidates = []
    if k_dim % n_dev == 0:
        candidates.append((k_dim, 0))
    if n_dim % n_dev == 0:
        candidates.append((n_dim, 1))
    if not candidates:
        return None
    _, best_ax = max(candidates)
    return best_ax


def _quantize_x_for_dw_int8(x: jax.Array, mesh: Optional[Mesh]):
    """
    Quantize forward activation `x` `(M, K)` along token axis 0 into 1-byte `int8`
    for saved backward residuals (halving activation HBM!).
    On multi-chip FSDP (`mesh.size > 1`), quantizes locally per chip via `shard_map`
    so there is ZERO cross-chip AllReduce(MAX) and 8x tighter per-chip dynamic range.
    """
    if mesh is not None and mesh.size > 1 and x.shape[0] % mesh.size == 0:
        axis_name = mesh.axis_names[0]

        def local_q(x_loc):
            return quantize_int8(x_loc, axis=0)

        return _SHARD_MAP(
            local_q,
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=(P(axis_name, None), P(axis_name, None)),
            **_SHARD_MAP_KWARGS,
        )(x)
    return quantize_int8(x, axis=0)


def _dw_from_saved_xt_int8(
    xt_q: jax.Array,
    xt_s: jax.Array,
    g: jax.Array,
    out_dtype,
    mesh: Optional[Mesh],
) -> jax.Array:
    """
    Compute `d_w = x.T @ g` in INT8 from saved 1-byte `xt_q, xt_s`.
    On multi-chip FSDP (`mesh.size > 1`), computes local `int8 x int8 -> int32`, dequantizes
    locally to `out_dtype` (`bf16`), and performs a 2-byte `bf16` `psum_scatter` instead of
    a 4-byte `int32` ReduceScatter (2x lower ICI ReduceScatter bandwidth!).
    """
    k_dim = xt_q.shape[1]
    n_dim = g.shape[1]
    if mesh is not None and mesh.size > 1 and xt_q.shape[0] % mesh.size == 0:
        sd = _choose_dw_scatter_dim(k_dim, n_dim, mesh.size)
        if sd is not None:
            axis_name = mesh.axis_names[0]
            out_spec = P(axis_name, None) if sd == 0 else P(None, axis_name)

            def local_dw(xq_loc, xs_loc, g_loc):
                gq_loc, gs_loc = quantize_int8(g_loc, axis=0)
                dw_i32 = jnp.dot(xq_loc.T, gq_loc, preferred_element_type=jnp.int32)
                dw_loc = (dw_i32.astype(jnp.float32) * (xs_loc.T * gs_loc)).astype(out_dtype)
                return jax.lax.psum_scatter(dw_loc, axis_name, scatter_dimension=sd, tiled=True)

            return _SHARD_MAP(
                local_dw,
                mesh=mesh,
                in_specs=(P(axis_name, None), P(axis_name, None), P(axis_name, None)),
                out_specs=out_spec,
                **_SHARD_MAP_KWARGS,
            )(xt_q, xt_s, g)

    gt_q, gt_s = quantize_int8(g, axis=0)
    dw_i32 = jnp.dot(xt_q.T, gt_q, preferred_element_type=jnp.int32)
    return (dw_i32.astype(jnp.float32) * (xt_s.T * gt_s)).astype(out_dtype)


# ---------------------------------------------------------------------------
# On-the-fly qdot (used when prequantized cache is not provided, e.g. unit tests)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def _qdot_int8_hybrid(x: jax.Array, w: jax.Array) -> jax.Array:
    """INT8 fwd + INT8 bwd_x + high-precision bf16 bwd_w (2D `(M, K) @ (K, N)`)."""
    x_q, x_s = quantize_int8(x, axis=-1)
    w_q, w_s = quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    return (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)


def _qdot_int8_hybrid_fwd(x: jax.Array, w: jax.Array):
    x_q, x_s = quantize_int8(x, axis=-1)
    w_q, w_s = quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    y = (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)
    return y, (x, w)


def _qdot_int8_hybrid_bwd(res, g: jax.Array):
    x, w = res
    g_q, g_s = quantize_int8(g, axis=-1)
    w_q, w_s = quantize_int8(w, axis=-1)
    dx_i32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.int32)
    dx = (dx_i32.astype(jnp.float32) * (g_s * w_s.T)).astype(x.dtype)
    dw = jnp.dot(x.T, g).astype(w.dtype)
    return dx, dw


_qdot_int8_hybrid.defvjp(_qdot_int8_hybrid_fwd, _qdot_int8_hybrid_bwd)


def _make_qdot_int8_all(mesh: Optional[Mesh]):
    @jax.custom_vjp
    def _fn(x: jax.Array, w: jax.Array) -> jax.Array:
        x_q, x_s = quantize_int8(x, axis=-1)
        w_q, w_s = quantize_int8(w, axis=0)
        y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
        return (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)

    def _fwd(x: jax.Array, w: jax.Array):
        x_q, x_s = quantize_int8(x, axis=-1)
        w_q, w_s = quantize_int8(w, axis=0)
        y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
        y = (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)
        # Save activation as 1-byte INT8 instead of 2-byte BF16 (halves activation HBM!)
        xt_q, xt_s = _quantize_x_for_dw_int8(x, mesh)
        return y, (xt_q, xt_s, w)

    def _bwd(res, g: jax.Array):
        xt_q, xt_s, w = res
        g_q, g_s = quantize_int8(g, axis=-1)
        w_q, w_s = quantize_int8(w, axis=-1)
        dx_i32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.int32)
        dx = (dx_i32.astype(jnp.float32) * (g_s * w_s.T)).astype(g.dtype)
        dw = _dw_from_saved_xt_int8(xt_q, xt_s, g, w.dtype, mesh)
        return dx, dw

    _fn.defvjp(_fwd, _bwd)
    return _fn


@jax.custom_vjp
def _qdot_fp8(x: jax.Array, w: jax.Array) -> jax.Array:
    """Native FP8 E4M3 forward + E5M2 backward d_x + bf16 d_w (TPU v6e / Blackwell G4)."""
    x_q, x_s = quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    w_q, w_s = quantize_fp8(w, axis=0, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    y_f32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.float32)
    return (y_f32 * (x_s * w_s)).astype(x.dtype)


def _qdot_fp8_fwd(x: jax.Array, w: jax.Array):
    x_q, x_s = quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    w_q, w_s = quantize_fp8(w, axis=0, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    y_f32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.float32)
    return (y_f32 * (x_s * w_s)).astype(x.dtype), (x, w)


def _qdot_fp8_bwd(res, g: jax.Array):
    x, w = res
    g_q, g_s = quantize_fp8(g, axis=-1, fp8_dtype=jnp.float8_e5m2, max_val=57344.0)
    w_q, w_s = quantize_fp8(w, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    dx_f32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.float32)
    dx = (dx_f32 * (g_s * w_s.T)).astype(x.dtype)
    dw = jnp.dot(x.T, g).astype(w.dtype)
    return dx, dw


_qdot_fp8.defvjp(_qdot_fp8_fwd, _qdot_fp8_bwd)


# ---------------------------------------------------------------------------
# Pre-quantized qdot (ZERO BF16 weight reads/AllGathers during grad_accum!)
# ---------------------------------------------------------------------------

def _make_qdot_prequant(mode: str, mesh: Optional[Mesh]):
    if mode == "int8":
        @jax.custom_vjp
        def _fn(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_int8(x, axis=-1)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            return (y_i32.astype(jnp.float32) * (x_s * w_sf)).astype(x.dtype)

        def _fwd(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_int8(x, axis=-1)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            y = (y_i32.astype(jnp.float32) * (x_s * w_sf)).astype(x.dtype)
            return y, (x, w_qb, w_sb)

        def _bwd(res, g):
            x, w_qb, w_sb = res
            g_q, g_s = quantize_int8(g, axis=-1)
            dx_i32 = jnp.dot(g_q, w_qb.T, preferred_element_type=jnp.int32)
            dx = (dx_i32.astype(jnp.float32) * (g_s * w_sb.T)).astype(x.dtype)
            dw = jnp.dot(x.T, g).astype(x.dtype)
            return dx, dw, None, None, None, None

        _fn.defvjp(_fwd, _bwd)
        return _fn

    if mode == "int8-all":
        @jax.custom_vjp
        def _fn(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_int8(x, axis=-1)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            return (y_i32.astype(jnp.float32) * (x_s * w_sf)).astype(x.dtype)

        def _fwd(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_int8(x, axis=-1)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            y = (y_i32.astype(jnp.float32) * (x_s * w_sf)).astype(x.dtype)
            # Save 1-byte INT8 activation instead of 2-byte BF16 `x` (halves activation HBM!)
            xt_q, xt_s = _quantize_x_for_dw_int8(x, mesh)
            return y, (xt_q, xt_s, w_qb, w_sb)

        def _bwd(res, g):
            xt_q, xt_s, w_qb, w_sb = res
            g_q, g_s = quantize_int8(g, axis=-1)
            dx_i32 = jnp.dot(g_q, w_qb.T, preferred_element_type=jnp.int32)
            dx = (dx_i32.astype(jnp.float32) * (g_s * w_sb.T)).astype(g.dtype)
            dw = _dw_from_saved_xt_int8(xt_q, xt_s, g, g.dtype, mesh)
            return dx, dw, None, None, None, None

        _fn.defvjp(_fwd, _bwd)
        return _fn

    if mode == "int8-fast":
        @jax.custom_vjp
        def _fn(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = _quantize_tensor_local(x, mesh)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            return (_apply_chip_scale(y_i32.astype(jnp.float32), x_s, mesh) * w_sf).astype(x.dtype)

        def _fwd(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            # Quantize x ONCE per layer (1-byte int8) — reused directly for BOTH fwd AND bwd d_w!
            x_q, x_s = _quantize_tensor_local(x, mesh)
            y_i32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.int32)
            y = (_apply_chip_scale(y_i32.astype(jnp.float32), x_s, mesh) * w_sf).astype(x.dtype)
            return y, (x_q, x_s, w_qb, w_sb)

        def _bwd(res, g):
            x_q, x_s, w_qb, w_sb = res
            # Quantize g ONCE per layer (1-byte int8) — reused directly for BOTH d_x AND d_w!
            g_q, g_s = _quantize_tensor_local(g, mesh)
            dx_i32 = jnp.dot(g_q, w_qb.T, preferred_element_type=jnp.int32)
            dx = (_apply_chip_scale(dx_i32.astype(jnp.float32), g_s, mesh) * w_sb.T).astype(g.dtype)
            dw = _dw_from_shared_int8(x_q, x_s, g_q, g_s, g.dtype, mesh)
            return dx, dw, None, None, None, None

        _fn.defvjp(_fwd, _bwd)
        return _fn

    if mode == "fp8":
        @jax.custom_vjp
        def _fn(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
            y_f32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.float32)
            return (y_f32 * (x_s * w_sf)).astype(x.dtype)

        def _fwd(x, w_unused, w_qf, w_sf, w_qb, w_sb):
            x_q, x_s = quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
            y_f32 = jnp.dot(x_q, w_qf, preferred_element_type=jnp.float32)
            return (y_f32 * (x_s * w_sf)).astype(x.dtype), (x, w_qb, w_sb)

        def _bwd(res, g):
            x, w_qb, w_sb = res
            g_q, g_s = quantize_fp8(g, axis=-1, fp8_dtype=jnp.float8_e5m2, max_val=57344.0)
            dx_f32 = jnp.dot(g_q, w_qb.T, preferred_element_type=jnp.float32)
            dx = (dx_f32 * (g_s * w_sb.T)).astype(x.dtype)
            dw = jnp.dot(x.T, g).astype(x.dtype)
            return dx, dw, None, None, None, None

        _fn.defvjp(_fwd, _bwd)
        return _fn

    raise ValueError(f"Unsupported prequantized matmul_quant mode: {mode!r}")


def _quantize_tensor_local(x: jax.Array, mesh: Optional[Mesh]):
    """Per-chip scalar absmax int8 quantization (`x_q`: `(M, K)`, `x_s`: `(n_dev, 1)` or `(1, 1)`)."""
    if mesh is not None and mesh.size > 1 and x.shape[0] % mesh.size == 0:
        axis_name = mesh.axis_names[0]
        return _SHARD_MAP(
            quantize_int8_tensor,
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=(P(axis_name, None), P(axis_name, None)),
            **_SHARD_MAP_KWARGS,
        )(x)
    return quantize_int8_tensor(x)


def _apply_chip_scale(arr_2d: jax.Array, s_chip: jax.Array, mesh: Optional[Mesh]) -> jax.Array:
    """Multiply 2D `(M, N)` by per-chip scalar `(n_dev, 1)` (or `(1, 1)` on 1 chip) with zero comm."""
    if mesh is not None and mesh.size > 1 and s_chip.shape[0] == mesh.size:
        n_dev = mesh.size
        return (arr_2d.reshape(n_dev, -1, arr_2d.shape[-1]) * s_chip[:, None, :]).reshape(arr_2d.shape)
    return arr_2d * s_chip


def _dw_from_shared_int8(
    x_q: jax.Array,
    x_s: jax.Array,
    g_q: jax.Array,
    g_s: jax.Array,
    out_dtype,
    mesh: Optional[Mesh],
) -> jax.Array:
    """Compute `d_w = x_q.T @ g_q` reusing BOTH `x_q` and `g_q` (zero extra quants; bf16 psum_scatter on FSDP!)."""
    k_dim = x_q.shape[1]
    n_dim = g_q.shape[1]
    if mesh is not None and mesh.size > 1 and x_q.shape[0] % mesh.size == 0:
        sd = _choose_dw_scatter_dim(k_dim, n_dim, mesh.size)
        if sd is not None:
            axis_name = mesh.axis_names[0]
            out_spec = P(axis_name, None) if sd == 0 else P(None, axis_name)

            def local_dw(xq_loc, xs_loc, gq_loc, gs_loc):
                dw_i32 = jnp.dot(xq_loc.T, gq_loc, preferred_element_type=jnp.int32)
                dw_loc = (dw_i32.astype(jnp.float32) * (xs_loc * gs_loc)).astype(out_dtype)
                return jax.lax.psum_scatter(dw_loc, axis_name, scatter_dimension=sd, tiled=True)

            return _SHARD_MAP(
                local_dw,
                mesh=mesh,
                in_specs=(P(axis_name, None), P(axis_name, None), P(axis_name, None), P(axis_name, None)),
                out_specs=out_spec,
                **_SHARD_MAP_KWARGS,
            )(x_q, x_s, g_q, g_s)

    dw_i32 = jnp.dot(x_q.T, g_q, preferred_element_type=jnp.int32)
    return (dw_i32.astype(jnp.float32) * (x_s * g_s)).astype(out_dtype)


def _make_qdot_int8_fast(mesh: Optional[Mesh]):
    @jax.custom_vjp
    def _fn(x: jax.Array, w: jax.Array) -> jax.Array:
        x_q, x_s = _quantize_tensor_local(x, mesh)
        w_q, w_s = quantize_int8(w, axis=0)
        y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
        return (_apply_chip_scale(y_i32.astype(jnp.float32), x_s, mesh) * w_s).astype(x.dtype)

    def _fwd(x: jax.Array, w: jax.Array):
        x_q, x_s = _quantize_tensor_local(x, mesh)
        w_q, w_s = quantize_int8(w, axis=0)
        y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
        y = (_apply_chip_scale(y_i32.astype(jnp.float32), x_s, mesh) * w_s).astype(x.dtype)
        return y, (x_q, x_s, w)

    def _bwd(res, g: jax.Array):
        x_q, x_s, w = res
        g_q, g_s = _quantize_tensor_local(g, mesh)
        w_q, w_s = quantize_int8(w, axis=-1)
        dx_i32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.int32)
        dx = (_apply_chip_scale(dx_i32.astype(jnp.float32), g_s, mesh) * w_s.T).astype(g.dtype)
        dw = _dw_from_shared_int8(x_q, x_s, g_q, g_s, w.dtype, mesh)
        return dx, dw

    _fn.defvjp(_fwd, _bwd)
    return _fn


def qdot(
    x: jax.Array,
    w: jax.Array,
    mode: str = "none",
    q_weight: Optional[tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """
    Dispatch 2D linear matmul `(M, K) @ (K, N)` to BF16/FP16/FP32, INT8, or FP8.
    When `q_weight=(w_qf, w_sf, w_qb, w_sb)` is provided from `prequantize_model_weights`,
    reads 1-byte quantized weights directly with zero BF16 weight reads/AllGathers!
    """
    if mode == "none" or x.ndim != 2 or w.ndim != 2:
        return jnp.dot(x, w)
    if q_weight is not None:
        w_qf, w_sf, w_qb, w_sb = q_weight
        return _make_qdot_prequant(mode, mesh)(x, w, w_qf, w_sf, w_qb, w_sb)
    if mode == "int8":
        return _qdot_int8_hybrid(x, w)
    if mode == "int8-all":
        return _make_qdot_int8_all(mesh)(x, w)
    if mode == "int8-fast":
        return _make_qdot_int8_fast(mesh)(x, w)
    if mode == "fp8":
        return _qdot_fp8(x, w)
    raise ValueError(f"Unsupported matmul_quant mode: {mode!r}. Choose from {VALID_MATMUL_QUANT}")
