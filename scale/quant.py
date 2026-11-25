"""
Experimental high-MFU quantization & low-precision training primitives in pure JAX:
  1. `stochastic_round_bf16(x_f32, key)` — unbiased IEEE-754 bitwise float32 -> bfloat16
     stochastic rounding for pure-BF16 master weights (`--param-dtype bfloat16`), cutting
     1B weight HBM by ~3x (5.93 GB -> 1.98 GB) and eliminating fp32->bf16 cast overhead.
  2. `qdot(x, w, mode)` — dynamic quantized linear matmul (`@jax.custom_vjp`):
     - `mode="none"`:     standard `jnp.dot(x, w)`
     - `mode="int8"`:     AQT-style dynamic INT8 forward + INT8 d_x (2x MXU throughput on
                          TPU v5e 394 TOPS!) with high-precision bf16 d_w
     - `mode="int8-all"`: dynamic INT8 for all 3 GEMMs (fwd, d_x, d_w) for peak TPU v5e TOPS
     - `mode="fp8"`:      native `float8_e4m3fn` (fwd) + `float8_e5m2` (bwd d_x) for TPU v6e
                          Trillium (1,836 TFLOPS) and Blackwell RTX PRO 6000 G4 (1,008 TFLOPS)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

VALID_MATMUL_QUANT = ("none", "int8", "int8-all", "fp8")


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


def _quantize_int8(x: jax.Array, axis: int):
    """Symmetric dynamic absmax int8 quantization along `axis` -> `(x_i8, scale_f32)`."""
    max_abs = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    scale = jnp.maximum(max_abs, jnp.float32(1e-8)) / jnp.float32(127.0)
    q = jnp.clip(jnp.rint(x.astype(jnp.float32) / scale), -127.0, 127.0).astype(jnp.int8)
    return q, scale


def _quantize_fp8(x: jax.Array, axis: int, fp8_dtype, max_val: float):
    """Symmetric dynamic absmax fp8 quantization along `axis` -> `(x_fp8, scale_f32)`."""
    max_abs = jnp.max(jnp.abs(x), axis=axis, keepdims=True).astype(jnp.float32)
    scale = jnp.maximum(max_abs, jnp.float32(1e-8)) / jnp.float32(max_val)
    q = jnp.clip(x.astype(jnp.float32) / scale, -max_val, max_val).astype(fp8_dtype)
    return q, scale


@jax.custom_vjp
def _qdot_int8_hybrid(x: jax.Array, w: jax.Array) -> jax.Array:
    """INT8 fwd + INT8 bwd_x + high-precision bf16 bwd_w (2D `(M, K) @ (K, N)`)."""
    x_q, x_s = _quantize_int8(x, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    return (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)


def _qdot_int8_hybrid_fwd(x: jax.Array, w: jax.Array):
    x_q, x_s = _quantize_int8(x, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    y = (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)
    return y, (x, w)


def _qdot_int8_hybrid_bwd(res, g: jax.Array):
    x, w = res
    # d_x = g @ w.T in INT8 (per-token g, per-in-feature w)
    g_q, g_s = _quantize_int8(g, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=-1)
    dx_i32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.int32)
    dx = (dx_i32.astype(jnp.float32) * (g_s * w_s.T)).astype(x.dtype)
    # High-precision bf16 weight gradient for rock-solid convergence
    dw = jnp.dot(x.T, g).astype(w.dtype)
    return dx, dw


_qdot_int8_hybrid.defvjp(_qdot_int8_hybrid_fwd, _qdot_int8_hybrid_bwd)


@jax.custom_vjp
def _qdot_int8_all(x: jax.Array, w: jax.Array) -> jax.Array:
    """All-INT8 (fwd, d_x, d_w) for maximum TPU v5e 394-TOPS INT8 throughput."""
    x_q, x_s = _quantize_int8(x, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    return (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)


def _qdot_int8_all_fwd(x: jax.Array, w: jax.Array):
    x_q, x_s = _quantize_int8(x, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=0)
    y_i32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.int32)
    y = (y_i32.astype(jnp.float32) * (x_s * w_s)).astype(x.dtype)
    return y, (x, w)


def _qdot_int8_all_bwd(res, g: jax.Array):
    x, w = res
    g_q, g_s = _quantize_int8(g, axis=-1)
    w_q, w_s = _quantize_int8(w, axis=-1)
    dx_i32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.int32)
    dx = (dx_i32.astype(jnp.float32) * (g_s * w_s.T)).astype(x.dtype)

    # d_w = x.T @ g in INT8 (quantize along contraction axis 0 = tokens!)
    xt_q, xt_s = _quantize_int8(x, axis=0)
    gt_q, gt_s = _quantize_int8(g, axis=0)
    dw_i32 = jnp.dot(xt_q.T, gt_q, preferred_element_type=jnp.int32)
    dw = (dw_i32.astype(jnp.float32) * (xt_s.T * gt_s)).astype(w.dtype)
    return dx, dw


_qdot_int8_all.defvjp(_qdot_int8_all_fwd, _qdot_int8_all_bwd)


@jax.custom_vjp
def _qdot_fp8(x: jax.Array, w: jax.Array) -> jax.Array:
    """Native FP8 E4M3 forward + E5M2 backward d_x + bf16 d_w (TPU v6e / Blackwell G4)."""
    x_q, x_s = _quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    w_q, w_s = _quantize_fp8(w, axis=0, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    y_f32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.float32)
    return (y_f32 * (x_s * w_s)).astype(x.dtype)


def _qdot_fp8_fwd(x: jax.Array, w: jax.Array):
    x_q, x_s = _quantize_fp8(x, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    w_q, w_s = _quantize_fp8(w, axis=0, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    y_f32 = jnp.dot(x_q, w_q, preferred_element_type=jnp.float32)
    return (y_f32 * (x_s * w_s)).astype(x.dtype), (x, w)


def _qdot_fp8_bwd(res, g: jax.Array):
    x, w = res
    g_q, g_s = _quantize_fp8(g, axis=-1, fp8_dtype=jnp.float8_e5m2, max_val=57344.0)
    w_q, w_s = _quantize_fp8(w, axis=-1, fp8_dtype=jnp.float8_e4m3fn, max_val=448.0)
    dx_f32 = jnp.dot(g_q, w_q.T, preferred_element_type=jnp.float32)
    dx = (dx_f32 * (g_s * w_s.T)).astype(x.dtype)
    dw = jnp.dot(x.T, g).astype(w.dtype)
    return dx, dw


_qdot_fp8.defvjp(_qdot_fp8_fwd, _qdot_fp8_bwd)


def qdot(x: jax.Array, w: jax.Array, mode: str = "none") -> jax.Array:
    """Dispatch 2D linear matmul `(M, K) @ (K, N)` to BF16/FP16/FP32, INT8, or FP8."""
    if mode == "none" or x.ndim != 2 or w.ndim != 2:
        return jnp.dot(x, w)
    if mode == "int8":
        return _qdot_int8_hybrid(x, w)
    if mode == "int8-all":
        return _qdot_int8_all(x, w)
    if mode == "fp8":
        return _qdot_fp8(x, w)
    raise ValueError(f"Unsupported matmul_quant mode: {mode!r}. Choose from {VALID_MATMUL_QUANT}")
