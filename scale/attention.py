"""
High-MFU fused attention kernels for TPU (Pallas SplashAttention), Ampere+/Blackwell
GPU (cuDNN FlashAttention), and universal XLA fallback (T4 / P100 / CPU).

Critical numerical invariants verified:
- Pallas SplashAttention does NOT automatically divide Q by sqrt(head_dim), so we
  explicitly pre-scale `q = q * (head_dim ** -0.5)` before invoking the kernel.
- Pallas SplashAttention operates on `(H, T, D)` per batch element; we transpose
  `(B, T, H, D) -> (B, H, T, D)`, `vmap` over batch `B`, and wrap in `shard_map`
  when `mesh.size > 1` so multi-chip TPU FSDP (e.g. Kaggle v5e-8) executes local
  SplashAttention on every chip with zero cross-chip attention communication.
- Supports both full causal (`window_size=None`) and sliding-window causal
  (`window_size=(W, 0)`) via SplashAttention's `CausalMask` / `LocalMask`.
"""

from __future__ import annotations

import inspect
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Cross-version shard_map (jax.shard_map in JAX >=0.8 uses check_vma, legacy uses check_rep)
_SHARD_MAP = getattr(jax, "shard_map", None)
if _SHARD_MAP is None:
    from jax.experimental.shard_map import shard_map as _SHARD_MAP  # type: ignore
_SHARD_MAP_PARAMS = inspect.signature(_SHARD_MAP).parameters
_SHARD_MAP_KWARGS = {"check_vma": False} if "check_vma" in _SHARD_MAP_PARAMS else {"check_rep": False}


def _is_ampere_or_newer_gpu(device_kind: str) -> bool:
    k = device_kind.lower()
    # Turing (T4), Volta (V100), Pascal (P100) do NOT support cuDNN FlashAttention
    if any(old in k for old in ("t4", "v100", "p100", "p40", "k80", "m60")):
        return False
    return True


def resolve_attn_backend(requested: str = "auto") -> str:
    """Choose fastest numerically exact attention backend for the active JAX platform."""
    if requested != "auto":
        return requested
    dev0 = jax.devices()[0]
    platform = str(dev0.platform).lower()
    if platform == "tpu":
        try:
            from jax.experimental.pallas.ops.tpu import splash_attention  # noqa: F401

            return "splash"
        except Exception:
            return "xla"
    if platform == "gpu":
        kind = str(getattr(dev0, "device_kind", ""))
        return "cudnn" if _is_ampere_or_newer_gpu(kind) else "xla"
    return "xla"


def _splash_attention_tpu(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    window_size: Optional[tuple[int, int]],
    block_size: int = 512,
    mesh: Optional[Mesh] = None,
    interpret: bool = False,
) -> jax.Array:
    """
    TPU Pallas SplashAttention for `(B, T, H, D)` Q and `(B, T, KV, D)` K, V.
    Zero O(T^2) HBM materialization — tiles Q/K/V inside on-chip VMEM SRAM!
    """
    from jax.experimental.pallas.ops.tpu import splash_attention

    B, T, H, D = q.shape
    # SplashAttention requires T % 128 == 0 and D % 128 == 0 (or >= 128) for peak MXU tiling;
    # fall back cleanly to XLA if shapes are tiny (e.g. short smoke tests or tiny head_dim).
    if T < 128 or T % 128 != 0 or D % 128 != 0:
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, local_window_size=window_size)

    # Pick largest valid block size <= min(block_size, T) that is a multiple of 128
    blk = min(block_size, T)
    blk = max(128, (blk // 128) * 128)
    while T % blk != 0 and blk > 128:
        blk -= 128

    if window_size is None:
        base_mask = splash_attention.CausalMask(shape=(T, T))
    else:
        base_mask = splash_attention.LocalMask(shape=(T, T), window_size=window_size, offset=0)
    multi_mask = splash_attention.MultiHeadMask([base_mask] * H)

    try:
        bs = splash_attention.BlockSizes(
            block_q=blk,
            block_kv=blk,
            block_kv_compute=min(blk, 512),
            block_q_dkv=blk,
            block_kv_dkv=blk,
            block_kv_dkv_compute=min(blk, 512),
            use_fused_bwd_kernel=True,
        )
    except TypeError:
        bs = splash_attention.BlockSizes(
            block_q=blk,
            block_kv=blk,
            block_kv_compute=min(blk, 512),
            block_q_dkv=blk,
            block_kv_dkv=blk,
            block_kv_dkv_compute=min(blk, 512),
            block_q_dq=blk,
            block_kv_dq=blk,
        )
    mk_kwargs = {"block_sizes": bs}
    if interpret:
        mk_kwargs["interpret"] = True
    kernel = splash_attention.make_splash_mha_single_device(multi_mask, **mk_kwargs)

    # SplashAttention computes Q @ K^T without 1/sqrt(D) scaling -> pre-scale Q!
    scale = jnp.asarray(D**-0.5, dtype=q.dtype)
    q_bhtd = jnp.swapaxes(q * scale, 1, 2)  # (B, H, T, D)
    k_bhtd = jnp.swapaxes(k, 1, 2)          # (B, KV, T, D)
    v_bhtd = jnp.swapaxes(v, 1, 2)          # (B, KV, T, D)

    def local_fn(ql, kl, vl):
        return jax.vmap(kernel)(ql, kl, vl)

    if mesh is not None and mesh.size > 1 and B % mesh.size == 0:
        axis_name = mesh.axis_names[0]
        spec = P(axis_name, None, None, None)
        out_bhtd = _SHARD_MAP(
            local_fn,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            **_SHARD_MAP_KWARGS,
        )(q_bhtd, k_bhtd, v_bhtd)
    else:
        out_bhtd = local_fn(q_bhtd, k_bhtd, v_bhtd)

    return jnp.swapaxes(out_bhtd, 1, 2)  # (B, T, H, D)


def fused_causal_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    window_size: Optional[tuple[int, int]] = None,
    backend: str = "auto",
    splash_block_size: int = 512,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """
    Unified high-MFU causal (and sliding-window) attention for `(B, T, H, D)` tensors.
    Dispatches automatically to:
      - TPU:             Pallas SplashAttention (on-chip VMEM tiled)
      - Ampere+/Blackwell GPU: cuDNN FlashAttention
      - T4 / P100 / CPU: XLA dot_product_attention
    """
    impl = resolve_attn_backend(backend)
    if impl == "splash":
        return _splash_attention_tpu(
            q, k, v, window_size=window_size, block_size=splash_block_size, mesh=mesh
        )
    if impl == "cudnn":
        # cuDNN flash attention requires fp16/bf16; fall back to xla if fp32
        if q.dtype in (jnp.bfloat16, jnp.float16):
            return jax.nn.dot_product_attention(
                q, k, v, is_causal=True, local_window_size=window_size, implementation="cudnn"
            )
        return jax.nn.dot_product_attention(
            q, k, v, is_causal=True, local_window_size=window_size, implementation="xla"
        )
    return jax.nn.dot_product_attention(
        q, k, v, is_causal=True, local_window_size=window_size, implementation="xla"
    )
