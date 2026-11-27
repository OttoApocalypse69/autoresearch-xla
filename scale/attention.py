"""
High-MFU fused attention kernels for TPU (Pallas SplashAttention), Ampere+/Blackwell
GPU (cuDNN FlashAttention), and universal XLA fallback (T4 / P100 / CPU).

Critical numerical & performance invariants verified:
- Pallas SplashAttention does NOT automatically divide Q by sqrt(head_dim), so we
  explicitly pre-scale `q = q * (head_dim ** -0.5)` before invoking the kernel.
- `fused_causal_attention_bhtd` operates directly on native `(B, H, T, D)` tensors so
  TPU SplashAttention executes with ZERO `(B, T, H, D) <-> (B, H, T, D)` HBM transpose
  copies per layer!
- Sliding-window (`window_size=(W, 0)`) layers automatically cap SplashAttention block
  size to `<= W // 2` (e.g. `512` for `W=1024`) so empty off-window blocks are actually
  skipped in hardware (FlyWheel / SplashAttention block-skipping invariant).
- Wraps in `shard_map` when `mesh.size > 1` so multi-chip TPU FSDP executes local
  SplashAttention on every chip with zero cross-chip attention communication.
"""

from __future__ import annotations

import functools
import inspect
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from scale.pallas_kernels import is_tpu_mesh, pallas_tpu_available

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


@functools.lru_cache(maxsize=1)
def _probe_pallas_tpu_supported() -> bool:
    """
    Verify Pallas TPU Mosaic lowering works against the installed libtpu (0-HBM `.lower()` probe).
    Catches Kaggle TPU v5e-8 images that ship newer JAX (e.g. 0.11) with an older bundled libtpu.
    """
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas.ops.tpu import splash_attention  # noqa: F401

        def _noop_kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...]

        spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
        fn = pl.pallas_call(_noop_kernel, out_shape=spec)
        jax.jit(fn).lower(spec)
        return True
    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(
            f"[Attention] WARNING: Pallas TPU SplashAttention unavailable ({first_line}). "
            "Falling back to XLA attention. On Kaggle TPU v5e-8, match JAX to Kaggle's bundled libtpu "
            "(e.g. `pip install -q jax==0.6.2 jaxlib==0.6.2`) or upgrade `libtpu` to unlock SplashAttention!",
            flush=True,
        )
        return False


def resolve_attn_backend(requested: str = "auto", mesh: Optional[Mesh] = None) -> str:
    """Choose fastest numerically exact attention backend for the active (or target AOT) JAX platform."""
    if requested != "auto":
        return requested
    if is_tpu_mesh(mesh):
        return "splash" if pallas_tpu_available(mesh) else "xla"
    dev0 = jax.devices()[0]
    platform = str(dev0.platform).lower()
    if platform == "tpu":
        return "splash" if _probe_pallas_tpu_supported() else "xla"
    if platform == "gpu":
        kind = str(getattr(dev0, "device_kind", ""))
        return "cudnn" if _is_ampere_or_newer_gpu(kind) else "xla"
    return "xla"


def _effective_splash_block_size(T: int, requested_block_size: int, window_size: Optional[tuple[int, int]]) -> int:
    """
    Pick largest valid multiple-of-128 block size dividing T.
    For sliding-window layers (`W = window_size[0]`), cap block size to `<= W // 2` (e.g., `512`
    for `W=1024`) so SplashAttention's block-sparse mask actually skips empty off-window tiles!
    """
    cap = requested_block_size
    if window_size is not None and window_size[0] >= 256:
        cap = min(cap, max(128, window_size[0] // 2))
    blk = min(cap, T)
    blk = max(128, (blk // 128) * 128)
    while T % blk != 0 and blk > 128:
        blk -= 128
    return blk


def _splash_attention_tpu_bhtd(
    q_bhtd: jax.Array,
    k_bhtd: jax.Array,
    v_bhtd: jax.Array,
    window_size: Optional[tuple[int, int]],
    block_size: int = 1024,
    mesh: Optional[Mesh] = None,
    interpret: bool = False,
) -> jax.Array:
    """
    Native `(B, H, T, D)` TPU Pallas SplashAttention (ZERO `(B,T,H,D) <-> (B,H,T,D)` HBM transposes!).
    """
    from jax.experimental.pallas.ops.tpu import splash_attention

    B, H, T, D = q_bhtd.shape
    if T < 128 or T % 128 != 0 or D % 128 != 0:
        out_bthd = jax.nn.dot_product_attention(
            jnp.swapaxes(q_bhtd, 1, 2),
            jnp.swapaxes(k_bhtd, 1, 2),
            jnp.swapaxes(v_bhtd, 1, 2),
            is_causal=True,
            local_window_size=window_size,
        )
        return jnp.swapaxes(out_bthd, 1, 2)

    blk = _effective_splash_block_size(T, block_size, window_size)

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
    scale = jnp.asarray(D**-0.5, dtype=q_bhtd.dtype)

    def local_fn(ql, kl, vl):
        return jax.vmap(kernel)(ql * scale, kl, vl)

    if mesh is not None and mesh.size > 1 and B % mesh.size == 0:
        axis_name = mesh.axis_names[0]
        spec = P(axis_name, None, None, None)
        return _SHARD_MAP(
            local_fn,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            **_SHARD_MAP_KWARGS,
        )(q_bhtd, k_bhtd, v_bhtd)
    return local_fn(q_bhtd, k_bhtd, v_bhtd)


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
    out_bhtd = _splash_attention_tpu_bhtd(
        jnp.swapaxes(q, 1, 2),
        jnp.swapaxes(k, 1, 2),
        jnp.swapaxes(v, 1, 2),
        window_size=window_size,
        block_size=block_size,
        mesh=mesh,
        interpret=interpret,
    )
    return jnp.swapaxes(out_bhtd, 1, 2)


def fused_causal_attention_bhtd(
    q_bhtd: jax.Array,
    k_bhtd: jax.Array,
    v_bhtd: jax.Array,
    window_size: Optional[tuple[int, int]] = None,
    backend: str = "auto",
    splash_block_size: int = 1024,
    mesh: Optional[Mesh] = None,
) -> jax.Array:
    """Native `(B, H, T, D)` causal/sliding-window attention (zero transpose overhead on TPU Splash!)."""
    impl = resolve_attn_backend(backend, mesh=mesh)
    if impl == "splash":
        return _splash_attention_tpu_bhtd(
            q_bhtd, k_bhtd, v_bhtd, window_size=window_size, block_size=splash_block_size, mesh=mesh
        )
    out_bthd = fused_causal_attention(
        jnp.swapaxes(q_bhtd, 1, 2),
        jnp.swapaxes(k_bhtd, 1, 2),
        jnp.swapaxes(v_bhtd, 1, 2),
        window_size=window_size,
        backend=impl,
        splash_block_size=splash_block_size,
        mesh=mesh,
    )
    return jnp.swapaxes(out_bthd, 1, 2)


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
    impl = resolve_attn_backend(backend, mesh=mesh)
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
