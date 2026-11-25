"""
Pure-JAX 1D FSDP (ZeRO-3 style) sharding utilities for multi-chip TPUs & GPUs.

On Kaggle TPU v5e-8 (8 chips x 16 GB = 128 GB HBM) or Kaggle 2xT4 (32 GB VRAM):
- Every batch tensor (x, y) is data-sharded along the batch axis: P("fsdp", None)
- Every large parameter & optimizer-state array (2D embeddings/heads and 3D scanned
  layer matrices `(n_layer, in_dim, out_dim)`) is sharded along its largest non-layer
  axis divisible by `num_devices`, e.g. P(None, "fsdp", None) or P("fsdp", None).
- Small 1D scalars/vectors stay replicated: P().

XLA's GSPMD compiler automatically inserts layer-wise all-gather before matmuls and
reduce-scatter on gradients, cutting per-chip static weight+optimizer HBM for a 1B
model from ~9.2 GB down to ~1.15 GB on 8-chip TPU v5e-8.
"""

from __future__ import annotations

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_fsdp_mesh() -> Mesh:
    """Create a 1D mesh named 'fsdp' spanning all visible JAX devices."""
    devices = np.array(jax.devices())
    return Mesh(devices, ("fsdp",))


def _choose_shard_spec(arr, num_devices: int, use_fsdp: bool, min_elements: int = 4096) -> P:
    """
    Pick an optimal PartitionSpec for a parameter or optimizer-state leaf.

    Rules:
    - If single device, FSDP disabled, scalar, or tiny array -> replicated P().
    - For 3D scanned layer tensors `(n_layer, d1, d2)`: never shard axis 0 (`n_layer`),
      so arbitrary depths (e.g. 20 layers on 8 chips) work cleanly; shard the largest
      of `(d1, d2)` divisible by `num_devices`.
    - For 2D tensors `(d0, d1)`: shard the largest axis divisible by `num_devices`.
    - For 1D tensors `(d0,)`: shard if large and divisible by `num_devices`, else replicate.
    """
    if not use_fsdp or num_devices <= 1:
        return P()

    shape = getattr(arr, "shape", ())
    if not shape:
        return P()

    numel = int(np.prod(shape))
    if numel < min_elements:
        return P()

    ndim = len(shape)
    if ndim >= 3:
        # Keep leading scan axis (axis 0) unsharded; pick largest trailing dim divisible by num_devices
        candidates = [(shape[ax], ax) for ax in range(1, ndim) if shape[ax] % num_devices == 0]
        if not candidates:
            return P()
        _, best_ax = max(candidates)
        spec = [None] * ndim
        spec[best_ax] = "fsdp"
        return P(*spec)

    if ndim == 2:
        candidates = [(shape[ax], ax) for ax in range(2) if shape[ax] % num_devices == 0]
        if not candidates:
            return P()
        _, best_ax = max(candidates)
        spec = [None, None]
        spec[best_ax] = "fsdp"
        return P(*spec)

    if ndim == 1 and shape[0] % num_devices == 0:
        return P("fsdp")

    return P()


def build_tree_shardings(tree, mesh: Mesh, use_fsdp: bool = True):
    """Build a matching pytree of `NamedSharding` objects for params or opt_state."""
    num_devices = mesh.size

    def leaf_sharding(leaf):
        pspec = _choose_shard_spec(leaf, num_devices, use_fsdp)
        return NamedSharding(mesh, pspec)

    return jax.tree_util.tree_map(leaf_sharding, tree)


def shard_pytree(tree, shardings):
    """Place every leaf of `tree` onto its target `NamedSharding`."""
    return jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), tree, shardings)


def summarize_sharding(tree, mesh: Mesh) -> dict:
    """Return memory & sharding stats (total bytes, sharded bytes, per-device bytes)."""
    total_bytes = 0
    sharded_bytes = 0
    replicated_bytes = 0
    n_sharded_leaves = 0
    n_replicated_leaves = 0

    for leaf in jax.tree_util.tree_leaves(tree):
        nbytes = int(leaf.size) * int(leaf.dtype.itemsize)
        total_bytes += nbytes
        pspec = getattr(leaf.sharding, "spec", P())
        is_sharded = any(ax is not None for ax in pspec)
        if is_sharded:
            sharded_bytes += nbytes
            n_sharded_leaves += 1
        else:
            replicated_bytes += nbytes
            n_replicated_leaves += 1

    per_device_bytes = (sharded_bytes / max(mesh.size, 1)) + replicated_bytes
    return {
        "num_devices": mesh.size,
        "total_mb": round(total_bytes / (1024**2), 2),
        "sharded_mb": round(sharded_bytes / (1024**2), 2),
        "replicated_mb": round(replicated_bytes / (1024**2), 2),
        "per_device_mb": round(per_device_bytes / (1024**2), 2),
        "sharded_leaves": n_sharded_leaves,
        "replicated_leaves": n_replicated_leaves,
    }
