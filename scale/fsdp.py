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


_BLOCK_LINEAR_KEYS = ("w_qkv", "w_attn_gate", "c_proj", "w_gate_up", "mlp_down", "mlp_fc", "mtp_proj")
# Tiny per-channel conv/gate/smear vectors (< 0.7 MB total across all 20 layers):
# keep replicated P() so GSPMD never all-to-all's 67 MB activation tensors to match
# a 12 KB depthwise Canon-4 conv kernel or all-gathers 8 KB gate vectors per layer.
_REPLICATED_SMALL_KEYS = (
    "canon_a",
    "canon_c",
    "mtp_canon",
    "ve_gate",
    "smear_gate",
    "smear_lambda",
    "smear_lambda2",
    "qk_gain",
    "v0_lambdas",
    "unet_lambdas",
    "resid_lambdas",
    "x0_lambdas",
)


def choose_linear_scatter_dim(k_dim: int, n_dim: int, num_devices: int) -> int | None:
    """
    Pick optimal FSDP sharding / `psum_scatter` axis for a 2D linear weight `(K, N)`.

    Empirically verified on TPU v5e:2x4 (`libtpu`): sharding the contracting input axis 0 (`K`)
    enables XLA windowed einsum (`collective-permute` overlapped into MXU with ZERO full-weight
    all-gather buffer), beating axis-1 sharding by 29% on `w_gate_up` and 5.3x on `mlp_down`.
    """
    if num_devices <= 1:
        return None
    if k_dim % num_devices == 0:
        return 0
    if n_dim % num_devices == 0:
        return 1
    return None


def _choose_shard_spec(
    arr,
    num_devices: int,
    use_fsdp: bool,
    min_elements: int = 4096,
    name: str = "",
) -> P:
    """
    Pick an optimal PartitionSpec for a parameter or optimizer-state leaf.

    Rules (100% checkpoint-shape compatible with baseline FSDP):
    - If single device, FSDP disabled, scalar, tiny array, or tiny per-channel conv/gate -> P().
    - `wte` `(V, C)` & `value_embeds` `(n_ve, V, kv_dim)`: shard along trailing hidden dim
      (`P(None, "fsdp")` / `P(None, None, "fsdp")`) for 6x lower embedding ICI via `all_to_all`
      and zero full-table `all-reduce` in backward!
    - 3D scanned block linear weights `(L, K, N)`: shard along contracting axis 1 (`K`) when
      divisible by `num_devices` (`P(None, "fsdp", None)`) so TPU windowed einsum streams weight
      shards directly into MXUs with zero full-weight `all-gather` buffers.
    - Other 2D/3D tensors (e.g. `lm_head` `(C, V)`): shard the largest non-layer axis divisible
      by `num_devices`.
    """
    if not use_fsdp or num_devices <= 1:
        return P()

    if name in _REPLICATED_SMALL_KEYS:
        return P()

    shape = getattr(arr, "shape", ())
    if not shape:
        return P()

    numel = int(np.prod(shape))
    if numel < min_elements:
        return P()

    ndim = len(shape)
    # Hidden-dim sharded embeddings (enables shard_map all_to_all lookup & local scatter_add bwd!)
    if name == "wte" and ndim == 2 and shape[1] % num_devices == 0:
        return P(None, "fsdp")
    if name == "value_embeds" and ndim == 3 and shape[2] % num_devices == 0:
        return P(None, None, "fsdp")

    # 3D scanned block linear weights: prefer contracting axis 1 (K) for windowed einsum!
    if ndim == 3 and name in _BLOCK_LINEAR_KEYS and shape[1] % num_devices == 0:
        return P(None, "fsdp", None)

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
    """Build a matching pytree of `NamedSharding` objects for params or opt_state (name-aware)."""
    num_devices = mesh.size

    def _rec(node, parent_name: str = ""):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                # Propagate parent parameter name into optimizer-state buffers ("m", "v", "v_*")
                child_name = parent_name if (k in ("m", "v") or k.startswith("v_")) and parent_name else k
                out[k] = _rec(v, child_name)
            return out
        pspec = _choose_shard_spec(node, num_devices, use_fsdp, name=parent_name)
        return NamedSharding(mesh, pspec)

    return _rec(tree)


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
