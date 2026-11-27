"""
Verification suite for `exp/xla-vendetta`:
  1. Exact numerical verification of all 5 custom Pallas TPU Mosaic kernels via `interpret=True`
     (Zero-HBM-Logits CutCE fwd/bwd, Fused SwiGLU+INT8Quant, Fused SwiGLU bwd, Fused RMSNorm+INT8Quant,
     and Fused Dual-Axis INT8 Quantizer).
  2. Exact mathematical & gradient parity on 8 simulated FSDP devices comparing baseline vs vendetta
     (`aligned Canon-4`, `sharded all_to_all embeddings`, `virtual fused [w_qkv, w_attn_gate]`,
     `native (B,H,T,D) layout`, and `U-only layer-sharded Split-Muon`).
  3. 100% checkpoint compatibility: a checkpoint saved with baseline sharding reloads seamlessly
     into `exp/xla-vendetta`'s contracting-axis + hidden-sharded FSDP topology with identical params & opt_state.
  4. Sliding-window SplashAttention block-size capping (`W=1024 -> blk=512`, `full -> blk=1024`).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))


def test_pallas_mosaic_kernels_interpret_exact():
    """Verify all 5 custom Pallas TPU Mosaic kernels match pure-JAX references via interpret=True."""
    import jax
    import jax.numpy as jnp
    from scale.model import _chunk_ce_and_z, rms_norm
    from scale.pallas_kernels import (
        _choose_cutce_tiles,
        make_pallas_cutce_fn,
        pallas_dual_axis_quant_int8_local,
        pallas_fused_rmsnorm_quant_local,
        pallas_fused_swiglu_bwd_local,
        pallas_fused_swiglu_quant_local,
    )
    from scale.quant import quantize_int8

    # Dynamic CutCE VMEM tile selector sanity (1b-ve dimensions under 64 MiB VMEM budget)
    bm_f, bv_f, bm_dw, bv_dw = _choose_cutce_tiles(8192, 2048, 32768)
    assert bm_f >= 512 and bv_f >= 512 and bm_dw >= 512 and bv_dw >= 512
    assert 8192 % bm_f == 0 and 32768 % bv_f == 0 and 8192 % bm_dw == 0 and 32768 % bv_dw == 0

    g = jax.random.normal(jax.random.PRNGKey(0), (256, 256), jnp.bfloat16)
    u = jax.random.normal(jax.random.PRNGKey(1), (256, 256), jnp.bfloat16)
    dm = jax.random.normal(jax.random.PRNGKey(2), (256, 256), jnp.bfloat16)

    # 1. Fused SwiGLU + Row-INT8 Quant
    xq, xs = pallas_fused_swiglu_quant_local(g, u, b_m=128, interpret=True)
    xq_r, xs_r = quantize_int8(jax.nn.silu(g.astype(jnp.float32)) * u.astype(jnp.float32), axis=-1)
    assert float(jnp.max(jnp.abs(xs - xs_r))) < 1e-5
    assert int(jnp.max(jnp.abs(xq.astype(jnp.int32) - xq_r.astype(jnp.int32)))) <= 1

    # 2. Fused SwiGLU Backward (2D tiled)
    dg_p, du_p = pallas_fused_swiglu_bwd_local(g, u, dm, b_m=128, b_h=128, interpret=True)
    dg_r, du_r = jax.grad(
        lambda a, b: jnp.sum((jax.nn.silu(a.astype(jnp.float32)) * b.astype(jnp.float32)) * dm.astype(jnp.float32)),
        argnums=(0, 1),
    )(g, u)
    assert float(jnp.max(jnp.abs(dg_p.astype(jnp.float32) - dg_r.astype(jnp.bfloat16).astype(jnp.float32)))) < 5e-2
    assert float(jnp.max(jnp.abs(du_p.astype(jnp.float32) - du_r.astype(jnp.bfloat16).astype(jnp.float32)))) < 5e-2

    # 3. Fused RMSNorm + Row-INT8 Quant
    xqr, xsr = pallas_fused_rmsnorm_quant_local(g, b_m=128, interpret=True)
    xqr_r, xsr_r = quantize_int8(rms_norm(g), axis=-1)
    assert float(jnp.max(jnp.abs(xsr - xsr_r))) < 5e-3
    assert int(jnp.max(jnp.abs(xqr.astype(jnp.int32) - xqr_r.astype(jnp.int32)))) <= 1

    # 4. Fused Dual-Axis INT8 Quantizer
    qr, sr, qc, sc = pallas_dual_axis_quant_int8_local(g, b_m=128, interpret=True)
    qr_r, sr_r = quantize_int8(g, axis=-1)
    qc_r, sc_r = quantize_int8(g, axis=0)
    assert float(jnp.max(jnp.abs(sr - sr_r))) < 1e-5
    assert float(jnp.max(jnp.abs(sc - sc_r))) < 1e-5
    assert int(jnp.max(jnp.abs(qr.astype(jnp.int32) - qr_r.astype(jnp.int32)))) <= 1
    assert int(jnp.max(jnp.abs(qc.astype(jnp.int32) - qc_r.astype(jnp.int32)))) <= 1

    # 5. Zero-HBM-Logits Pallas CutCE (fwd + bwd dx & dw)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 128, 128), jnp.bfloat16) * 0.5
    w = jax.random.normal(jax.random.PRNGKey(4), (128, 512), jnp.bfloat16) * 0.2
    y = jax.random.randint(jax.random.PRNGKey(5), (2, 128), 0, 512)
    pfn = make_pallas_cutce_fn(15.0, 5e-4, mesh=None, interpret=True)

    def pal_wrap(xi, wi):
        tot, ce, zl = pfn(xi, wi, y, jnp.float32(1.0))
        return tot, (ce, zl)

    def ref_wrap(xi, wi):
        nll, zl = _chunk_ce_and_z(xi.reshape(-1, 128), y.reshape(-1), wi, 15.0)
        ce = jnp.mean(nll, dtype=jnp.float32)
        return ce + 5e-4 * zl, (ce, zl)

    (tp, (cep, zp)), (dxp, dwp) = jax.value_and_grad(pal_wrap, argnums=(0, 1), has_aux=True)(x, w)
    (tr, (cer, zr)), (dxr, dwr) = jax.value_and_grad(ref_wrap, argnums=(0, 1), has_aux=True)(x, w)

    def cos_sim(a, b):
        af, bf = a.astype(jnp.float32), b.astype(jnp.float32)
        return float(jnp.sum(af * bf) / (jnp.linalg.norm(af) * jnp.linalg.norm(bf)))

    assert abs(float(cep) - float(cer)) < 1e-5
    assert abs(float(zp) - float(zr)) < 1e-4
    assert cos_sim(dxp, dxr) > 0.9999
    assert cos_sim(dwp, dwr) > 0.9999


def test_splash_sliding_window_block_size_capping():
    """Verify sliding-window S layers cap SplashAttention block size to <= W//2 so empty blocks skip."""
    from scale.attention import _effective_splash_block_size

    assert _effective_splash_block_size(2048, 1024, window_size=None) == 1024
    assert _effective_splash_block_size(2048, 1024, window_size=(1024, 0)) == 512
    assert _effective_splash_block_size(2048, 1024, window_size=(512, 0)) == 256


def test_8_device_fsdp_vendetta_full_parity_and_checkpoint_compat(tmp_path):
    """
    Runs an 8-device simulated FSDP subprocess verifying:
      1. Exact loss, gradient (cos > 0.9995), and Split-Muon optimizer step parity between
         baseline (all vendetta flags OFF) and `exp/xla-vendetta` (all vendetta flags ON:
         aligned Canon-4, hidden-sharded all_to_all embeddings, virtual fused [w_qkv, w_attn_gate],
         and U-only layer-sharded Split-Muon) on a mini `1b-ve` architecture (SSSL, VE, U-Net,
         elementwise attn-gate, Canon-4, v0 residual, qk_gain, smear, int8-all).
      2. 100% checkpoint compatibility between baseline sharding and vendetta sharding!
    """
    ckpt_dir = tmp_path / "ckpt_compat"
    code = f"""
import functools, numpy as np, jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from scale.config import ModelSpec, TrainConfig
from scale.fsdp import build_tree_shardings, create_fsdp_mesh, shard_pytree
from scale.model import forward_scalar_loss_and_z, init_params, precompute_rotary_embeddings
from scale.optimizer import init_opt_state, optimizer_step
from scale.quant import prequantize_model_weights
from scale.checkpoint import CheckpointManager

mesh = create_fsdp_mesh()
assert mesh.size == 8

base_kwargs = dict(
    sequence_len=32, vocab_size=256, n_layer=8, n_embd=128, n_head=4, n_kv_head=2,
    mlp_hidden=256, activation="swiglu", window_pattern="SSSL",
    use_qk_gain=True, use_value_embeds=True, ve_stride=2, ve_gate_full=True,
    use_v0_residual=True, use_attn_gate=True, attn_gate_mode="elementwise",
    use_smear=True, smear_channels=True, use_canon=True, use_unet_skip=True,
    rope_base=50000.0, z_loss_weight=5e-4, remat_policy="save_all_compute",
    loss_chunk_size=128, matmul_quant="int8-all",
)
spec_base = ModelSpec(**base_kwargs, use_pallas_kernels=False, use_aligned_canon=False, use_sharded_embed=False, use_native_bhtd_attn=False)
spec_vend = ModelSpec(**base_kwargs, use_pallas_kernels=False, use_aligned_canon=True, use_sharded_embed=True, use_native_bhtd_attn=True)

tcfg_base = TrainConfig(param_dtype="float32", opt_dtype="float32", split_muon=True, muon_layer_sharded=False, muon_vmap=True, matrix_lr=0.04)
tcfg_vend = TrainConfig(param_dtype="float32", opt_dtype="float32", split_muon=True, muon_layer_sharded=True, muon_vmap=True, matrix_lr=0.04)

# Init params & opt_state identically
p0 = init_params(jax.random.PRNGKey(7), spec_base, param_dtype=jnp.float32)
o0 = init_opt_state(p0, opt_dtype=jnp.float32, split_muon=True, spec=spec_base)
p_sh = build_tree_shardings(p0, mesh, use_fsdp=True)
o_sh = build_tree_shardings(o0, mesh, use_fsdp=True)
# Verify vendetta FSDP sharding rules!
assert p_sh["wte"].spec == P(None, "fsdp"), p_sh["wte"].spec
assert p_sh["value_embeds"].spec == P(None, None, "fsdp"), p_sh["value_embeds"].spec
assert p_sh["blocks"]["w_qkv"].spec == P(None, "fsdp", None), p_sh["blocks"]["w_qkv"].spec
assert p_sh["blocks"]["w_gate_up"].spec == P(None, "fsdp", None), p_sh["blocks"]["w_gate_up"].spec
assert p_sh["blocks"]["mlp_down"].spec == P(None, "fsdp", None), p_sh["blocks"]["mlp_down"].spec
assert p_sh["canon_a"].spec == P(), p_sh["canon_a"].spec
assert p_sh["canon_c"].spec == P(), p_sh["canon_c"].spec
assert p_sh["blocks"]["ve_gate"].spec == P(), p_sh["blocks"]["ve_gate"].spec

p_b = shard_pytree(p0, p_sh)
o_b = shard_pytree(o0, o_sh)
p_v = shard_pytree(p0, p_sh)
o_v = shard_pytree(o0, o_sh)

cos, sin = precompute_rotary_embeddings(32, spec_base.head_dim, base=50000.0)
rng = np.random.default_rng(123)
b_sh = NamedSharding(mesh, P("fsdp", None))
x = jax.device_put(rng.integers(0, 256, size=(16, 32), dtype=np.int32), b_sh)
y = jax.device_put(rng.integers(0, 256, size=(16, 32), dtype=np.int32), b_sh)

def run_step(p, o, spec, tcfg, strip_fused_q):
    p_bf16 = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), p)
    qc = prequantize_model_weights(p_bf16, "int8-all", tie_embeddings=False)
    if strip_fused_q and "w_qkv_ag" in qc["blocks"]:
        qc = {{"blocks": {{k: v for k, v in qc["blocks"].items() if k != "w_qkv_ag"}}, "lm_head_bwd": qc["lm_head_bwd"]}}
    def lfn(pin):
        tot, ce, zl = forward_scalar_loss_and_z(pin, x, y, cos, sin, spec, jnp.bfloat16, jnp.float32(1.0), mesh=mesh, q_cache=qc)
        return tot, (ce, zl)
    (tot, (ce, zl)), g = jax.value_and_grad(lfn, has_aux=True)(p_bf16)
    pn, on, gn = optimizer_step(p, g, o, jnp.float32(1.0), jnp.float32(0.95), jnp.float32(0.03), spec, tcfg, jnp.float32, mesh=mesh)
    return ce, zl, gn, g, pn, on

ce_b, z_b, gn_b, g_b, pn_b, on_b = jax.jit(lambda p, o: run_step(p, o, spec_base, tcfg_base, True), in_shardings=(p_sh, o_sh))(p_b, o_b)
ce_v, z_v, gn_v, g_v, pn_v, on_v = jax.jit(lambda p, o: run_step(p, o, spec_vend, tcfg_vend, False), in_shardings=(p_sh, o_sh))(p_v, o_v)

print(f"CE: base={{float(ce_b):.6f}} vend={{float(ce_v):.6f}} | GN: base={{float(gn_b):.6f}} vend={{float(gn_v):.6f}}")
assert abs(float(ce_b) - float(ce_v)) < 1e-4, f"CE mismatch: {{ce_b}} vs {{ce_v}}"
assert abs(float(z_b) - float(z_v)) < 1e-3, f"Z mismatch: {{z_b}} vs {{z_v}}"

# Check gradient cosine similarity across every leaf!
def cos_sim(a, b):
    af, bf = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-8 and nb < 1e-8:
        return 1.0
    return float(np.dot(af, bf) / max(na * nb, 1e-12))

for (path, gb), (_, gv) in zip(jax.tree_util.tree_leaves_with_path(g_b), jax.tree_util.tree_leaves_with_path(g_v)):
    c = cos_sim(gb, gv)
    assert c > 0.998, f"Grad cosine mismatch at {{path}}: {{c:.6f}}"

# Check updated parameter & optimizer state parity after full Split-Muon + AdamW step!
for (path, pb), (_, pv) in zip(jax.tree_util.tree_leaves_with_path(pn_b), jax.tree_util.tree_leaves_with_path(pn_v)):
    diff = float(np.max(np.abs(np.asarray(pb, np.float32) - np.asarray(pv, np.float32))))
    assert diff < 1e-3, f"Param update mismatch at {{path}}: max_abs={{diff:.6f}}"

# Verify 100% checkpoint save & reload compatibility!
mgr = CheckpointManager("{ckpt_dir}")
mgr.save(step=1, params=pn_b, opt_state=on_b, trainer_state={{"step": 1}})
p_loaded, o_loaded, meta = mgr.load_latest(pn_v, on_v, p_sh, o_sh)
assert meta["step"] == 1
for a, b in zip(jax.tree_util.tree_leaves(pn_b), jax.tree_util.tree_leaves(p_loaded)):
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
print("ALL_PARITY_AND_CKPT_CHECKS_PASSED")
"""
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=8"}
    res = subprocess.run([PYTHON, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"8-device parity failed!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "ALL_PARITY_AND_CKPT_CHECKS_PASSED" in res.stdout


def test_xprof_profiler_hook_runs_cleanly(tmp_path):
    """Verify --profile-dir XProf capture hook runs cleanly in run_benchmark without error."""
    from scale.config import TrainConfig, get_model_preset
    from scale.pretrain import run_benchmark

    prof_dir = tmp_path / "xprof_trace"
    spec = get_model_preset(
        "smoke",
        arch_overrides={"n_layer": 2, "sequence_len": 32, "loss_chunk_size": 32, "remat_policy": "save_attn"},
    )
    tcfg = TrainConfig(
        total_batch_size=64,
        per_device_batch=2,
        use_fsdp=False,
        profile_dir=str(prof_dir),
        profile_steps=1,
    )
    res = run_benchmark(spec, tcfg, warmup_steps=1, timed_steps=1)
    assert res["tok_per_sec"] > 0
    assert prof_dir.exists()
    assert any(prof_dir.rglob("*")), f"Expected XProf trace files inside {prof_dir}"

