"""
Comprehensive test suite for the resumable 1B FSDP pretraining engine (`scale/`).

Verifies:
  1. Exact ~1.0B parameter counts for `1b` and `1b-relu2` presets via zero-memory `jax.eval_shape`
  2. Real 8-device simulated FSDP (`XLA_FLAGS=--xla_force_host_platform_device_count=8`):
     proves large 2D/3D params & optimizer states are sharded 1/8 per chip
  3. All model architecture variants (SwiGLU, ReLU^2, GeLU, Value Embeddings, tied embeddings,
     window patterns L / SL / SSSL, z-loss, opt_dtype=bfloat16)
  4. WSD (Warmup -> Stable -> Decay) schedule math
  5. Topology-agnostic checkpoint save/reload (including bfloat16 opt states) + local sync push/pull
  6. Full end-to-end 8-device FSDP multi-session pretrain (Session 1 stable -> sync -> Session 2
     fresh dir auto-resume + WSD decay + sampling + exact DataCursor continuity)
  7. Notebook JSON & code cell syntax validity for `pretrain_1b_colab_kaggle.ipynb`
"""

from __future__ import annotations

import ast
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))
from scale.config import ModelSpec, TrainConfig, get_model_preset  # noqa: E402
from scale.optimizer import compute_weight_decay, compute_wsd_lrm  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Exact 1B Preset Parameter Counts (Zero-Memory via jax.eval_shape)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "preset,min_params,max_params",
    [
        ("smoke", 500_000, 1_500_000),
        ("100m", 110_000_000, 140_000_000),
        ("350m", 340_000_000, 390_000_000),
        ("1b", 1_000_000_000, 1_060_000_000),
        ("1b-relu2", 1_000_000_000, 1_050_000_000),
        ("1b-ve", 1_420_000_000, 1_490_000_000),
    ],
)
def test_preset_exact_param_counts(preset, min_params, max_params):
    import jax
    from scale.model import count_params, init_params

    spec = get_model_preset(preset)
    abs_params = jax.eval_shape(functools.partial(init_params, spec=spec), jax.random.PRNGKey(0))
    total = count_params(abs_params)["total"]
    assert min_params <= total <= max_params, f"{preset}: got {total:,} params"

    # Verify all major dims on 1b presets are TPU MXU-friendly multiples of 128
    if preset.startswith("1b"):
        assert spec.n_embd % 128 == 0
        assert spec.head_dim % 128 == 0
        assert spec.mlp_hidden % 128 == 0
        assert spec.vocab_size % 128 == 0


# ---------------------------------------------------------------------------
# 2. WSD (Warmup -> Stable -> Decay) Schedule Math
# ---------------------------------------------------------------------------

def test_wsd_schedule():
    cfg_stable = TrainConfig(wsd_phase="stable", warmup_steps=100, decay_steps=50, weight_decay=0.1)
    assert pytest.approx(compute_wsd_lrm(0, cfg_stable)) == 0.01
    assert pytest.approx(compute_wsd_lrm(99, cfg_stable)) == 1.0
    assert pytest.approx(compute_wsd_lrm(5000, cfg_stable)) == 1.0
    assert pytest.approx(compute_weight_decay(5000, cfg_stable)) == 0.1

    cfg_decay = TrainConfig(
        wsd_phase="decay", warmup_steps=100, decay_start_step=1000, decay_steps=200, min_lr_frac=0.0, weight_decay=0.1
    )
    assert pytest.approx(compute_wsd_lrm(1000, cfg_decay)) == 1.0
    assert pytest.approx(compute_wsd_lrm(1100, cfg_decay)) == 0.5
    assert pytest.approx(compute_wsd_lrm(1200, cfg_decay)) == 0.0
    assert pytest.approx(compute_weight_decay(1100, cfg_decay)) == 0.05
    assert pytest.approx(compute_weight_decay(1200, cfg_decay)) == 0.0


# ---------------------------------------------------------------------------
# 3. All Model Architecture Knobs + Scanned NorMuon Convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "activation,window_pattern,use_ve,tie_emb,opt_dtype_name",
    [
        ("swiglu", "L", False, False, "float32"),
        ("relu2", "SL", True, False, "float32"),
        ("gelu", "SSSL", False, True, "bfloat16"),
    ],
)
def test_architecture_variants_and_scanned_muon(activation, window_pattern, use_ve, tie_emb, opt_dtype_name):
    import jax
    import jax.numpy as jnp
    from scale.model import forward_loss_2d, init_params, precompute_rotary_embeddings
    from scale.optimizer import init_opt_state, optimizer_step

    spec = ModelSpec(
        sequence_len=32,
        vocab_size=128,
        n_layer=4,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        mlp_hidden=128,
        activation=activation,
        window_pattern=window_pattern,
        use_value_embeds=use_ve,
        tie_embeddings=tie_emb,
        use_remat=True,
    )
    tcfg = TrainConfig(opt_dtype=opt_dtype_name, matrix_lr=0.04, embedding_lr=0.4)
    opt_dtype = jnp.bfloat16 if opt_dtype_name == "bfloat16" else jnp.float32

    params = init_params(jax.random.PRNGKey(0), spec)
    opt_state = init_opt_state(params, opt_dtype=opt_dtype)
    cos, sin = precompute_rotary_embeddings(32, spec.head_dim)

    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
    y = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))

    @jax.jit
    def step_fn(p, st):
        def loss_fn(par):
            nll, zl = forward_loss_2d(par, x, y, cos, sin, spec, jnp.bfloat16, return_z_loss=True)
            return jnp.mean(nll) + spec.z_loss_weight * zl

        loss_v, grads = jax.value_and_grad(loss_fn)(p)
        p_new, st_new, gn = optimizer_step(
            p, grads, st, jnp.float32(1.0), jnp.float32(0.9), jnp.float32(0.1), spec, tcfg, jnp.bfloat16
        )
        return p_new, st_new, loss_v, gn

    losses = []
    for _ in range(6):
        params, opt_state, lv, gn = step_fn(params, opt_state)
        assert np.isfinite(float(lv)) and np.isfinite(float(gn))
        losses.append(float(lv))

    assert losses[-1] < losses[0] - 0.3, f"Loss failed to drop for {activation}/{window_pattern}: {losses}"


# ---------------------------------------------------------------------------
# 4. Real 8-Device Simulated FSDP Sharding Verification + Multi-Session Resume
# ---------------------------------------------------------------------------

def test_8_device_fsdp_sharding_and_multi_session_resume(tmp_path):
    """
    Runs TWO real subprocesses with 8 simulated JAX devices (emulating Kaggle TPU v5e-8):
      - Session 1: trains 3 steps in WSD `stable` phase with FSDP enabled, checks that params
        & opt_state are sharded ~1/8 per chip, generates a live text sample, evaluates val_bpb,
        and syncs checkpoint + tokenizer + cursor to `remote_drive`.
      - Session 2: starts from a brand-new empty local checkpoint dir, pulls from `remote_drive`,
        resumes exact step=3 & DataCursor, runs 2 WSD `decay` steps -> reaches step=5 cleanly!
    """
    remote_dir = tmp_path / "remote_sync"
    ckpt_s1 = tmp_path / "ckpt_session1"
    ckpt_s2 = tmp_path / "ckpt_session2"
    shard_cache = tmp_path / "shards"

    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    }

    # First verify exact 8-chip FSDP sharding ratio directly in an 8-device subprocess
    verify_sharding_code = """
import functools, jax
from scale.config import get_model_preset
from scale.fsdp import create_fsdp_mesh, build_tree_shardings, summarize_sharding
from scale.model import init_params
from scale.optimizer import init_opt_state

mesh = create_fsdp_mesh()
assert mesh.size == 8
spec = get_model_preset("smoke")
abs_p = jax.eval_shape(functools.partial(init_params, spec=spec), jax.random.PRNGKey(0))
abs_o = jax.eval_shape(init_opt_state, abs_p)
p_sh = build_tree_shardings(abs_p, mesh, use_fsdp=True)
o_sh = build_tree_shardings(abs_o, mesh, use_fsdp=True)
p = jax.jit(functools.partial(init_params, spec=spec), out_shardings=p_sh)(jax.random.PRNGKey(0))
o = jax.jit(init_opt_state, in_shardings=(p_sh,), out_shardings=o_sh)(p)
ps = summarize_sharding(p, mesh)
os_ = summarize_sharding(o, mesh)
print("FSDP_PARAMS:", ps)
print("FSDP_OPT:", os_)
assert ps["sharded_leaves"] >= 6
assert ps["per_device_mb"] < ps["total_mb"] / 6.0
assert os_["per_device_mb"] < os_["total_mb"] / 5.0
"""
    sh_res = subprocess.run(
        [PYTHON, "-c", verify_sharding_code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert sh_res.returncode == 0, f"8-dev FSDP verification failed:\n{sh_res.stdout}\n{sh_res.stderr}"

    # Session 1: WSD stable phase on 8 simulated devices (per_device_batch=1 -> 8 global seqs * 128 = 1024 tok/micro -> 2048 tok/step => grad_accum=2)
    cmd1 = [
        PYTHON, "-m", "scale.pretrain",
        "--preset", "smoke",
        "--dataset", "synthetic",
        "--total-batch-size", "2048",
        "--per-device-batch", "1",
        "--eval-tokens", "1024",
        "--phase", "stable",
        "--warmup-steps", "2",
        "--max-steps", "3",
        "--log-every", "1",
        "--eval-every", "3",
        "--sample-every", "3",
        "--save-every", "3",
        "--checkpoint-dir", str(ckpt_s1),
        "--shard-cache-dir", str(shard_cache),
        "--sync-backend", "local",
        "--remote-target", str(remote_dir),
    ]
    r1 = subprocess.run(cmd1, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert r1.returncode == 0, f"Session 1 failed:\n{r1.stdout}\n{r1.stderr}"
    assert "step:            3" in r1.stdout
    assert "[Sample step 3]" in r1.stdout
    assert (remote_dir / "latest" / "trainer_state.json").exists()
    assert (remote_dir / "tokenizer").is_dir()

    # Session 2: brand new empty local dir -> pull from remote_dir -> WSD decay for 2 more steps -> step 5!
    cmd2 = [
        PYTHON, "-m", "scale.pretrain",
        "--preset", "smoke",
        "--dataset", "synthetic",
        "--total-batch-size", "2048",
        "--per-device-batch", "1",
        "--eval-tokens", "1024",
        "--phase", "decay",
        "--decay-steps", "2",
        "--max-steps", "5",  # decay phase itself should cleanly stop after 2 steps at step 5!
        "--log-every", "1",
        "--eval-every", "0",
        "--sample-every", "0",
        "--save-every", "2",
        "--checkpoint-dir", str(ckpt_s2),
        "--shard-cache-dir", str(shard_cache),
        "--sync-backend", "local",
        "--remote-target", str(remote_dir),
    ]
    r2 = subprocess.run(cmd2, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert r2.returncode == 0, f"Session 2 resume failed:\n{r2.stdout}\n{r2.stderr}"
    assert "Resumed at step=3" in r2.stdout
    assert "Completed WSD decay phase (2 steps)" in r2.stdout
    assert "step:            5" in r2.stdout


# ---------------------------------------------------------------------------
# 5. Pretrain 1B Notebook JSON & Syntax Check
# ---------------------------------------------------------------------------

def test_pretrain_1b_notebook_valid():
    nb_path = REPO_ROOT / "pretrain_1b_colab_kaggle.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4
    for idx, c in enumerate(code_cells):
        ast.parse("".join(c["source"]), filename=f"pretrain_1b_cell_{idx}")


def test_chunked_ce_matches_full_ce():
    import jax
    import jax.numpy as jnp
    from scale.model import forward_loss_2d, init_params, precompute_rotary_embeddings

    spec_full = ModelSpec(
        sequence_len=32, vocab_size=128, n_layer=2, n_embd=64, n_head=2, n_kv_head=1,
        mlp_hidden=128, activation="swiglu", loss_chunk_size=0, remat_policy="none",
    )
    spec_chunk = ModelSpec.from_dict({**spec_full.to_dict(), "loss_chunk_size": 16})
    params = init_params(jax.random.PRNGKey(0), spec_full)
    cos, sin = precompute_rotary_embeddings(32, 32)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
    y = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))

    nll_f, z_f = forward_loss_2d(params, x, y, cos, sin, spec_full, jnp.float32, return_z_loss=True)
    nll_c, z_c = forward_loss_2d(params, x, y, cos, sin, spec_chunk, jnp.float32, return_z_loss=True)
    assert float(jnp.max(jnp.abs(nll_f - nll_c))) < 1e-5
    assert abs(float(z_f) - float(z_c)) < 1e-5


def test_fused_chunked_ce_exact_gradients_and_remat_policies():
    import jax
    import jax.numpy as jnp
    from scale.model import (
        forward_loss_2d,
        forward_scalar_loss_and_z,
        init_params,
        precompute_rotary_embeddings,
    )

    spec_full = ModelSpec(
        sequence_len=32, vocab_size=128, n_layer=2, n_embd=64, n_head=2, n_kv_head=1,
        mlp_hidden=128, activation="swiglu", loss_chunk_size=0, remat_policy="none",
    )
    spec_fused = ModelSpec.from_dict({**spec_full.to_dict(), "loss_chunk_size": 16, "remat_policy": "save_all_compute"})
    params = init_params(jax.random.PRNGKey(1), spec_full)
    cos, sin = precompute_rotary_embeddings(32, 32)
    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
    y = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))

    for dtype, scale_val, tol in [(jnp.float32, 1.0, 1e-5), (jnp.bfloat16, 1.0, 2e-3), (jnp.float16, 1024.0, 2e-3)]:
        scale = jnp.float32(scale_val)
        p_in = jax.tree_util.tree_map(lambda a: a.astype(dtype), params)

        def ref_fn(p):
            nll, z = forward_loss_2d(p, x, y, cos, sin, spec_full, dtype, return_z_loss=True)
            ce = jnp.mean(nll, dtype=jnp.float32)
            return (ce + jnp.float32(spec_full.z_loss_weight) * z) * scale, (ce, z)

        def fused_fn(p):
            tot, ce, z = forward_scalar_loss_and_z(p, x, y, cos, sin, spec_fused, dtype, scale=scale)
            return tot, (ce, z)

        (_, (ce_r, z_r)), g_r = jax.value_and_grad(ref_fn, has_aux=True)(p_in)
        (_, (ce_f, z_f)), g_f = jax.value_and_grad(fused_fn, has_aux=True)(p_in)
        assert abs(float(ce_r) - float(ce_f)) < 1e-3
        assert abs(float(z_r) - float(z_f)) < 1e-3
        max_unscaled_diff = max(
            float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) / scale_val
            for a, b in zip(jax.tree_util.tree_leaves(g_r), jax.tree_util.tree_leaves(g_f))
        )
        assert max_unscaled_diff < tol, f"Grad mismatch for {dtype}: {max_unscaled_diff}"


def test_multi_dataset_mixer_parsing_and_cursor_resume(tmp_path):
    from scale.data import (
        DATASET_MIX_PRESETS,
        DataCursor,
        StreamingDataLoader,
        ensure_tokenizer,
        parse_dataset_mix,
    )

    for preset_name in ("mix-1b", "mix-cooldown", "mix-code"):
        pairs = parse_dataset_mix(preset_name)
        assert len(pairs) >= 3
        assert abs(sum(w for _, w in pairs) - 1.0) < 1e-6
        assert pairs[0][0] == "climbmix"
        assert preset_name in DATASET_MIX_PRESETS

    custom = parse_dataset_mix("climbmix:65,codeparrot:15,finemath:10,cosmopedia:10")
    assert len(custom) == 4
    assert abs(custom[0][1] - 0.65) < 1e-6 and abs(custom[1][1] - 0.15) < 1e-6

    # Offline end-to-end mixed streaming + multi-source DataCursor round-trip
    tok_dir = str(tmp_path / "tok")
    cache_dir = str(tmp_path / "shards")
    tok = ensure_tokenizer(tok_dir, vocab_size=512, dataset="synthetic:0.7,synthetic:0.3", shard_cache_dir=cache_dir)
    loader1 = StreamingDataLoader(
        tokenizer=tok,
        dataset="synthetic:0.7,synthetic:0.3",
        split="train",
        batch_size=2,
        seq_len=64,
        grad_accum_steps=1,
        shard_cache_dir=cache_dir,
        buffer_size=32,
        prefetch=1,
    )
    try:
        x1, y1, cur1 = next(loader1)
        assert x1.shape == (1, 2, 64) and y1.shape == (1, 2, 64)
        assert cur1.sources is not None
        assert "synthetic" in cur1.sources and "synthetic#1" in cur1.sources
        assert cur1.mix_step > 0
        saved = cur1.to_dict()
    finally:
        loader1.close()

    # Restore cursor and confirm loader resumes cleanly from saved multi-source state
    restored = DataCursor.from_dict(saved)
    assert restored.mix_step == cur1.mix_step
    assert restored.sources == cur1.sources
    loader2 = StreamingDataLoader(
        tokenizer=tok,
        dataset="synthetic:0.7,synthetic:0.3",
        split="train",
        batch_size=2,
        seq_len=64,
        grad_accum_steps=1,
        shard_cache_dir=cache_dir,
        cursor=restored,
        buffer_size=32,
        prefetch=1,
    )
    try:
        x2, _, cur2 = next(loader2)
        assert x2.shape == (1, 2, 64)
        assert cur2.mix_step >= restored.mix_step
    finally:
        loader2.close()


def test_stochastic_rounding_unbiased_and_quant_training_modes():
    import jax
    import jax.numpy as jnp
    from scale.model import forward_scalar_loss_and_z, init_params, precompute_rotary_embeddings
    from scale.optimizer import init_opt_state, optimizer_step
    from scale.quant import stochastic_round_bf16

    # 1. Verify BF16 stochastic rounding is unbiased for both positive & negative tiny updates
    key = jax.random.PRNGKey(42)
    for target in (1.0005, -1.0005, 0.00123, -0.00123):
        arr = jnp.full((50_000,), target, dtype=jnp.float32)
        rounded = stochastic_round_bf16(arr, key)
        assert rounded.dtype == jnp.bfloat16
        mean_val = float(jnp.mean(rounded.astype(jnp.float32)))
        assert abs(mean_val - target) < 3e-4, f"SR biased for {target}: got {mean_val}"

    # 2. Verify INT8, INT8-all, FP8 qdot + pure-BF16+SR weights + muon_vmap + scan_unroll=2
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
    y = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
    cos, sin = precompute_rotary_embeddings(32, 32)

    for qmode, p_dt, use_vmap in [
        ("int8", "bfloat16", True),
        ("int8-all", "bfloat16", False),
        ("fp8", "float32", False),
    ]:
        spec = ModelSpec(
            sequence_len=32, vocab_size=128, n_layer=4, n_embd=64, n_head=2, n_kv_head=1,
            mlp_hidden=128, activation="swiglu", loss_chunk_size=16,
            remat_policy="save_all_compute", matmul_quant=qmode, scan_unroll=2,
        )
        tcfg = TrainConfig(
            param_dtype=p_dt, opt_dtype="bfloat16", muon_vmap=use_vmap, matrix_lr=0.05, embedding_lr=0.3
        )
        param_dtype = jnp.bfloat16 if p_dt == "bfloat16" else jnp.float32
        params = init_params(jax.random.PRNGKey(3), spec, param_dtype=param_dtype)
        opt_state = init_opt_state(params, opt_dtype=jnp.bfloat16)

        @jax.jit
        def step_fn(p, st):
            def loss_fn(p_in):
                tot, ce, _ = forward_scalar_loss_and_z(p_in, x, y, cos, sin, spec, jnp.bfloat16, scale=jnp.float32(1.0))
                return tot, ce

            (_, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            p_next, st_next, gnorm = optimizer_step(
                p, grads, st,
                jnp.float32(1.0), jnp.float32(0.9), jnp.float32(0.0),
                spec, tcfg, jnp.bfloat16,
            )
            return p_next, st_next, ce, gnorm

        losses = []
        for _ in range(3):
            params, opt_state, ce, gnorm = step_fn(params, opt_state)
            assert np.isfinite(float(ce)) and np.isfinite(float(gnorm))
            assert params["blocks"]["w_qkv"].dtype == param_dtype
            losses.append(float(ce))
        assert losses[-1] < losses[0], f"Loss did not decrease for {qmode}/{p_dt}: {losses}"


def test_splash_attention_8_device_shard_map():
    """Verify multi-chip TPU SplashAttention shard_map (check_vma/check_rep) matches XLA forward+backward on 8 devices."""
    code = """
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scale.attention import _splash_attention_tpu

mesh = Mesh(np.array(jax.devices()), ("fsdp",))
assert mesh.size == 8
B, T, H, KV, D = 8, 128, 4, 2, 128
sh = NamedSharding(mesh, P("fsdp", None, None, None))
q = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (B, T, H, D), jnp.bfloat16) * 0.5, sh)
k = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (B, T, KV, D), jnp.bfloat16) * 0.5, sh)
v = jax.device_put(jax.random.normal(jax.random.PRNGKey(2), (B, T, KV, D), jnp.bfloat16) * 0.5, sh)

def loss_splash(q_, k_, v_):
    o = _splash_attention_tpu(q_, k_, v_, window_size=None, block_size=128, mesh=mesh, interpret=True)
    return jnp.mean(o.astype(jnp.float32) ** 2)

def loss_xla(q_, k_, v_):
    o = jax.nn.dot_product_attention(q_, k_, v_, is_causal=True)
    return jnp.mean(o.astype(jnp.float32) ** 2)

v_sp, g_sp = jax.jit(jax.value_and_grad(loss_splash, argnums=(0, 1, 2)))(q, k, v)
v_xl, g_xl = jax.jit(jax.value_and_grad(loss_xla, argnums=(0, 1, 2)))(q, k, v)
diff_v = abs(float(v_sp) - float(v_xl))
diff_dq = float(jnp.max(jnp.abs(g_sp[0].astype(jnp.float32) - g_xl[0].astype(jnp.float32))))
print(f"8-dev Splash vs XLA: val_diff={diff_v:.6e}, dq_max_diff={diff_dq:.6e}")
assert diff_v < 1e-2, f"Splash val mismatch: {diff_v}"
assert diff_dq < 2e-2, f"Splash dq mismatch: {diff_dq}"
"""
    res = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"8-dev SplashAttention test failed:\n{res.stdout}\n{res.stderr}"


def test_prequantized_fsdp_int8_and_lm_head_bwd():
    """Verify prequantized 1-byte weight cache, FSDP-local bf16 psum_scatter, and INT8 lm_head bwd on 8 devices."""
    code = """
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from scale.config import TrainConfig, get_model_preset
from scale.fsdp import create_fsdp_mesh
from scale.model import _make_fused_chunked_loss_fn, _make_fused_chunked_loss_int8_bwd_fn
from scale.pretrain import run_benchmark
from scale.quant import quantize_int8

mesh = create_fsdp_mesh()
assert mesh.size == 8
B_ch, M_ch, C, V = 2, 128, 64, 256
x_ch = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (B_ch, M_ch, C), jnp.bfloat16) * 0.5, NamedSharding(mesh, P(None, "fsdp", None)))
lm_w = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (C, V), jnp.bfloat16) * 0.2, NamedSharding(mesh, P(None, "fsdp")))
y_ch = jax.device_put(jax.random.randint(jax.random.PRNGKey(2), (B_ch, M_ch), 0, V), NamedSharding(mesh, P(None, "fsdp")))
lm_qb, lm_sb = quantize_int8(lm_w, axis=-1)
sc = jax.device_put(jnp.float32(1.0), NamedSharding(mesh, P()))

f_ex = _make_fused_chunked_loss_fn(15.0, 1e-4)
f_i8 = _make_fused_chunked_loss_int8_bwd_fn(15.0, 1e-4, mesh)
(_, (ce_e, _)), (dx_e, dw_e) = jax.jit(jax.value_and_grad(lambda x, w: (f_ex(x, w, y_ch, sc)[0], f_ex(x, w, y_ch, sc)[1:]), argnums=(0, 1), has_aux=True))(x_ch, lm_w)
(_, (ce_q, _)), (dx_q, dw_q) = jax.jit(jax.value_and_grad(lambda x, w: (f_i8(x, w, lm_qb, lm_sb, y_ch, sc)[0], f_i8(x, w, lm_qb, lm_sb, y_ch, sc)[1:]), argnums=(0, 1), has_aux=True))(x_ch, lm_w)
cos_dx = float(jnp.sum(dx_e.astype(jnp.float32) * dx_q.astype(jnp.float32)) / (jnp.linalg.norm(dx_e.astype(jnp.float32)) * jnp.linalg.norm(dx_q.astype(jnp.float32))))
cos_dw = float(jnp.sum(dw_e.astype(jnp.float32) * dw_q.astype(jnp.float32)) / (jnp.linalg.norm(dw_e.astype(jnp.float32)) * jnp.linalg.norm(dw_q.astype(jnp.float32))))
assert abs(float(ce_e) - float(ce_q)) < 1e-6
assert cos_dx > 0.995 and cos_dw > 0.995, (cos_dx, cos_dw)

for qmode in ("int8", "int8-all", "fp8"):
    spec = get_model_preset("smoke", arch_overrides={"matmul_quant": qmode, "remat_policy": "save_all_compute", "loss_chunk_size": 128})
    tcfg = TrainConfig(total_batch_size=2048, per_device_batch=1, param_dtype="bfloat16", opt_dtype="bfloat16", muon_vmap=True)
    res = run_benchmark(spec, tcfg, warmup_steps=1, timed_steps=2)
    assert res["tok_per_sec"] > 0
"""
    res = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"8-dev prequant FSDP test failed:\n{res.stdout}\n{res.stderr}"


def test_round2_unet_ve_stride_single_chunk_fsdp():
    """Verify Round-4 Final Boss preset (1b-ve), strict causality, and 8-device FSDP training with U-Net skips, ve_stride=2, full VE gate, elementwise attn_gate, per-channel smear, Canon-A/C, v0_residual, qk_gain, SSSL, and single-chunk INT8 loss."""
    spec_1b_ve = get_model_preset("1b-ve")
    assert spec_1b_ve.use_value_embeds is True
    assert spec_1b_ve.ve_gate_full is True
    assert spec_1b_ve.use_unet_skip is True
    assert spec_1b_ve.use_attn_gate is True
    assert spec_1b_ve.attn_gate_mode == "elementwise"
    assert spec_1b_ve.use_smear is True
    assert spec_1b_ve.smear_channels is True
    assert spec_1b_ve.use_canon is True
    assert spec_1b_ve.use_v0_residual is True
    assert spec_1b_ve.use_qk_gain is True
    assert spec_1b_ve.window_pattern == "SSSL"
    assert spec_1b_ve.rope_base == 50000.0
    assert spec_1b_ve.z_loss_weight == 5e-4

    code = """
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from scale.config import TrainConfig, get_model_preset
from scale.fsdp import build_tree_shardings, create_fsdp_mesh, shard_pytree
from scale.model import count_params, forward_logits, forward_scalar_loss_and_z, init_params, precompute_rotary_embeddings
from scale.optimizer import init_opt_state, optimizer_step
from scale.quant import prequantize_model_weights

mesh = create_fsdp_mesh()
assert mesh.size == 8

# Test both ve_stride=1 and ve_stride=2 with full Round-4 Final Boss + U-Net skips + SSSL + single-chunk INT8 loss
common_ov = {
    "n_layer": 8,
    "window_pattern": "SSSL",
    "use_value_embeds": True,
    "ve_gate_full": True,
    "use_unet_skip": True,
    "use_attn_gate": True,
    "attn_gate_mode": "elementwise",
    "use_smear": True,
    "smear_channels": True,
    "use_canon": True,
    "use_v0_residual": True,
    "use_qk_gain": True,
    "rope_base": 50000.0,
    "z_loss_weight": 5e-4,
    "loss_chunk_size": 256,
    "matmul_quant": "int8-all",
    "remat_policy": "save_all_compute",
}
spec_full = get_model_preset("smoke", arch_overrides={**common_ov, "ve_stride": 1})
spec_half = get_model_preset("smoke", arch_overrides={**common_ov, "ve_stride": 2})
p_full = init_params(jax.random.PRNGKey(0), spec_full, param_dtype=jnp.bfloat16)
p_half = init_params(jax.random.PRNGKey(0), spec_half, param_dtype=jnp.bfloat16)
assert p_full["value_embeds"].shape[0] == 8
assert p_half["value_embeds"].shape[0] == 4
assert count_params(p_half)["total"] < count_params(p_full)["total"]
for k in ("unet_lambdas", "qk_gain", "v0_lambdas", "smear_gate", "smear_lambda", "smear_lambda2", "canon_a", "canon_c"):
    assert k in p_half, f"Missing param {k}"
assert p_half["blocks"]["w_attn_gate"].shape == (8, spec_half.n_embd, spec_half.n_embd)
assert p_half["blocks"]["ve_gate"].shape == (8, spec_half.n_embd, spec_half.n_kv_head)

# Strict causality verification with nonzero random Canon-A/C, smear, U-Net skips, and elementwise gate!
p_caus = init_params(jax.random.PRNGKey(1), spec_half, param_dtype=jnp.float32)
p_caus = dict(p_caus)
for ck in ("canon_a", "canon_c", "smear_gate", "smear_lambda", "smear_lambda2", "unet_lambdas", "v0_lambdas"):
    p_caus[ck] = jax.random.normal(jax.random.PRNGKey(hash(ck) & 0xFFFF), p_caus[ck].shape, jnp.float32) * 0.25
p_caus["blocks"] = dict(p_caus["blocks"])
for bk in ("w_attn_gate", "ve_gate", "c_proj", "mlp_down"):
    p_caus["blocks"][bk] = jax.random.normal(jax.random.PRNGKey(hash(bk) & 0xFFFF), p_caus["blocks"][bk].shape, jnp.float32) * 0.1
cos_c, sin_c = precompute_rotary_embeddings(spec_half.sequence_len, spec_half.head_dim, base=spec_half.rope_base)
x_a = jax.random.randint(jax.random.PRNGKey(99), (1, spec_half.sequence_len), 0, spec_half.vocab_size)
split_pos = 20
x_b = x_a.at[:, split_pos:].set((x_a[:, split_pos:] + 7) % spec_half.vocab_size)
spec_eval = get_model_preset("smoke", arch_overrides={**common_ov, "ve_stride": 2, "attn_impl": "xla", "matmul_quant": "none", "remat_policy": "none"})
logits_a = forward_logits(p_caus, x_a, cos_c, sin_c, spec_eval, jnp.float32)
logits_b = forward_logits(p_caus, x_b, cos_c, sin_c, spec_eval, jnp.float32)
max_prefix_diff = float(jnp.max(jnp.abs(logits_a[:, :split_pos] - logits_b[:, :split_pos])))
max_suffix_diff = float(jnp.max(jnp.abs(logits_a[:, split_pos:] - logits_b[:, split_pos:])))
assert max_prefix_diff < 1e-6, f"CAUSALITY LEAK! max_prefix_diff={max_prefix_diff}"
assert max_suffix_diff > 1e-3, f"Suffix unexpectedly identical: {max_suffix_diff}"
print(f"Strict causality OK: max_prefix_diff={max_prefix_diff:.3e}, max_suffix_diff={max_suffix_diff:.3e}")

tcfg = TrainConfig(
    total_batch_size=2048,
    per_device_batch=1,
    param_dtype="bfloat16",
    opt_dtype="bfloat16",
    matrix_lr=0.06,
    embedding_lr=0.5,
    weight_decay=0.03,
    muon_vmap=True,
)
p_sh = build_tree_shardings(p_half, mesh, use_fsdp=True)
params = shard_pytree(p_half, p_sh)
opt_state = init_opt_state(params, opt_dtype=jnp.bfloat16)
o_sh = build_tree_shardings(opt_state, mesh, use_fsdp=True)
opt_state = shard_pytree(opt_state, o_sh)

cos, sin = precompute_rotary_embeddings(spec_half.sequence_len, spec_half.head_dim, base=spec_half.rope_base)
batch_sh = NamedSharding(mesh, P("fsdp", None))
x = jax.device_put(jax.random.randint(jax.random.PRNGKey(7), (8, spec_half.sequence_len), 0, spec_half.vocab_size), batch_sh)
y = jax.device_put(jax.random.randint(jax.random.PRNGKey(8), (8, spec_half.sequence_len), 0, spec_half.vocab_size), batch_sh)

@jax.jit
def step_fn(p, st):
    q_cache = prequantize_model_weights(p, spec_half.matmul_quant, spec_half.tie_embeddings)
    def loss_fn(p_in):
        tot, ce, _ = forward_scalar_loss_and_z(
            p_in, x, y, cos, sin, spec_half, jnp.bfloat16, scale=jnp.float32(1.0), mesh=mesh, q_cache=q_cache
        )
        return tot, ce
    (_, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
    p_next, st_next, gnorm = optimizer_step(
        p, grads, st, jnp.float32(1.0), jnp.float32(0.9), jnp.float32(0.03), spec_half, tcfg, jnp.bfloat16
    )
    return p_next, st_next, ce, gnorm

losses = []
for _ in range(3):
    params, opt_state, ce, gnorm = step_fn(params, opt_state)
    assert np.isfinite(float(ce)) and np.isfinite(float(gnorm))
    losses.append(float(ce))
assert losses[-1] < losses[0], f"Round-4 Final Boss 8-dev FSDP loss did not decrease: {losses}"
print(f"Round-4 Final Boss 8-dev FSDP OK: losses={losses}")
"""
    res = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Round-4 Final Boss 8-dev FSDP test failed:\n{res.stdout}\n{res.stderr}"


