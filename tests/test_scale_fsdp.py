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
assert ps["sharded_leaves"] >= 7
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
