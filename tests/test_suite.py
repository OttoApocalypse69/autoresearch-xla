"""
Comprehensive end-to-end test suite for autoresearch-xla.

Verifies:
  1. prepare.py synthetic dataset, BPE tokenizer, exact UTF-8 token_bytes, and BOS-aligned dataloader
  2. Hardware auto-scaling profiles for all Colab & Kaggle GPUs and TPUs
  3. Pure JAX GPT forward/backward with GQA, SSSL sliding windows, remat, bf16/fp16/fp32, & NorMuon convergence
  4. Full multi-device SPMD training + grad_accum via XLA_FLAGS=--xla_force_host_platform_device_count=4
  5. agent.py diff parsing, syntax & evaluate_bpb guardrails
  6. Full autonomous git ratchet loop (baseline -> edit -> keep/discard -> crash revert -> sync -> resume -> plot)
  7. Colab/Kaggle notebook JSON & Python cell syntax validity
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))
import agent  # noqa: E402
import prepare  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Hardware Profiles Across All Colab & Kaggle GPUs and TPUs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "platform,kind,n_dev,hbm,exp_dtype,exp_depth",
    [
        ("tpu", "TPU v5 lite", 8, 16.0, "bfloat16", 8),
        ("tpu", "TPU v5 lite", 1, 16.0, "bfloat16", 6),
        ("tpu", "TPU v6 lite", 1, 32.0, "bfloat16", 8),
        ("tpu", "TPU v3", 8, 16.0, "bfloat16", 8),
        ("tpu", "TPU v2", 8, 8.0, "bfloat16", 6),
        ("gpu", "NVIDIA H100 80GB HBM3", 1, 80.0, "bfloat16", 8),
        ("gpu", "NVIDIA A100-SXM4-40GB", 1, 40.0, "bfloat16", 8),
        ("gpu", "NVIDIA L4", 1, 24.0, "bfloat16", 6),
        ("gpu", "Tesla T4", 2, 16.0, "float16", 6),
        ("gpu", "Tesla T4", 1, 16.0, "float16", 4),
        ("gpu", "Tesla P100-PCIE-16GB", 1, 16.0, "float16", 4),
        ("cpu", "cpu", 4, 0.0, "float32", 2),
    ],
)
def test_hardware_profiles(platform, kind, n_dev, hbm, exp_dtype, exp_depth):
    dtype, tflops, depth, total_batch, per_dev_batch, win_pat, remat = (
        prepare._estimate_peak_tflops_and_profile(platform, kind, n_dev, hbm)
    )
    assert dtype == exp_dtype
    assert depth == exp_depth
    assert tflops > 0
    assert total_batch >= per_dev_batch * n_dev
    assert all(c in "SL" for c in win_pat)
    assert isinstance(remat, bool)


# ---------------------------------------------------------------------------
# 2. Data Preparation, BPE Tokenizer & BOS-Aligned Packing Dataloader
# ---------------------------------------------------------------------------

def test_prepare_synthetic_and_dataloader(tmp_path):
    cache_dir = tmp_path / "cache"
    env = {
        **os.environ,
        "AUTORESEARCH_CACHE_DIR": str(cache_dir),
        "AUTORESEARCH_VOCAB_SIZE": "512",
        "AUTORESEARCH_SEQ_LEN": "64",
    }
    res = subprocess.run(
        [PYTHON, "prepare.py", "--dataset", "synthetic"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Ready to run" in res.stdout

    tok_dir = str(cache_dir / "tokenizer")
    tok = prepare.Tokenizer.from_directory(tok_dir)
    assert tok.get_vocab_size() == 512

    # Roundtrip including Unicode
    sample = "Autoresearch on TPU v5e-8 & T4! 🚀 你好 12345"
    assert tok.decode(tok.encode(sample)) == sample

    # Exact UTF-8 token_bytes checks
    tb = prepare.get_token_bytes(tok_dir)
    assert tb.shape == (512,)
    assert tb.dtype == np.int32
    assert tb[tok.get_bos_token_id()] == 0
    assert int((tb > 0).sum()) == 512 - len(prepare.SPECIAL_TOKENS)

    # Check BOS-aligned best-fit packing shapes (single & grad_accum)
    orig_data_dir = prepare.DATA_DIR
    try:
        prepare.DATA_DIR = str(cache_dir / "data")
        loader = prepare.make_dataloader(tok, batch_size=4, seq_len=64, split="train", prefetch=1, grad_accum_steps=3)
        try:
            x, y, ep = next(loader)
            assert x.shape == (3, 4, 64)
            assert y.shape == (3, 4, 64)
            assert x.dtype == np.int32
            assert ep >= 1
            # Every sequence must begin with BOS token
            assert np.all(x[:, :, 0] == tok.get_bos_token_id())
        finally:
            loader.close()
    finally:
        prepare.DATA_DIR = orig_data_dir


# ---------------------------------------------------------------------------
# 3. Pure JAX GPT + GQA + SSSL Sliding Window + Remat + dtypes + Muon Convergence
# ---------------------------------------------------------------------------

def test_model_gqa_sssl_remat_dtypes_and_muon_convergence():
    import jax
    import jax.numpy as jnp
    import train

    for compute_dtype, ns_dtype in [
        (jnp.float32, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
        (jnp.float16, jnp.float32),
    ]:
        cfg = train.GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=4,
            n_head=4,
            n_kv_head=2,         # GQA (2:1)
            n_embd=64,
            window_pattern="SSSL",
            use_remat=True,      # Activation checkpointing enabled
            softcap=15.0,
        )
        win_sizes = train.compute_window_sizes(cfg)
        assert win_sizes == ((16, 0), (16, 0), (16, 0), None)

        params = train.init_gpt_params(jax.random.PRNGKey(0), cfg)
        opt_state = train.init_opt_state(params)
        cos, sin = train.precompute_rotary_embeddings(32, 16)

        rng = np.random.default_rng(7)
        x = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))
        y = jnp.asarray(rng.integers(0, 128, size=(2, 32), dtype=np.int32))

        def loss_fn(p):
            nll = train.gpt_forward_loss_2d(p, x, y, cos, sin, cfg, win_sizes, compute_dtype)
            assert nll.shape == (2, 32)
            assert nll.dtype == jnp.float32
            return jnp.mean(nll)

        step_fn = jax.jit(lambda p, st: _one_opt_step(train, loss_fn, p, st, ns_dtype))

        losses = []
        for _ in range(8):
            params, opt_state, loss_val = step_fn(params, opt_state)
            lf = float(loss_val)
            assert np.isfinite(lf), f"Non-finite loss for dtype={compute_dtype}"
            losses.append(lf)

        # Loss must strictly drop over 8 steps of NorMuon + AdamW across fp32, bf16, and fp16
        assert losses[-1] < losses[0] - 0.5, f"Loss failed to decrease for {compute_dtype}: {losses}"


def test_hf_tokenizers_fallback_exact_utf8_bytes(tmp_path, monkeypatch):
    """Ensure fallback to HuggingFace `tokenizers` works cleanly when rustbpe is absent."""
    import builtins

    cache_dir = tmp_path / "hf_cache"
    monkeypatch.setattr(prepare, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(prepare, "DATA_DIR", str(cache_dir / "data"))
    monkeypatch.setattr(prepare, "TOKENIZER_DIR", str(cache_dir / "tokenizer"))
    monkeypatch.setattr(prepare, "VOCAB_SIZE", 512)

    prepare.generate_synthetic_data(num_train_docs=300, num_val_docs=50)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rustbpe":
            raise ImportError("simulated missing rustbpe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    prepare.train_tokenizer()

    tok = prepare.Tokenizer.from_directory(str(cache_dir / "tokenizer"))
    assert tok.backend == "tokenizers"
    sample = "Colab TPU v5e-8 & Kaggle T4! 你好 🚀"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample

    tb = prepare.get_token_bytes(str(cache_dir / "tokenizer"))
    assert int(tb[ids].sum()) == len(sample.encode("utf-8"))


def _one_opt_step(train_mod, loss_fn, params, opt_state, ns_dtype):
    import jax
    import jax.numpy as jnp

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    new_params, new_state = train_mod.optimizer_step(
        params,
        grads,
        opt_state,
        jnp.float32(1.0),
        jnp.float32(0.9),
        jnp.float32(0.1),
        dmodel_lr_scale=1.0,
        ns_dtype=ns_dtype,
    )
    return new_params, new_state, loss_val


# ---------------------------------------------------------------------------
# 4. Multi-Device SPMD Training Subprocess (Simulated 4-Chip TPU/GPU + Grad Accum)
# ---------------------------------------------------------------------------

def test_train_spmd_4_devices_with_grad_accum(tmp_path):
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "repo"
    work_dir.mkdir()
    shutil.copy2(REPO_ROOT / "prepare.py", work_dir / "prepare.py")
    # Modify TOTAL_BATCH_SIZE to force grad_accum_steps == 2 on 4 simulated devices!
    # 4 devices * per_device=2 * seq_len=64 = 512 tokens/micro -> 1024 tokens = 2 grad_accum_steps
    train_src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    train_src = train_src.replace("TOTAL_BATCH_SIZE = None", "TOTAL_BATCH_SIZE = 1024", 1)
    train_src = train_src.replace("WINDOW_PATTERN = None", 'WINDOW_PATTERN = "SSSL"', 1)
    (work_dir / "train.py").write_text(train_src, encoding="utf-8")

    env = {
        **os.environ,
        "AUTORESEARCH_CACHE_DIR": str(cache_dir),
        "AUTORESEARCH_SEQ_LEN": "64",
        "AUTORESEARCH_TIME_BUDGET": "2",
        "AUTORESEARCH_EVAL_TOKENS": "1024",
        "AUTORESEARCH_VOCAB_SIZE": "512",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }

    subprocess.run([PYTHON, "prepare.py", "--dataset", "synthetic"], cwd=str(work_dir), env=env, check=True)
    res = subprocess.run([PYTHON, "train.py"], cwd=str(work_dir), env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"train.py failed:\n{res.stdout}\n{res.stderr}"
    assert "grad_accum=2" in res.stdout

    metrics = agent.parse_run_log(res.stdout)
    assert metrics.ok
    assert 0.0 < metrics.val_bpb < 10.0
    assert metrics.num_steps >= 3
    assert "x4" in metrics.accelerator


# ---------------------------------------------------------------------------
# 5. Agent Diff Parsing & Static Guardrails
# ---------------------------------------------------------------------------

def test_agent_diff_parser_and_guardrails():
    base_code = (REPO_ROOT / "train.py").read_text(encoding="utf-8")

    # Valid edit
    resp = """
<hypothesis>
Higher embedding LR speeds up early convergence on short time budgets.
</hypothesis>
<description>
increase EMBEDDING_LR from 0.6 to 0.8
</description>
<edit>
<<<<<<< SEARCH
EMBEDDING_LR = 0.6        # Learning rate for token & value embeddings (AdamW)
=======
EMBEDDING_LR = 0.8        # Learning rate for token & value embeddings (AdamW)
>>>>>>> REPLACE
</edit>
"""
    prop = agent.parse_llm_proposal(resp)
    assert prop.description == "increase EMBEDDING_LR from 0.6 to 0.8"
    updated = agent.apply_edits_safely(base_code, prop.blocks)
    assert "EMBEDDING_LR = 0.8" in updated

    # Rejects missing SEARCH block
    with pytest.raises(ValueError, match="not found"):
        agent.apply_edits_safely(base_code, [("NON_EXISTENT_LINE_XYZ", "x = 1")])

    # Rejects Python syntax error
    with pytest.raises(ValueError, match="SyntaxError"):
        agent.apply_edits_safely(
            base_code,
            [("SEED = 42", "SEED = (42  # broken paren")],
        )

    # Rejects removing evaluate_bpb call
    with pytest.raises(ValueError, match="Guardrail violation"):
        agent.apply_edits_safely(
            base_code,
            [
                (
                    "    val_bpb = evaluate_bpb(\n"
                    "        eval_loss_fn,\n"
                    "        params,\n"
                    "        tokenizer,\n"
                    "        global_micro_batch,\n"
                    "        data_sharding=batch_sharding,\n"
                    "    )",
                    "    val_bpb = 0.0001",
                )
            ],
        )


# ---------------------------------------------------------------------------
# 6. Full Autonomous Git Ratchet Loop (Baseline -> Keep/Discard -> Crash -> Sync -> Resume)
# ---------------------------------------------------------------------------

def test_agent_full_git_ratchet_loop_and_resume(tmp_path):
    work_dir = tmp_path / "repo"
    sync_dir = tmp_path / "drive_sync"
    cache_dir = tmp_path / "cache"
    work_dir.mkdir()

    for fname in ("prepare.py", "train.py", "program.md", ".gitignore"):
        shutil.copy2(REPO_ROOT / fname, work_dir / fname)

    env_overrides = {
        "AUTORESEARCH_CACHE_DIR": str(cache_dir),
        "AUTORESEARCH_SEQ_LEN": "64",
        "AUTORESEARCH_TIME_BUDGET": "1",
        "AUTORESEARCH_EVAL_TOKENS": "512",
        "AUTORESEARCH_VOCAB_SIZE": "512",
    }
    old_env = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        subprocess.run([PYTHON, "prepare.py", "--dataset", "synthetic"], cwd=str(work_dir), check=True)

        # Scripted mock LLM:
        #   Call 1: valid LR tweak (trains cleanly -> keep or discard)
        #   Call 2: deliberate runtime crash -> verifies crash logging & git revert!
        #   Call 3 (on resumed session): valid softcap tweak -> verifies resume skips baseline!
        proposals = [
            """
<hypothesis>Lower weight decay slightly for ultra-short runs.</hypothesis>
<description>reduce WEIGHT_DECAY to 0.1</description>
<edit>
<<<<<<< SEARCH
WEIGHT_DECAY = 0.2        # Cautious weight decay for Muon (decayed linearly to 0)
=======
WEIGHT_DECAY = 0.1        # Cautious weight decay for Muon (decayed linearly to 0)
>>>>>>> REPLACE
</edit>
""",
            """
<hypothesis>Deliberate crash test to verify git rollback and crash logging.</hypothesis>
<description>trigger deliberate runtime error</description>
<edit>
<<<<<<< SEARCH
    t_start = time.time()
=======
    t_start = time.time()
    raise RuntimeError("simulated OOM crash")
>>>>>>> REPLACE
</edit>
""",
            """
<hypothesis>Adjust logit softcap slightly.</hypothesis>
<description>set SOFTCAP to 20.0</description>
<edit>
<<<<<<< SEARCH
SOFTCAP = 15.0            # Logit softcapping threshold
=======
SOFTCAP = 20.0            # Logit softcapping threshold
>>>>>>> REPLACE
</edit>
""",
        ]
        call_idx = {"n": 0}

        def mock_llm(_sys_prompt: str, _user_prompt: str) -> str:
            idx = call_idx["n"]
            call_idx["n"] += 1
            return proposals[idx]

        # Session 1: baseline + 2 experiments (1 valid, 1 crash)
        rows1 = agent.run_autoresearch_loop(
            repo_dir=work_dir,
            tag="test-run",
            llm_complete=mock_llm,
            python_exe=PYTHON,
            max_experiments=2,
            run_timeout=120,
            sync_dir=sync_dir,
            resume=True,
            fix_crashes=False,
        )

        assert len(rows1) == 3  # baseline + 2 proposals
        assert rows1[0]["status"] == "keep" and rows1[0]["description"] == "baseline"
        assert rows1[1]["status"] in ("keep", "discard")
        assert rows1[2]["status"] == "crash"

        # Crash must have been cleanly reverted from train.py
        assert "simulated OOM crash" not in (work_dir / "train.py").read_text(encoding="utf-8")

        # Sync dir must contain persisted results, jsonl, best_train.py, and progress.png
        for f in ("results.tsv", "experiments.jsonl", "best_train.py", "progress.png"):
            assert (sync_dir / f).exists(), f"Missing synced artifact: {f}"

        # Session 2: wipe local results.tsv and resume from sync_dir -> should NOT re-run baseline
        (work_dir / "results.tsv").unlink()
        rows2 = agent.run_autoresearch_loop(
            repo_dir=work_dir,
            tag="test-run",
            llm_complete=mock_llm,
            python_exe=PYTHON,
            max_experiments=1,
            run_timeout=120,
            sync_dir=sync_dir,
            resume=True,
            fix_crashes=False,
        )
        assert len(rows2) == 4
        assert sum(1 for r in rows2 if r["description"] == "baseline") == 1
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# 7. Notebook JSON & Code Cell Syntax Validation
# ---------------------------------------------------------------------------

def test_notebook_valid_json_and_python_syntax():
    nb_path = REPO_ROOT / "autoresearch_colab_kaggle.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        ast.parse(src, filename=f"notebook_cell_{idx}")
