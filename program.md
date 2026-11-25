# autoresearch-xla

Autonomous LLM pretraining research in pure JAX/XLA for Google Colab & Kaggle GPUs and TPUs
(inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)).

## Setup

When running with an interactive coding agent (Claude Code, Codex CLI, Gemini CLI, etc.):

1. **Agree on a run tag**: propose a short tag based on today's date (e.g. `nov24-tpu` or `nov24-t4`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**:
   - `README.md` — repository overview and hardware profiles.
   - `prepare.py` — immutable constants, data prep, BPE tokenizer, hardware detection, dataloader, and `evaluate_bpb`. **Do not modify.**
   - `train.py` — the **only** file you modify. Pure JAX GPT model, Polar-Express NorMuon + AdamW optimizer, SPMD multi-chip sharding, and training loop.
4. **Verify data exists**: check `~/.cache/autoresearch-xla/` (or `$AUTORESEARCH_CACHE_DIR`) contains parquet shards and a tokenizer. If not, run `python prepare.py`.
5. **Initialize `results.tsv`**: create `results.tsv` with the header row (leave untracked by git so `git reset` never erases experiment history).
6. **Confirm and go**: run the unmodified baseline first, then enter the autonomous loop.

*(Note: when running unattended inside Google Colab or Kaggle via `agent.py` / `autoresearch_colab_kaggle.ipynb`, `agent.py` handles setup, git ratcheting, logging, and Drive/Kaggle persistence automatically using this document as its research brief.)*

## Experimentation

Each experiment runs `python train.py` on whatever accelerator is attached — single GPU (T4, P100, L4, A100, H100), multi-GPU (Kaggle 2×T4), single TPU (Colab v5e-1, v6e-1), or multi-chip TPU (Kaggle v3-8, v5e-8, Colab v2-8).

The training script runs for a **fixed wall-clock training budget of 5 minutes** (`TIME_BUDGET = 300` seconds, excluding XLA JIT compilation and warmup steps).

**What you CAN do:**
- Modify `train.py` — this is the **only** file you edit. Everything is fair game: model architecture, depth/width/heads/GQA, activations, norms, RoPE, value embeddings, sliding windows, optimizer (Muon/AdamW/Lion/etc.), LR schedules, batch sizes, rematerialization (`USE_REMAT`), and XLA/TPU/GPU throughput optimizations.

**What you CANNOT do:**
- Modify `prepare.py`. It is strictly read-only. It holds the fixed evaluation metric (`evaluate_bpb`), tokenizer, dataloader, sequence length, and time budget.
- Install new packages or change dependencies. Use only pure JAX, NumPy, Python stdlib, and imports from `prepare.py`.
- Cheat on evaluation: `evaluate_bpb` in `prepare.py` must be called with the real `eval_loss_fn` returning exact per-token negative log-likelihoods in nats over `MAX_SEQ_LEN`.

**The goal is simple: get the lowest `val_bpb` (validation bits-per-byte).** Lower is better. Because the wall-clock time budget is fixed, faster throughput (more tokens/steps in 5 minutes) and better sample efficiency (lower loss per token) compete on completely equal footing.

**Memory (VRAM / TPU HBM)** is a soft constraint: moderate increases are fine for meaningful `val_bpb` gains, but avoid OOM crashes.

**Simplicity criterion**: All else being equal, simpler is better. A tiny `0.0005` gain that adds 40 lines of brittle code is not worth keeping; deleting code for equal or better `val_bpb` is a huge win.

## JAX / GPU / TPU Engineering Rules

Keep these XLA-specific principles in mind when hacking `train.py`:

- **Keep array shapes static inside `train_step`**: never pass Python floats that change each step as static args; keep schedules as JAX scalar arrays (`jnp.float32`) so XLA compiles once and hits the persistent compilation cache.
- **TPU MXU alignment**: TPU matrix units operate on 128×128 tiles. Model dimensions (`n_embd`), MLP hidden dimensions, head dims (`64` or `128`), and padded vocab sizes that are multiples of `128` yield significantly higher MFU on TPUs (v2/v3/v5e/v6e) and Tensor Core GPUs.
- **Preserve multi-chip SPMD sharding**: `train.py` automatically shards batches across all devices (`Mesh(devices, ("data",))`). Ensure `TOTAL_BATCH_SIZE` stays cleanly divisible by `PER_DEVICE_BATCH * hw.num_devices * MAX_SEQ_LEN`.
- **Precision safety**: TPUs and Ampere+ GPUs (L4/A100/H100) excel at `bfloat16`; T4/P100 GPUs use `float16` with automatic dynamic loss scaling and `float32` Polar Express orthogonalization. Keep reductions (softmax, RMSNorm, loss, master weights) in `float32`.
- **Throughput levers worth exploring**:
  - `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `N_KV_HEADS` (GQA cuts KV memory and speeds up attention)
  - `TOTAL_BATCH_SIZE`, `PER_DEVICE_BATCH`, `USE_REMAT` (disabling remat speeds up compute if HBM permits; enabling it unlocks bigger models/batches on 16GB T4 or TPU v5e-1)
  - `WINDOW_PATTERN` (`"L"` vs `"SSSL"` vs `"SL"`), RoPE base frequency, learnable RMSNorm scales, SwiGLU vs ReLU², weight tying, z-loss, QK-Norm temperature, logit softcap
  - Muon hyperparameters (`MATRIX_LR`, `WEIGHT_DECAY`, Polar Express `ns_steps`, momentum warmup), AdamW betas & per-group LRs, cosine / trapezoidal / WSD cooldown schedules
  - `lax.scan` across homogeneous transformer layers (drastically cuts XLA compile time on deep models)

## Output Format

When `train.py` finishes cleanly, it prints:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4210.5
mfu_percent:      41.20
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
accelerator:      TPU v5 lite x8 (tpu, bfloat16)
```

Extract key metrics from `run.log` with:

```bash
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:" run.log
```

## Logging Results

Record every run in `results.tsv` (tab-separated, untracked by git):

```
commit	val_bpb	memory_gb	status	description
```

1. Short 7-char git commit hash
2. `val_bpb` (6 decimal places; `0.000000` on crash/timeout)
3. Peak per-device memory in GB (`peak_vram_mb / 1024`, `.1f`; `0.0` on crash)
4. Status: `keep`, `discard`, or `crash`
5. Concise single-line description of the experiment (no tabs or newlines)

## The Experiment Loop

On branch `autoresearch/<tag>`, repeat indefinitely:

1. Review current `train.py`, recent `git log`, and `results.tsv` (avoid repeating failed ideas).
2. Formulate a clear hypothesis and edit `train.py`.
3. `git commit -am "<short description>"`
4. Run `python train.py > run.log 2>&1` (never let stdout flood your context; enforce a ~10 min timeout).
5. Parse `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If empty (crash/OOM/NaN), inspect `tail -n 50 run.log`. Fix trivial typos once; if the idea is broken or OOM, log `crash` and revert.
7. Append the row to `results.tsv`.
8. If `val_bpb` improved (strictly lower, considering simplicity): **keep** the commit and advance the branch.
9. Otherwise: `git reset --hard HEAD~1` to revert `train.py` cleanly.

**NEVER STOP**: Once the experiment loop begins, continue autonomously until interrupted or until the Colab/Kaggle session wall-clock limit is reached. Think creatively — combine near-misses, optimize XLA throughput, refine optimizer dynamics, and push `val_bpb` as low as it will go!
