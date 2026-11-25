# autoresearch-xla

[![Open Autoresearch in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OttoApocalypse69/autoresearch-xla/blob/main/autoresearch_colab_kaggle.ipynb)
[![Open 1B FSDP Pretraining in Colab](https://img.shields.io/badge/Colab-1B%20FSDP%20Pretraining-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/OttoApocalypse69/autoresearch-xla/blob/main/pretrain_1b_colab_kaggle.ipynb)

A pure **JAX/XLA** port of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch) — plus a resumable **1B+ FSDP pretraining engine** (`scale/`) — engineered for **Google Colab** and **Kaggle** across **both GPUs and TPUs**.

## How It Works

Like Karpathy's original `autoresearch`, the entire research system fits into a handful of readable files:

| File | Owner | Role |
|---|---|---|
| `prepare.py` | **Immutable** | Fixed constants (`MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`), hardware detection, dataset download (`climbmix` / `tinystories` / `synthetic`), BPE tokenizer, background-prefetched BOS-aligned best-fit dataloader, and ground-truth `evaluate_bpb()`. |
| `train.py` | **AI Agent** | The **single editable file**. Pure JAX GPT (RoPE, QK-Norm, ResFormer Value Embeddings, ReLU², logit softcapping, sliding-window attention, residual lambdas), Polar-Express NorMuon + AdamW optimizer, automatic multi-chip SPMD sharding, and fixed 5-minute training loop. |
| `program.md` | **Human** | Research agenda and XLA/GPU/TPU engineering guidelines (used both by external CLI agents like Claude Code / Codex / Gemini CLI and by `agent.py`). |
| `agent.py` | **Runner** | Self-contained autonomous research loop for Colab & Kaggle notebooks. Calls Gemini, Anthropic, OpenAI, or OpenRouter/OpenAI-compatible APIs, applies verified `SEARCH/REPLACE` diffs, runs each experiment in an isolated subprocess, ratchets via git commit/reset, logs `results.tsv` + `experiments.jsonl`, plots `progress.png`, and syncs to Drive/Kaggle. |
| `autoresearch_colab_kaggle.ipynb` | **Notebook** | Ready-to-run Colab & Kaggle notebook with environment detection, secret loading, Drive/Kaggle persistence, hardware inspection, and live progress plots. |

## Why Pure JAX/XLA for Colab & Kaggle GPUs + TPUs?

- **One unified codebase for every accelerator**: Zero custom CUDA kernels (`flash-attn3` fails on T4, P100, and TPU). Uses `jax.nn.dot_product_attention` (cuDNN FlashAttention on Ampere+ GPUs, fused XLA attention on TPU and older GPUs).
- **Automatic multi-chip SPMD data parallelism**: Automatically builds a 1D `Mesh(jax.devices(), ("data",))` and shards batches across all 8 chips on Kaggle TPU `v5e-8` / `v3-8` or both GPUs on Kaggle `2×T4`, with zero code changes from 1-device setups.
- **Zero Cloud TPU lock conflicts**: `agent.py` and the notebook kernel **never** import JAX in the parent process — every `train.py` run executes in a clean subprocess, avoiding `/tmp/libtpu_lockfile` conflicts and freeing 100% of HBM/VRAM between experiments.
- **Hardware-safe mixed precision**: Uses `bfloat16` on TPUs and Ampere+ GPUs (`L4`, `A100`, `H100`), and `float16` with automatic dynamic loss scaling + `float32` Polar Express orthogonalization on Turing/Pascal GPUs (`T4`, `P100`).
- **Fast iteration**: Persistent XLA compilation caching (`jax_compilation_cache_dir`) lets hyperparameter tweaks reuse cached HLO executables in <1s, and a background prefetch thread overlaps CPU BPE packing with accelerator compute.
- **Exact UTF-8 `val_bpb`**: Computes exact raw byte lengths per token (`decode_single_token_bytes`), avoiding multi-byte UTF-8 replacement inflation.

## Auto-Scaled Hardware Profiles

Out of the box (`DEPTH = None`, `TOTAL_BATCH_SIZE = None`), `train.py` detects your accelerator and selects a fast, memory-safe baseline (which the agent can freely override during experiments):

| Accelerator | Platform | Devices | Dtype | Est. Peak TFLOPS | Default Depth | Default Batch (tok) | Remat |
|---|---|---|---|---|---|---|---|
| `TPU v5e-8` | Kaggle (Free) | 8 TPU | `bfloat16` | ~1,576 | 8 | 524,288 | Off |
| `TPU v6e-1` | Colab Pro | 1 TPU | `bfloat16` | ~918 | 8 | 524,288 | Off |
| `H100` | Colab Pro+ | 1 GPU | `bfloat16` | ~989 | 8 | 524,288 | Off |
| `RTX PRO 6000 (G4)` | Colab Pro+ | 1 GPU (96 GB) | `bfloat16` | ~504 | 8 | 524,288 | Off |
| `TPU v3-8` | Kaggle (Free) | 8 TPU | `bfloat16` | ~492 | 8 | 262,144 | On |
| `A100` | Colab Pro | 1 GPU | `bfloat16` | ~312 | 8 | 262,144 | Off |
| `TPU v5e-1` | Colab (Free) | 1 TPU | `bfloat16` | ~197 | 6 | 131,072 | Off |
| `TPU v2-8` | Colab (Free) | 8 TPU | `bfloat16` | ~180 | 6 | 131,072 | On |
| `2× T4` | Kaggle (Free) | 2 GPU | `float16` | ~130 | 6 | 65,536 | Off |
| `L4` | Colab Pro | 1 GPU | `bfloat16` | ~121 | 6 | 131,072 | On |
| `T4` / `P100` | Colab / Kaggle | 1 GPU | `float16` | ~65 / 21 | 4 | 32,768 | Off / On |
| `CPU` | Local / CI | N CPU | `float32` | ~0.5/core | 2 | 1,024+ | Off |

## Quickstart

### Option 1: Google Colab or Kaggle Notebook (Recommended)

1. Open `autoresearch_colab_kaggle.ipynb` in Google Colab or Kaggle Notebooks.
2. Select any **GPU** or **TPU** runtime (`T4`, `2×T4`, `L4`, `A100`, `G4`, `TPU v5e-1`, `TPU v5e-8`, etc.).
3. Add one API key to **Colab Secrets** or **Kaggle Secrets**:
   - `GEMINI_API_KEY` (Google AI Studio)
   - `ANTHROPIC_API_KEY` (Claude)
   - `OPENAI_API_KEY` (OpenAI)
   - `OPENROUTER_API_KEY` (OpenRouter)
4. Run all cells — results automatically sync to Google Drive (Colab) or `/kaggle/working/autoresearch-runs` (Kaggle) and safely resume if the runtime disconnects.

### Option 2: CLI Autonomous Runner (`agent.py`)

```bash
# Install lightweight deps (add `.[cpu]` or `.[cuda12]` locally; skip jax on Colab/Kaggle!)
pip install -e .

# One-time data & BPE tokenizer setup
python prepare.py --dataset climbmix --num-shards 4
# (Or `--dataset tinystories` for faster T4 demos, or `--dataset synthetic` offline)

# Run standalone baseline
python train.py

# Launch unattended research loop
export GEMINI_API_KEY="..."   # or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
python agent.py --provider auto --tag nov24-tpu --max-experiments 100 --max-hours 8.5
```

### Option 3: External Coding Agent (Claude Code, Codex CLI, Gemini CLI)

Just like Karpathy's original `autoresearch`, open your coding agent in this directory (or inside Colab Terminal!) and prompt:

> Read `program.md` and `train.py`, check that data is ready, create a fresh `autoresearch/<tag>` branch, run the baseline, and begin the autonomous experiment loop.

### Option 4: Remote GPU/TPU Orchestration via Official Colab CLI & Kaggle CLI

Run directly from your local terminal against remote Colab or Kaggle accelerators:

```bash
# Google Colab CLI (https://github.com/googlecolab/google-colab-cli)
uv tool install google-colab-cli
colab new -s autoresearch --tpu v5e1   # or --gpu T4 / L4 / A100 / H100 / G4
colab exec -s autoresearch -f autoresearch_colab_kaggle.ipynb
colab stop -s autoresearch             # always release the VM when finished

# Official Kaggle CLI (https://github.com/Kaggle/kaggle-cli)
uv tool install --with six kaggle
kaggle auth login                      # or place ~/.kaggle/kaggle.json / KAGGLE_API_TOKEN
kaggle quota                           # inspect weekly free GPU & TPU quota
kaggle kernels push -p . --accelerator TpuV5E8   # or NvidiaTeslaT4 (2xT4 GPU)
```

## Scaling Up to 1B+ Parameters (`scale/`)

Once `autoresearch-xla` discovers a strong architecture/optimizer recipe at proxy scale, scale it up with the high-MFU resumable multi-chip **1D FSDP pretraining engine** ([`scale/pretrain.py`](scale/pretrain.py) & [`pretrain_1b_colab_kaggle.ipynb`](pretrain_1b_colab_kaggle.ipynb)):

- **True 1D FSDP across 8 TPU chips / multi-GPU**: Shards 1.036B weights + NorMuon/AdamW states across 8 chips on Kaggle TPU `v5e-8` (**~1.05 GB/chip** static HBM in fp32, or **~525 MB/chip** in pure BF16+SR!) — and fits **1.036B on a single 16 GB Colab TPU `v5e-1` at `67.46%` pure-BF16 MFU (`84.83%` BF16-eq / `24,505 tok/s` with pre-quantized INT8-ALL)!**
- **Full MFU & Experimental Stack**:
  - Fused `w_qkv` & SwiGLU `w_gate_up` projections (all dims 256-aligned for TPU v5e/v6e MXUs & NVIDIA Tensor Cores)
  - Auto-dispatched **Pallas TPU `SplashAttention(1024)`** (fused backward kernel + cross-version `shard_map` FSDP) & **GPU `cuDNN FlashAttention`** (also wired into proxy `train.py`!)
  - **Flash/Splash-aware split rematerialization**: `save_all_compute` (0% GEMM & 0% Attention recompute, ~3× lower HBM than `none` by discarding redundant `silu`/`rms_norm` buffers) and `save_attn` (0% Attention/QKV recompute, ~7.5× lower HBM)
  - **Single-pass FSDP Fused Cut-Cross-Entropy** (`@jax.custom_vjp`): 0% `lm_head` backward recomputation, zero cross-chip resharding, ~16× lower logits HBM (`~268 MB` vs `4.3 GB`), plus exact-forward / **394-TOPS INT8-backward** `lm_head` in `int8-all`
  - **Pure-BF16 Master Weights + Unbiased Stochastic Rounding** (`--param-dtype bfloat16`, [`scale/quant.py`](scale/quant.py)): cuts 1B weight HBM by ~3× (`5.93 GB → 1.98 GB`), eliminates `fp32→bf16` staging casts, and unlocks `save_all_compute` even on a single 16 GB chip
  - **Pre-Quantized 1-Byte INT8 / FP8 Linear Engine** (`--matmul-quant int8 | int8-all | fp8`): pre-quantizes weights ONCE per step before gradient accumulation (halving weight HBM reads & multi-chip FSDP AllGather ICI traffic to 1 byte/weight!), saves 1-byte INT8 activations for `d_w`, uses reciprocal-multiply VPU quant, and runs FSDP-local `bf16` `d_w` `psum_scatter` on `v5e-8` — hitting **`84.83%` BF16-eq MFU (`24,505 tok/s`) on 1.036B** while **beating pure BF16 on live `mix-1b` `val_bpb`** (`1.807677`/`1.807863` vs `1.808011`)!
  - All 10 verified MaxText TPU `v5e-8` FSDP flags (`windowed_einsum` + backward AllGather pipelining + async collective fusion), 90% GPU VRAM preallocation, optional `--muon-vmap`, and async H2D double-buffering
  - **Autoresearch-discovered `1b-ve` preset**: Proxy sweep on live `mix-1b` (`100m`, 15.7M tokens/exp) found ResFormer Value Embeddings (`use_value_embeds=True`, zero extra GEMM FLOPs!) + `SSSL` sliding window + `--matrix-lr 0.04 --embedding-lr 0.4 --weight-decay 0.05` drops `val_bpb` by **`-0.0967` BPB** (`1.7867 → 1.6900`; `-0.0158` BPB at matched params without VE!).
- **Disk-bounded Multi-Dataset Streaming Mixer** ([`scale/data.py`](scale/data.py)): Deterministic smooth-weighted streaming across public, ungated HF Parquet corpora with exact per-source resumable `DataCursor` and rolling `/tmp` cleanup (`~3.4 GB` live verified on `mix-1b`, safe on Kaggle's 20 GB cap!):
  - `mix-1b` *(recommended 1B stable phase)*: `65%` ClimbMix-400B + `15%` CodeParrot-Clean (Python) + `10%` FineMath-4plus + `10%` Cosmopedia-v2
  - `mix-cooldown` *(recommended WSD decay phase)*: `40%` ClimbMix + `20%` Cosmopedia-v2 + `20%` FineMath-4plus + `20%` CodeParrot
  - `mix-code`: `50%` ClimbMix + `25%` CodeParrot + `15%` The-Stack-Smol-XL + `10%` FineMath-4plus
  - Or pass any single source (`climbmix`, `codeparrot`, `stack-code`, `finemath`, `cosmopedia`, `fineweb-edu`, `tinystories`, `synthetic`) or inline mix (`--dataset "climbmix:0.7,codeparrot:0.2,finemath:0.1"`). All mixes evaluate `val_bpb` on ClimbMix's pinned `shard_06542` for apples-to-apples tracking.
- **WSD (Warmup → Stable → Decay)** + **Topology-agnostic checkpoints**: Syncs weights (`safetensors`/`npz`), optimizer state, tokenizer, and multi-source cursor to **HuggingFace Hub** (`--sync-backend hf`), **Google Drive** (`--sync-backend local`), or **Kaggle Datasets** (`--sync-backend kaggle`).

### Verified Live MFU Benchmarks (Real Colab & Kaggle Hardware)

| Hardware | Model / Engine | Configuration | Throughput | MFU |
|---|---|---|---:|---:|
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | Proxy `train.py` (depth=6) | Baseline (unfused XLA attention) | 226,200 tok/s | 13.81% |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | Proxy `train.py` (depth=6) | **+ Pallas `SplashAttention(1024)`** | **694,000 tok/s** | **39.58%** (**3.07×**) |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` `350M` (367M) | Baseline (`xla` + `full` remat + unchunked CE) | 23,296 tok/s | 29.02% |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` `350M` (367M) | `Splash(1024)` + `save_all_compute` + Fused Cut-CE (`b=4`) | 48,059 tok/s | 59.86% (**2.06×**) |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` **`1B` (1.036B)** | `Splash(1024)` + `save_attn` + Fused Cut-CE (`fp32 w`, `bf16 opt`, `b=2`) | 18,138 tok/s | 62.79% |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` **`1B` (1.036B)** | **`BF16+SR` + `save_all_compute` + `Splash(1024)` + Fused Cut-CE (512K batch)** | **19,485 tok/s** | **67.46%** (pure BF16) |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` **`1B` (1.036B)** | **Pre-quant `--matmul-quant int8` (INT8 fwd/dx + BF16 dw, 262K)** | **22,116 tok/s** | **76.56%** *(BF16-eq)* |
| **Colab TPU `v5e-1`** (16 GB, 197 TF) | `scale/` **`1B` (1.036B)** | **Pre-quant `--matmul-quant int8-all` (1B + INT8 `lm_head` bwd, 512K)** | **24,505 tok/s** | **84.83%** *(BF16-eq)* |
| **Colab GPU `L4`** (24 GB, 121 TF) | `scale/` `350M` (367M) | Baseline (`xla` + `full` remat + unchunked CE) | 6,196 tok/s | 12.56% |
| **Colab GPU `L4`** (24 GB, 121 TF) | `scale/` `350M` (367M) | `cuDNN FlashAttention` + `save_attn` + chunked CE (`b=8`) | 10,885 tok/s | 22.07% (**1.76×**) |
| **Kaggle `2× T4`** (32 GB, 130 TF) | Proxy `train.py` (26.3M) | Multi-GPU SPMD `float16` + dynamic loss scaling | 95,000 tok/s | 9.90% |
| **Colab `T4`** (16 GB, 65 TF) | Proxy `train.py` (11.5M) | Single-GPU `float16` + dynamic loss scaling | 101,000 tok/s | 8.85% |

```bash
# Multi-session 1B pretraining on Kaggle TPU v5e-8 (stable phase on mix-1b, ~190K tok/s -> ~28h for 20B tokens!):
# Use `--preset 1b` (1.036B dense) OR `--preset 1b-ve` (1.37B total / 1.036B active autoresearch winner, -0.097 BPB!)
python -m scale.pretrain --preset 1b-ve --dataset mix-1b \
    --remat-policy save_all_compute --splash-block-size 1024 \
    --param-dtype bfloat16 --opt-dtype bfloat16 --matmul-quant int8-all --muon-vmap \
    --matrix-lr 0.04 --embedding-lr 0.4 --weight-decay 0.05 \
    --sync-backend hf --remote-target <your-user>/my-1b-base \
    --phase stable --max-hours 8.5

# Trigger final WSD cooldown decay in pure BF16 on high-density mix-cooldown (~16-18B tokens, same checkpoint!):
python -m scale.pretrain --preset 1b-ve --dataset mix-cooldown \
    --remat-policy save_all_compute --splash-block-size 1024 \
    --param-dtype bfloat16 --opt-dtype bfloat16 --matmul-quant none --muon-vmap \
    --matrix-lr 0.04 --embedding-lr 0.4 --weight-decay 0.05 \
    --sync-backend hf --remote-target <your-user>/my-1b-base \
    --phase decay --decay-steps 4000 --max-hours 8.5
```

## Running the Test Suite

Run both the proxy autoresearch suite (20 tests) and the 8-device FSDP 1B pretraining, multi-dataset mixer, BF16+SR, pre-quantized INT8/FP8, 8-device SplashAttention `shard_map`, & Fused Cut-Cross-Entropy suite (17 tests):

```bash
pytest -v tests/
```

## License

MIT
