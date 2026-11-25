# autoresearch-xla

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OttoApocalypse69/autoresearch-xla/blob/main/autoresearch_colab_kaggle.ipynb)

A pure **JAX/XLA** port of [Andrej Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch) engineered to run autonomous LLM pretraining research loops on **Google Colab** and **Kaggle** across **both GPUs and TPUs** — from a free single T4 GPU up to free 8-chip Kaggle TPU `v5e-8` pods and Colab `A100` / `H100` / `v6e-1` runtimes.

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
| `TPU v3-8` | Kaggle (Free) | 8 TPU | `bfloat16` | ~492 | 8 | 262,144 | On |
| `A100` | Colab Pro | 1 GPU | `bfloat16` | ~312 | 8 | 262,144 | Off |
| `TPU v5e-1` | Colab (Free) | 1 TPU | `bfloat16` | ~197 | 6 | 131,072 | On |
| `TPU v2-8` | Colab (Free) | 8 TPU | `bfloat16` | ~180 | 6 | 131,072 | On |
| `2× T4` | Kaggle (Free) | 2 GPU | `float16` | ~130 | 6 | 65,536 | On |
| `L4` | Colab Pro | 1 GPU | `bfloat16` | ~121 | 6 | 131,072 | On |
| `T4` / `P100` | Colab / Kaggle | 1 GPU | `float16` | ~65 / 21 | 4 | 32,768 | On |
| `CPU` | Local / CI | N CPU | `float32` | ~0.5/core | 2 | 1,024+ | Off |

## Quickstart

### Option 1: Google Colab or Kaggle Notebook (Recommended)

1. Open `autoresearch_colab_kaggle.ipynb` in Google Colab or Kaggle Notebooks.
2. Select any **GPU** or **TPU** runtime (`T4`, `2×T4`, `L4`, `A100`, `TPU v5e-1`, `TPU v5e-8`, etc.).
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
colab new -s autoresearch --tpu v5e1   # or --gpu T4 / L4 / A100 / H100
colab exec -s autoresearch -f autoresearch_colab_kaggle.ipynb
colab stop -s autoresearch             # always release the VM when finished

# Official Kaggle CLI (https://github.com/Kaggle/kaggle-cli)
uv tool install --with six kaggle
kaggle auth login                      # or place ~/.kaggle/kaggle.json / KAGGLE_API_TOKEN
kaggle quota                           # inspect weekly free GPU & TPU quota
kaggle kernels push -p . --accelerator TpuV5E8   # or NvidiaTeslaT4 (2xT4 GPU)
```

## Running the Test Suite

The test suite verifies data prep, tokenizer roundtrip & exact BPB math, `train.py` on both single-device and **4-device simulated SPMD** (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), `float16` dynamic loss scaling, Muon+AdamW convergence, `agent.py` diff patching + full end-to-end git ratchet with a mock LLM, and notebook validity:

```bash
pytest -v tests/test_suite.py
```

## License

MIT
