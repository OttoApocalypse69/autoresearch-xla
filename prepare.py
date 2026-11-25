"""
Immutable data preparation, tokenizer, hardware detection, dataloader, and BPB
evaluation harness for autoresearch-xla (pure JAX/XLA for Colab & Kaggle GPUs/TPUs).

DO NOT MODIFY THIS FILE DURING EXPERIMENTS.
Everything here defines the fixed evaluation yardstick (`val_bpb`).

Usage:
    python prepare.py                          # ClimbMix (default: 4 train shards + pinned val)
    python prepare.py --num-shards 10          # More training data (~1 GB)
    python prepare.py --dataset tinystories    # Smaller/faster dataset for quick T4 tests
    python prepare.py --dataset synthetic      # Zero-network synthetic corpus for smoke tests
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from multiprocessing import Pool
import os
import pickle
import queue
import sys
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY during an experiment series)
# Env overrides exist strictly for initial setup / automated smoke tests.
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = int(os.environ.get("AUTORESEARCH_SEQ_LEN", "2048"))
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "300"))  # 5 min wall-clock
EVAL_TOKENS = int(os.environ.get("AUTORESEARCH_EVAL_TOKENS", str(8 * 524288)))  # ~4.19M tokens
VOCAB_SIZE = int(os.environ.get("AUTORESEARCH_VOCAB_SIZE", "8192"))

# ---------------------------------------------------------------------------
# Paths & Dataset Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get(
    "AUTORESEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-xla"),
)
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
JAX_CACHE_DIR = os.path.join(CACHE_DIR, "jax_cache")

CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542
CLIMBMIX_VAL_SHARD = CLIMBMIX_MAX_SHARD
VAL_FILENAME = f"shard_{CLIMBMIX_VAL_SHARD:05d}.parquet"

TINYSTORIES_TRAIN_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
]
TINYSTORIES_VAL_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
    "validation-00000-of-00001-869c898b519ad725.parquet"
)

# GPT-4 / nanochat style BPE split pattern
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


# ---------------------------------------------------------------------------
# Hardware Detection & Colab/Kaggle Auto-Scaling Profiles
# (Lazy-imports JAX so calling `python prepare.py` never grabs TPU locks!)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareInfo:
    platform: str                  # "tpu", "gpu", or "cpu"
    device_kind: str               # e.g., "TPU v5 lite", "Tesla T4", "NVIDIA A100-SXM4-40GB"
    num_devices: int               # 1, 2, 4, 8, ...
    hbm_per_device_gb: float       # memory per JAX device in GiB
    compute_dtype: str             # "bfloat16", "float16", or "float32"
    peak_tflops_total: float       # aggregate peak matmul TFLOPS across all devices
    default_depth: int
    default_total_batch_size: int  # tokens per optimizer step
    default_per_device_batch: int  # sequences per device per micro-step
    default_window_pattern: str
    default_remat: bool


def _estimate_peak_tflops_and_profile(platform: str, kind: str, num_devices: int, hbm_gb: float):
    """Map detected Colab/Kaggle GPU or TPU to peak TFLOPS and sensible defaults."""
    k = kind.lower()

    if platform == "tpu":
        if "v6" in k or "trillium" in k:
            # TPU v6e (Colab Pro v6e-1: 32GB, 918 TFLOPS bf16 per chip)
            per_dev_tflops = 918.0
            depth = 8
            per_dev_batch = 16 if num_devices == 1 else 16
            total_batch = 2**19
            remat = False
        elif "v5e" in k or "v5 lite" in k:
            # TPU v5e (Colab free v5e-1: 16GB; Kaggle free v5e-8: 8x16GB=128GB)
            per_dev_tflops = 197.0
            if num_devices >= 8:
                depth, per_dev_batch, total_batch, remat = 8, 16, 2**19, False
            elif num_devices >= 4:
                depth, per_dev_batch, total_batch, remat = 8, 16, 2**18, False
            else:
                depth, per_dev_batch, total_batch, remat = 6, 16, 2**17, False
        elif "v5p" in k or "v5" in k:
            per_dev_tflops = 459.0
            depth, per_dev_batch, total_batch, remat = 8, 16, 2**19, False
        elif "v4" in k:
            per_dev_tflops = 275.0
            depth, per_dev_batch, total_batch, remat = 8, 8, 2**18, False
        elif "v3" in k:
            # Kaggle TPU v3-8 exposes 8 cores (16GB HBM each, ~61.5 TFLOPS bf16 per core)
            per_dev_tflops = 61.5
            depth = 8 if num_devices >= 8 else 6
            per_dev_batch = 8
            total_batch = 2**18 if num_devices >= 8 else 2**17
            remat = True
        else:
            # TPU v2-8 (Colab free: 8 cores, 8GB HBM each, ~22.5 TFLOPS bf16 per core)
            per_dev_tflops = 22.5
            depth = 6
            per_dev_batch = 4
            total_batch = 2**17 if num_devices >= 8 else 2**16
            remat = True
        return "bfloat16", per_dev_tflops * num_devices, depth, total_batch, per_dev_batch, "SSSL", remat

    if platform == "gpu":
        if "b200" in k or "b100" in k:
            return "bfloat16", 2250.0 * num_devices, 10, 2**19, 32, "SSSL", False
        if "h200" in k or "h100" in k:
            return "bfloat16", 989.5 * num_devices, 8, 2**19, 32, "SSSL", False
        if "rtx pro 6000" in k or ("6000" in k and "blackwell" in k):
            # Colab G4: NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7, 504 TFLOPS dense bf16)
            return "bfloat16", 504.0 * num_devices, 8, 2**19, 32, "SSSL", False
        if "a100" in k:
            per_dev_batch = 32 if hbm_gb >= 70 else 16
            return "bfloat16", 312.0 * num_devices, 8, 2**18, per_dev_batch, "SSSL", False
        if "l40" in k:
            return "bfloat16", 181.0 * num_devices, 8, 2**18, 16, "SSSL", False
        if "l4" in k:
            return "bfloat16", 121.0 * num_devices, 6, 2**17, 16, "SSSL", False
        if "a10" in k or "rtx 40" in k or "rtx 50" in k or "rtx 30" in k or "blackwell" in k:
            return "bfloat16", 125.0 * num_devices, 6, 2**17, 8, "SSSL", True
        if "v100" in k:
            return "float16", 125.0 * num_devices, 6, 2**17, 8, "L", True
        if "t4" in k:
            # Colab T4 (1x16GB) or Kaggle 2xT4 (2x16GB): fp16 Tensor Cores, no remat needed at depth=4..6
            if num_devices >= 2:
                return "float16", 65.0 * num_devices, 6, 2**16, 8, "L", False
            return "float16", 65.0 * num_devices, 4, 2**15, 8, "L", False
        if "p100" in k:
            # Kaggle P100 (16GB, SM 6.0): fp16 ALU is 2x fp32
            return "float16", 21.2 * num_devices, 4, 2**15, 4, "L", True
        # Unknown modern GPU fallback
        return "bfloat16", 100.0 * num_devices, 6, 2**17, 8, "L", True

    # CPU fallback (local development / smoke tests)
    # Keep tiny so tests & quick sanity checks finish fast.
    seq_cap = max(MAX_SEQ_LEN, 1)
    per_dev_batch = 2
    total_batch = max(per_dev_batch * num_devices * seq_cap, 2**10)
    return "float32", 0.5 * num_devices, 2, total_batch, per_dev_batch, "L", False


def detect_hardware() -> HardwareInfo:
    """Detect active JAX backend (TPU / GPU / CPU) and return HardwareInfo."""
    import jax

    devices = jax.devices()
    dev0 = devices[0]
    platform = str(dev0.platform).lower()
    device_kind = str(getattr(dev0, "device_kind", platform))
    num_devices = len(devices)

    hbm_gb = 16.0
    try:
        stats = dev0.memory_stats()
        if stats and stats.get("bytes_limit"):
            hbm_gb = stats["bytes_limit"] / (1024**3)
    except Exception:
        pass
    if platform == "cpu":
        hbm_gb = 0.0

    dtype, tflops, depth, total_batch, per_dev_batch, win_pat, remat = (
        _estimate_peak_tflops_and_profile(platform, device_kind, num_devices, hbm_gb)
    )

    # Ensure total_batch is cleanly divisible by (per_dev_batch * num_devices * MAX_SEQ_LEN)
    step_tokens = per_dev_batch * num_devices * MAX_SEQ_LEN
    if total_batch < step_tokens:
        total_batch = step_tokens
    elif total_batch % step_tokens != 0:
        total_batch = ((total_batch + step_tokens - 1) // step_tokens) * step_tokens

    return HardwareInfo(
        platform=platform,
        device_kind=device_kind,
        num_devices=num_devices,
        hbm_per_device_gb=round(hbm_gb, 2),
        compute_dtype=dtype,
        peak_tflops_total=tflops,
        default_depth=depth,
        default_total_batch_size=total_batch,
        default_per_device_batch=per_dev_batch,
        default_window_pattern=win_pat,
        default_remat=remat,
    )


def get_peak_memory_mb() -> float:
    """Return peak accelerator memory (per-device max in MiB) on TPU/GPU, or RSS on CPU."""
    import jax

    peaks = []
    for dev in jax.devices():
        try:
            stats = dev.memory_stats()
            if stats:
                val = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
                if val is not None and val > 0:
                    peaks.append(float(val) / (1024.0 * 1024.0))
        except Exception:
            pass
    if peaks:
        return max(peaks)
    # CPU fallback: Linux ru_maxrss is in KiB (macOS is bytes)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(rss) / 1024.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Data Download & Preparation
# ---------------------------------------------------------------------------

def _download_url(url: str, filepath: str, max_attempts: int = 5) -> bool:
    """Stream-download a single file with atomic rename and exponential backoff."""
    if os.path.exists(filepath):
        return True
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        tmp_path = filepath + ".tmp"
        try:
            with requests.get(url, stream=True, timeout=45) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, filepath)
            print(f"  Downloaded {os.path.basename(filepath)}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  Attempt {attempt}/{max_attempts} failed for {os.path.basename(filepath)}: {exc}")
            for p in (tmp_path, filepath):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 16))
    return False


def _download_climbmix_shard(index: int) -> bool:
    filename = f"shard_{index:05d}.parquet"
    return _download_url(f"{CLIMBMIX_BASE_URL}/{filename}", os.path.join(DATA_DIR, filename))


def download_climbmix(num_shards: int, download_workers: int = 8) -> None:
    """Download ClimbMix training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, CLIMBMIX_MAX_SHARD)
    ids = list(range(num_train))
    if CLIMBMIX_VAL_SHARD not in ids:
        ids.append(CLIMBMIX_VAL_SHARD)

    existing = sum(
        1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet"))
    )
    if existing == len(ids):
        print(f"Data: all {len(ids)} ClimbMix shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} ClimbMix shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    if workers == 1:
        results = [_download_climbmix_shard(i) for i in ids]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_download_climbmix_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")
    if ok < 2:
        raise RuntimeError("Failed to download at least 1 train shard and 1 val shard.")


def download_tinystories(num_shards: int = 2) -> None:
    """Download TinyStories parquet files and save into the standard shard layout."""
    os.makedirs(DATA_DIR, exist_ok=True)
    n = max(1, min(num_shards, len(TINYSTORIES_TRAIN_URLS)))
    tasks = [
        (TINYSTORIES_TRAIN_URLS[i], os.path.join(DATA_DIR, f"shard_{i:05d}.parquet"))
        for i in range(n)
    ]
    tasks.append((TINYSTORIES_VAL_URL, os.path.join(DATA_DIR, VAL_FILENAME)))
    print(f"Data: preparing TinyStories ({n} train + 1 val parquet)...")
    ok = sum(1 for url, path in tasks if _download_url(url, path))
    print(f"Data: {ok}/{len(tasks)} TinyStories files ready at {DATA_DIR}")
    if ok < 2:
        raise RuntimeError("Failed to download TinyStories dataset.")


def generate_synthetic_data(num_train_docs: int = 2500, num_val_docs: int = 400) -> None:
    """Generate deterministic local parquet shards for offline testing & CI."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "shard_00000.parquet")
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data: synthetic shards already exist at {DATA_DIR}")
        return

    rng = np.random.default_rng(42)
    topics = [
        "neural networks", "matrix multiplication", "gradient descent",
        "attention mechanisms", "TPU systolic arrays", "GPU tensor cores",
        "XLA compiler fusion", "byte pair encoding", "information theory",
        "sharded data parallelism", "cosine learning schedules", "residual streams",
    ]
    verbs = [
        "accelerates", "stabilizes", "optimizes", "compresses", "generalizes",
        "approximates", "regularizes", "transforms", "balances", "evaluates",
    ]
    adjectives = [
        "efficient", "scalable", "deterministic", "orthogonal", "adaptive",
        "compact", "numerical", "high-throughput", "low-latency", "expressive",
    ]

    def make_docs(count: int) -> list[str]:
        docs = []
        for idx in range(count):
            n_sentences = int(rng.integers(6, 18))
            sentences = []
            for s in range(n_sentences):
                t1 = topics[int(rng.integers(0, len(topics)))]
                t2 = topics[int(rng.integers(0, len(topics)))]
                v = verbs[int(rng.integers(0, len(verbs)))]
                a = adjectives[int(rng.integers(0, len(adjectives)))]
                num = int(rng.integers(1, 4096))
                sentences.append(
                    f"Experiment {idx}.{s}: A {a} design for {t1} {v} {t2} "
                    f"at scale {num} with measurable bits-per-byte gains."
                )
            docs.append(" ".join(sentences) + "\n")
        return docs

    pq.write_table(pa.table({"text": make_docs(num_train_docs)}), train_path)
    pq.write_table(pa.table({"text": make_docs(num_val_docs)}), val_path)
    print(f"Data: generated synthetic shards ({num_train_docs} train, {num_val_docs} val) at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Tokenizer Training & Runtime Wrapper
# ---------------------------------------------------------------------------

def list_parquet_files() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    files = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000) -> Iterator[str]:
    """Yield training documents (excluding the pinned validation shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                if not text:
                    continue
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer() -> None:
    """
    Train an 8192-vocab BPE tokenizer and compute exact UTF-8 token_bytes.npy.
    Prefers `rustbpe` + `tiktoken` (matching Karpathy's autoresearch exactly),
    with seamless fallback to HuggingFace `tokenizers` (preinstalled on Colab/Kaggle).
    """
    tiktoken_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    hf_json = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    if (os.path.exists(tiktoken_pkl) or os.path.exists(hf_json)) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    min_vocab = 256 + len(SPECIAL_TOKENS)
    if VOCAB_SIZE < min_vocab:
        raise ValueError(f"VOCAB_SIZE ({VOCAB_SIZE}) must be at least {min_vocab} (256 byte tokens + {len(SPECIAL_TOKENS)} special tokens).")
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 shards (1 train + 1 val). Run data download first.")
        sys.exit(1)

    t0 = time.time()
    backend_used = None

    try:
        import rustbpe
        import tiktoken

        print(f"Tokenizer: training BPE (vocab={VOCAB_SIZE}) via rustbpe + tiktoken...")
        bpe = rustbpe.Tokenizer()
        vocab_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
        bpe.train_from_iterator(text_iterator(), vocab_no_special, pattern=SPLIT_PATTERN)

        pattern = bpe.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
        offset = len(mergeable_ranks)
        special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        with open(tiktoken_pkl, "wb") as f:
            pickle.dump(enc, f)

        # Exact raw UTF-8 byte lengths (avoids U+FFFD inflation on partial UTF-8 bytes)
        special_ids = set(special_tokens.values())
        token_bytes = np.zeros(enc.n_vocab, dtype=np.int32)
        for tid in range(enc.n_vocab):
            if tid in special_ids:
                token_bytes[tid] = 0
            else:
                token_bytes[tid] = len(enc.decode_single_token_bytes(tid))
        np.save(token_bytes_path, token_bytes)
        backend_used = "tiktoken"
    except ImportError:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        print(f"Tokenizer: rustbpe not found; training ByteLevel BPE (vocab={VOCAB_SIZE}) via tokenizers...")
        hf_tok = HFTokenizer(models.BPE())
        hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tok.train_from_iterator(text_iterator(), trainer=trainer)
        hf_tok.save(hf_json)

        special_set = set(SPECIAL_TOKENS)
        vocab_size_actual = hf_tok.get_vocab_size()
        # ByteLevel alphabet maps each raw byte 0..255 to a unique unicode char
        byte_decoder = {ch: i for i, ch in enumerate(pre_tokenizers.ByteLevel.alphabet())}
        token_bytes = np.zeros(vocab_size_actual, dtype=np.int32)
        for tid in range(vocab_size_actual):
            tok_str = hf_tok.id_to_token(tid)
            if tok_str is None or tok_str in special_set:
                token_bytes[tid] = 0
            else:
                # Every char in a ByteLevel BPE token corresponds to exactly 1 original UTF-8 byte
                if all(ch in byte_decoder for ch in tok_str):
                    token_bytes[tid] = len(tok_str)
                else:
                    token_bytes[tid] = len(hf_tok.decode([tid]).encode("utf-8"))
        np.save(token_bytes_path, token_bytes)
        backend_used = "tokenizers"

    tok = Tokenizer.from_directory(TOKENIZER_DIR)
    sample = "Hello world! Numbers: 123. Unicode: 你好"
    roundtrip = tok.decode(tok.encode(sample))
    assert roundtrip == sample, f"Tokenizer roundtrip failed: {sample!r} -> {roundtrip!r}"
    print(
        f"Tokenizer: trained in {time.time() - t0:.1f}s ({backend_used}, "
        f"vocab_size={tok.get_vocab_size()}) -> {TOKENIZER_DIR}"
    )


class Tokenizer:
    """Unified minimal tokenizer wrapper supporting tiktoken (primary) and HF tokenizers."""

    def __init__(self, backend: str, enc, bos_token_id: int, vocab_size: int):
        self.backend = backend
        self.enc = enc
        self.bos_token_id = bos_token_id
        self._vocab_size = vocab_size

    @classmethod
    def from_directory(cls, tokenizer_dir: str = TOKENIZER_DIR) -> "Tokenizer":
        tiktoken_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
        hf_json = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(tiktoken_pkl):
            with open(tiktoken_pkl, "rb") as f:
                enc = pickle.load(f)
            return cls("tiktoken", enc, enc.encode_single_token(BOS_TOKEN), enc.n_vocab)
        if os.path.exists(hf_json):
            from tokenizers import Tokenizer as HFTokenizer

            enc = HFTokenizer.from_file(hf_json)
            bos_id = enc.token_to_id(BOS_TOKEN)
            return cls("tokenizers", enc, bos_id, enc.get_vocab_size())
        raise FileNotFoundError(
            f"No tokenizer found in {tokenizer_dir}. Run `python prepare.py` first."
        )

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend: Optional[int | str] = None, num_threads: int = 8):
        prepend_id = None
        if prepend is not None:
            if isinstance(prepend, int):
                prepend_id = prepend
            elif self.backend == "tiktoken":
                prepend_id = self.enc.encode_single_token(prepend)
            else:
                prepend_id = self.enc.token_to_id(prepend)

        if isinstance(text, str):
            if self.backend == "tiktoken":
                ids = self.enc.encode_ordinary(text)
            else:
                ids = self.enc.encode(text, add_special_tokens=False).ids
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids

        if isinstance(text, list):
            if self.backend == "tiktoken":
                batch_ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            else:
                batch_ids = [e.ids for e in self.enc.encode_batch(text, add_special_tokens=False)]
            if prepend_id is not None:
                for row in batch_ids:
                    row.insert(0, prepend_id)
            return batch_ids

        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def get_token_bytes(tokenizer_dir: str = TOKENIZER_DIR) -> np.ndarray:
    path = os.path.join(tokenizer_dir, "token_bytes.npy")
    return np.load(path).astype(np.int32)


# ---------------------------------------------------------------------------
# BOS-Aligned Best-Fit Packing Dataloader (Pure NumPy + Background Prefetch)
# ---------------------------------------------------------------------------

def _document_batches(split: str, tokenizer_batch_size: int = 128):
    """Infinite iterator over document text batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run `python prepare.py` first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
        assert os.path.exists(val_path), f"Validation shard missing: {val_path}"

    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                for i in range(0, len(texts), tokenizer_batch_size):
                    yield texts[i : i + tokenizer_batch_size], epoch
        epoch += 1


def _pack_batch_iterator(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    buffer_size: int = 1000,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """
    BOS-aligned best-fit document packer yielding (inputs, targets, epoch) as int32 NumPy arrays.
    Every row starts with BOS; documents are packed using best-fit with 100% token utilization.
    """
    assert split in ("train", "val")
    row_capacity = seq_len + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []
    epoch = 1

    def refill_buffer() -> None:
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    while True:
        row_buffer = np.empty((batch_size, row_capacity), dtype=np.int32)
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = np.ascontiguousarray(row_buffer[:, :-1])
        targets = np.ascontiguousarray(row_buffer[:, 1:])
        yield inputs, targets, epoch


def make_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    buffer_size: int = 1000,
    prefetch: int = 2,
    grad_accum_steps: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """
    Background-prefetched BOS-aligned dataloader.

    When `grad_accum_steps == 1`: yields `(x, y, epoch)` of shape `(batch_size, seq_len)`.
    When `grad_accum_steps > 1`:  yields `(x, y, epoch)` of shape `(grad_accum_steps, batch_size, seq_len)`
    so `train.py` can transfer a full optimizer step to GPU/TPU in one shot and run `lax.scan`.
    """
    base_iter = _pack_batch_iterator(tokenizer, batch_size, seq_len, split, buffer_size=buffer_size)

    def make_step_item():
        if grad_accum_steps == 1:
            return next(base_iter)
        xs = np.empty((grad_accum_steps, batch_size, seq_len), dtype=np.int32)
        ys = np.empty((grad_accum_steps, batch_size, seq_len), dtype=np.int32)
        ep = 1
        for k in range(grad_accum_steps):
            x, y, ep = next(base_iter)
            xs[k] = x
            ys[k] = y
        return xs, ys, ep

    if prefetch <= 0:
        while True:
            yield make_step_item()

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop_event = threading.Event()

    def worker() -> None:
        try:
            while not stop_event.is_set():
                item = make_step_item()
                while not stop_event.is_set():
                    try:
                        q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            q.put(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop_event.set()


# ---------------------------------------------------------------------------
# Fixed Evaluation Metric (DO NOT CHANGE — ground-truth yardstick)
# ---------------------------------------------------------------------------

def evaluate_bpb(
    eval_loss_fn: Callable,
    params,
    tokenizer: Tokenizer,
    global_batch_size: int,
    data_sharding=None,
) -> float:
    """
    Bits per byte (BPB): vocab-size-independent evaluation metric.

    `eval_loss_fn(params, x, y)` must return per-token cross-entropy losses in nats
    of shape `(global_batch_size, MAX_SEQ_LEN)` in float32.
    Special tokens (byte length 0) are excluded from both numerator and denominator.
    Uses fixed `MAX_SEQ_LEN` and `EVAL_TOKENS` so results are apples-to-apples comparable.
    """
    import jax
    import jax.numpy as jnp

    from jax.sharding import NamedSharding, PartitionSpec

    token_bytes_np = get_token_bytes()
    if data_sharding is not None:
        replicated = NamedSharding(data_sharding.mesh, PartitionSpec())
        token_bytes = jax.device_put(token_bytes_np, replicated)
    else:
        token_bytes = jnp.asarray(token_bytes_np, dtype=jnp.int32)

    val_loader = make_dataloader(
        tokenizer, global_batch_size, MAX_SEQ_LEN, "val", prefetch=2, grad_accum_steps=1
    )
    steps = max(1, EVAL_TOKENS // (global_batch_size * MAX_SEQ_LEN))

    @jax.jit
    def eval_batch_totals(p, x, y, tb):
        loss_2d = eval_loss_fn(p, x, y).astype(jnp.float32)
        nbytes = tb[y]
        mask = nbytes > 0
        nats = jnp.sum(jnp.where(mask, loss_2d, 0.0), dtype=jnp.float32)
        n_bytes = jnp.sum(nbytes, dtype=jnp.int32)
        return nats, n_bytes

    total_nats = 0.0
    total_bytes = 0
    try:
        for _ in range(steps):
            x_np, y_np, _ = next(val_loader)
            if data_sharding is not None:
                x = jax.device_put(x_np, data_sharding)
                y = jax.device_put(y_np, data_sharding)
            else:
                x = jnp.asarray(x_np)
                y = jnp.asarray(y_np)
            nats, n_bytes = eval_batch_totals(params, x, y, token_bytes)
            total_nats += float(nats)
            total_bytes += int(n_bytes)
    finally:
        val_loader.close()

    return total_nats / (math.log(2.0) * total_bytes)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dataset shards and BPE tokenizer for autoresearch-xla"
    )
    parser.add_argument(
        "--dataset",
        choices=["climbmix", "tinystories", "synthetic"],
        default="climbmix",
        help="Dataset source (default: climbmix — Karpathy's climbmix-400b-shuffle)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Number of training shards to download (-1 = all ClimbMix shards). Val shard is always pinned.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel download workers for ClimbMix shards",
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dataset:         {args.dataset}")
    print(f"Fixed constants: MAX_SEQ_LEN={MAX_SEQ_LEN}, TIME_BUDGET={TIME_BUDGET}s, EVAL_TOKENS={EVAL_TOKENS:,}, VOCAB_SIZE={VOCAB_SIZE}")
    print()

    if args.dataset == "climbmix":
        num_shards = CLIMBMIX_MAX_SHARD if args.num_shards == -1 else args.num_shards
        download_climbmix(num_shards, download_workers=args.download_workers)
    elif args.dataset == "tinystories":
        download_tinystories(num_shards=max(1, args.num_shards))
    elif args.dataset == "synthetic":
        generate_synthetic_data()

    print()
    train_tokenizer()
    print()
    print("Done! Ready to run `python train.py`.")


if __name__ == "__main__":
    main()
