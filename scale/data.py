"""
Disk-bounded streaming data pipeline and resumable BPE tokenizer for 1B+ pretraining.

Designed specifically for Kaggle (20 GB disk cap) and Colab:
- Streams ClimbMix shards (`karpathy/climbmix-400b-shuffle`, 6,542 shards ≈ 400B tokens)
  using a rolling scratch window (`keep_shards=2`, ~200 MB on disk at any time!).
- Prefetches the next shard in a background thread and deletes consumed shards immediately.
- Tracks an exact `DataCursor(shard_idx, row_group_idx, epoch)` saved inside every
  checkpoint so multi-session 9h Kaggle / 12h Colab runs resume without repeating tokens.
- Trains a 32K (or custom vocab) BPE tokenizer with chat & tool-calling special tokens
  baked in from step 0 so Phase 3 post-training never needs to resize embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import pickle
import queue
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_TRAIN_SHARDS = 6542  # shard_00000 .. shard_06541 are train
CLIMBMIX_VAL_SHARD = 6542         # shard_06542 is pinned validation

TINYSTORIES_TRAIN_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00002-of-00004-a26307300439e943.parquet",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00003-of-00004-d243063613e5a057.parquet",
]
TINYSTORIES_VAL_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
    "validation-00000-of-00001-869c898b519ad725.parquet"
)

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Baked-in special tokens from step 0 (future-proof for chat/tool-calling post-training)
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|tool_call|>",
    "<|tool_response|>",
    "<|reserved_7|>",
]
BOS_TOKEN = "<|bos|>"


@dataclass
class DataCursor:
    shard_idx: int = 0
    row_group_idx: int = 0
    epoch: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DataCursor":
        if not d:
            return cls()
        return cls(
            shard_idx=int(d.get("shard_idx", 0)),
            row_group_idx=int(d.get("row_group_idx", 0)),
            epoch=int(d.get("epoch", 1)),
        )


# ---------------------------------------------------------------------------
# Download Helper & Synthetic Corpus Generator
# ---------------------------------------------------------------------------

def _download_file(url: str, filepath: str, max_attempts: int = 5) -> str:
    if os.path.exists(filepath):
        return filepath
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        tmp_path = filepath + f".tmp.{os.getpid()}"
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, filepath)
            return filepath
        except (requests.RequestException, OSError) as exc:
            for p in (tmp_path, filepath):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt == max_attempts:
                raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
            time.sleep(min(2**attempt, 16))
    return filepath


def _ensure_synthetic_shards(cache_dir: str, num_train_shards: int = 3) -> tuple[list[str], str]:
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(1234)
    topics = [
        "sharded tensor parallelism", "systolic matrix units", "causal self attention",
        "weight decay schedules", "orthogonal gradient updates", "byte pair tokenization",
        "autoregressive sampling", "information compression", "transformer scaling laws",
    ]
    verbs = ["improves", "accelerates", "stabilizes", "compresses", "generalizes", "scales"]
    adjs = ["fast", "clean", "scalable", "deterministic", "compact", "resumable"]

    def build_docs(n_docs: int, shard_id: int) -> list[str]:
        docs = []
        for i in range(n_docs):
            n_sent = int(rng.integers(6, 16))
            sents = []
            for s in range(n_sent):
                t1 = topics[int(rng.integers(0, len(topics)))]
                t2 = topics[int(rng.integers(0, len(topics)))]
                v = verbs[int(rng.integers(0, len(verbs)))]
                a = adjs[int(rng.integers(0, len(adjs)))]
                sents.append(
                    f"Shard {shard_id} doc {i}.{s}: A {a} implementation of {t1} {v} {t2} "
                    f"across multi-chip TPU pods with value {int(rng.integers(10, 9999))}."
                )
            docs.append(" ".join(sents) + "\n")
        return docs

    train_paths = []
    for s in range(num_train_shards):
        p = os.path.join(cache_dir, f"synthetic_train_{s:03d}.parquet")
        if not os.path.exists(p):
            pq.write_table(pa.table({"text": build_docs(800, s)}), p)
        train_paths.append(p)

    val_path = os.path.join(cache_dir, "synthetic_val.parquet")
    if not os.path.exists(val_path):
        pq.write_table(pa.table({"text": build_docs(300, 999)}), val_path)
    return train_paths, val_path


# ---------------------------------------------------------------------------
# Scalable BPE Tokenizer (32K default, tiktoken primary + HF tokenizers fallback)
# ---------------------------------------------------------------------------

class ScaleTokenizer:
    """Minimal unified BPE tokenizer supporting tiktoken (primary) and HF tokenizers."""

    def __init__(self, backend: str, enc, bos_token_id: int, vocab_size: int):
        self.backend = backend
        self.enc = enc
        self.bos_token_id = bos_token_id
        self._vocab_size = vocab_size

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "ScaleTokenizer":
        pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                enc = pickle.load(f)
            return cls("tiktoken", enc, enc.encode_single_token(BOS_TOKEN), enc.n_vocab)
        if os.path.exists(json_path):
            from tokenizers import Tokenizer as HFTokenizer

            enc = HFTokenizer.from_file(json_path)
            return cls("tokenizers", enc, enc.token_to_id(BOS_TOKEN), enc.get_vocab_size())
        raise FileNotFoundError(f"No tokenizer found in {tokenizer_dir}")

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
            ids = (
                self.enc.encode_ordinary(text)
                if self.backend == "tiktoken"
                else self.enc.encode(text, add_special_tokens=False).ids
            )
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids

        if isinstance(text, list):
            batch_ids = (
                self.enc.encode_ordinary_batch(text, num_threads=num_threads)
                if self.backend == "tiktoken"
                else [e.ids for e in self.enc.encode_batch(text, add_special_tokens=False)]
            )
            if prepend_id is not None:
                for row in batch_ids:
                    row.insert(0, prepend_id)
            return batch_ids

        raise ValueError(f"Invalid text type: {type(text)}")

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def ensure_tokenizer(
    tokenizer_dir: str,
    vocab_size: int,
    dataset: str,
    shard_cache_dir: str,
    train_shards: int = 2,
) -> ScaleTokenizer:
    """Load existing tokenizer from `tokenizer_dir`, or train one cleanly on initial shards."""
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tb_path = os.path.join(tokenizer_dir, "token_bytes.npy")

    if (os.path.exists(pkl_path) or os.path.exists(json_path)) and os.path.exists(tb_path):
        tok = ScaleTokenizer.from_directory(tokenizer_dir)
        if tok.get_vocab_size() == vocab_size:
            return tok

    min_vocab = 256 + len(SPECIAL_TOKENS)
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size ({vocab_size}) must be >= {min_vocab}")

    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(shard_cache_dir, exist_ok=True)

    # Fetch 1-2 initial shards for tokenizer training
    if dataset == "synthetic":
        train_paths, _ = _ensure_synthetic_shards(shard_cache_dir)
    elif dataset == "tinystories":
        train_paths = [
            _download_file(TINYSTORIES_TRAIN_URLS[i], os.path.join(shard_cache_dir, f"ts_train_{i:02d}.parquet"))
            for i in range(min(train_shards, len(TINYSTORIES_TRAIN_URLS)))
        ]
    else:
        train_paths = [
            _download_file(
                f"{CLIMBMIX_BASE_URL}/shard_{i:05d}.parquet",
                os.path.join(shard_cache_dir, f"shard_{i:05d}.parquet"),
            )
            for i in range(train_shards)
        ]

    def text_iter(max_chars: int = 400_000_000, doc_cap: int = 10_000):
        nchars = 0
        for fp in train_paths:
            pf = pq.ParquetFile(fp)
            for rg_idx in range(pf.num_row_groups):
                for txt in pf.read_row_group(rg_idx).column("text").to_pylist():
                    if not txt:
                        continue
                    doc = txt[:doc_cap] if len(txt) > doc_cap else txt
                    nchars += len(doc)
                    yield doc
                    if nchars >= max_chars:
                        return

    t0 = time.time()
    try:
        import rustbpe
        import tiktoken

        print(f"[Tokenizer] Training {vocab_size:,}-vocab BPE via rustbpe + tiktoken...")
        bpe = rustbpe.Tokenizer()
        bpe.train_from_iterator(text_iter(), vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
        mergeable = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
        offset = len(mergeable)
        specials = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="scale_rustbpe",
            pat_str=bpe.get_pattern(),
            mergeable_ranks=mergeable,
            special_tokens=specials,
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(enc, f)

        special_ids = set(specials.values())
        tb = np.zeros(enc.n_vocab, dtype=np.int32)
        for tid in range(enc.n_vocab):
            tb[tid] = 0 if tid in special_ids else len(enc.decode_single_token_bytes(tid))
        np.save(tb_path, tb)
    except ImportError:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        print(f"[Tokenizer] Training {vocab_size:,}-vocab ByteLevel BPE via HF tokenizers...")
        hf_tok = HFTokenizer(models.BPE())
        hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tok.train_from_iterator(text_iter(), trainer=trainer)
        hf_tok.save(json_path)

        special_set = set(SPECIAL_TOKENS)
        alphabet = set(pre_tokenizers.ByteLevel.alphabet())
        actual_v = hf_tok.get_vocab_size()
        tb = np.zeros(actual_v, dtype=np.int32)
        for tid in range(actual_v):
            tok_str = hf_tok.id_to_token(tid)
            if tok_str is None or tok_str in special_set:
                tb[tid] = 0
            elif all(ch in alphabet for ch in tok_str):
                tb[tid] = len(tok_str)
            else:
                tb[tid] = len(hf_tok.decode([tid]).encode("utf-8"))
        np.save(tb_path, tb)

    tok = ScaleTokenizer.from_directory(tokenizer_dir)
    print(f"[Tokenizer] Ready in {time.time() - t0:.1f}s ({tok.backend}, vocab={tok.get_vocab_size():,})")
    return tok


def load_token_bytes(tokenizer_dir: str) -> np.ndarray:
    return np.load(os.path.join(tokenizer_dir, "token_bytes.npy")).astype(np.int32)


# ---------------------------------------------------------------------------
# Disk-Bounded Streaming Shard Loader with Exact Resumable DataCursor
# ---------------------------------------------------------------------------

class StreamingDataLoader:
    """
    Streams parquet shards with a bounded rolling disk footprint (`keep_shards`),
    BOS-aligned best-fit packing, background prefetching, and an exact resumable cursor.
    """

    def __init__(
        self,
        tokenizer: ScaleTokenizer,
        dataset: str,
        split: str,
        batch_size: int,
        seq_len: int,
        grad_accum_steps: int = 1,
        shard_cache_dir: str = "/tmp/autoresearch_shards",
        keep_shards: int = 2,
        cursor: Optional[DataCursor] = None,
        buffer_size: int = 1000,
        prefetch: int = 2,
    ):
        assert split in ("train", "val")
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.grad_accum_steps = grad_accum_steps
        self.shard_cache_dir = shard_cache_dir
        self.keep_shards = max(1, keep_shards)
        self.buffer_size = buffer_size
        self.prefetch = prefetch

        os.makedirs(self.shard_cache_dir, exist_ok=True)
        self._cursor = DataCursor.from_dict(cursor.to_dict() if cursor else None)
        self._stop = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._prefetch_threads: dict[int, threading.Thread] = {}
        self._synthetic_paths: Optional[tuple[list[str], str]] = None
        if dataset == "synthetic":
            self._synthetic_paths = _ensure_synthetic_shards(shard_cache_dir)

        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _num_train_shards(self) -> int:
        if self.dataset == "synthetic":
            assert self._synthetic_paths is not None
            return len(self._synthetic_paths[0])
        if self.dataset == "tinystories":
            return len(TINYSTORIES_TRAIN_URLS)
        return CLIMBMIX_MAX_TRAIN_SHARDS

    def _shard_path_and_url(self, shard_idx: int) -> tuple[str, Optional[str]]:
        if self.dataset == "synthetic":
            assert self._synthetic_paths is not None
            if self.split == "val":
                return self._synthetic_paths[1], None
            return self._synthetic_paths[0][shard_idx % len(self._synthetic_paths[0])], None
        if self.dataset == "tinystories":
            if self.split == "val":
                return os.path.join(self.shard_cache_dir, "ts_val.parquet"), TINYSTORIES_VAL_URL
            idx = shard_idx % len(TINYSTORIES_TRAIN_URLS)
            return os.path.join(self.shard_cache_dir, f"ts_train_{idx:02d}.parquet"), TINYSTORIES_TRAIN_URLS[idx]
        # ClimbMix
        if self.split == "val":
            fn = f"shard_{CLIMBMIX_VAL_SHARD:05d}.parquet"
            return os.path.join(self.shard_cache_dir, fn), f"{CLIMBMIX_BASE_URL}/{fn}"
        idx = shard_idx % CLIMBMIX_MAX_TRAIN_SHARDS
        fn = f"shard_{idx:05d}.parquet"
        return os.path.join(self.shard_cache_dir, fn), f"{CLIMBMIX_BASE_URL}/{fn}"

    def _ensure_shard(self, shard_idx: int) -> str:
        path, url = self._shard_path_and_url(shard_idx)
        if url is not None and not os.path.exists(path):
            _download_file(url, path)
        return path

    def _kick_prefetch_shard(self, shard_idx: int) -> None:
        if self.dataset == "synthetic" or self.split == "val":
            return
        path, url = self._shard_path_and_url(shard_idx)
        if url is None or os.path.exists(path) or shard_idx in self._prefetch_threads:
            return

        def dl():
            try:
                _download_file(url, path)
            except Exception:
                pass

        t = threading.Thread(target=dl, daemon=True)
        self._prefetch_threads[shard_idx] = t
        t.start()

    def _cleanup_old_shard(self, shard_idx: int) -> None:
        """Delete consumed ClimbMix training shards to stay strictly within Kaggle's disk cap."""
        if self.dataset != "climbmix" or self.split == "val":
            return
        if self.keep_shards <= 0:
            return
        old_idx = shard_idx - (self.keep_shards - 1)
        if old_idx < 0:
            return
        old_path, _ = self._shard_path_and_url(old_idx)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
        self._prefetch_threads.pop(old_idx, None)

    def _doc_batch_stream(self, doc_batch_size: int = 128):
        n_shards = self._num_train_shards() if self.split == "train" else 1
        shard_idx = self._cursor.shard_idx % n_shards
        start_rg = self._cursor.row_group_idx
        epoch = self._cursor.epoch

        while not self._stop.is_set():
            # Prefetch next shard while we consume the current one
            if self.split == "train" and n_shards > 1:
                self._kick_prefetch_shard((shard_idx + 1) % n_shards)

            fp = self._ensure_shard(shard_idx)
            pf = pq.ParquetFile(fp)
            for rg_idx in range(start_rg, pf.num_row_groups):
                if self._stop.is_set():
                    return
                texts = pf.read_row_group(rg_idx).column("text").to_pylist()
                cursor_snap = DataCursor(shard_idx=shard_idx, row_group_idx=rg_idx, epoch=epoch)
                for i in range(0, len(texts), doc_batch_size):
                    yield texts[i : i + doc_batch_size], cursor_snap

            start_rg = 0
            self._cleanup_old_shard(shard_idx)
            shard_idx += 1
            if shard_idx >= n_shards:
                shard_idx = 0
                epoch += 1

    def _run_worker(self) -> None:
        try:
            bos_id = self.tokenizer.get_bos_token_id()
            row_cap = self.seq_len + 1
            stream = self._doc_batch_stream()
            doc_buffer: list[list[int]] = []
            latest_cursor = DataCursor.from_dict(self._cursor.to_dict())

            def refill():
                nonlocal latest_cursor
                batch_texts, latest_cursor = next(stream)
                clean = [t for t in batch_texts if t]
                if clean:
                    doc_buffer.extend(self.tokenizer.encode(clean, prepend=bos_id))

            while not self._stop.is_set():
                xs = np.empty((self.grad_accum_steps, self.batch_size, self.seq_len), dtype=np.int32)
                ys = np.empty((self.grad_accum_steps, self.batch_size, self.seq_len), dtype=np.int32)
                for k in range(self.grad_accum_steps):
                    row_buf = np.empty((self.batch_size, row_cap), dtype=np.int32)
                    for r in range(self.batch_size):
                        pos = 0
                        while pos < row_cap:
                            while len(doc_buffer) < self.buffer_size:
                                refill()
                            rem = row_cap - pos
                            best_i, best_l = -1, 0
                            for idx_d, d in enumerate(doc_buffer):
                                dl = len(d)
                                if dl <= rem and dl > best_l:
                                    best_i, best_l = idx_d, dl
                            if best_i >= 0:
                                doc = doc_buffer.pop(best_i)
                                row_buf[r, pos : pos + len(doc)] = doc
                                pos += len(doc)
                            else:
                                short_i = min(range(len(doc_buffer)), key=lambda j: len(doc_buffer[j]))
                                doc = doc_buffer.pop(short_i)
                                row_buf[r, pos : pos + rem] = doc[:rem]
                                pos += rem
                    xs[k] = row_buf[:, :-1]
                    ys[k] = row_buf[:, 1:]

                item = (xs, ys, DataCursor.from_dict(latest_cursor.to_dict()))
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self._queue.put(exc)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, DataCursor]]:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, DataCursor]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        self._cursor = item[2]
        return item

    @property
    def cursor(self) -> DataCursor:
        return DataCursor.from_dict(self._cursor.to_dict())

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Exact Ground-Truth Validation Bits-Per-Byte Evaluator
# ---------------------------------------------------------------------------

def evaluate_val_bpb(
    eval_loss_fn: Callable,
    params: dict,
    tokenizer: ScaleTokenizer,
    tokenizer_dir: str,
    dataset: str,
    global_batch_size: int,
    seq_len: int,
    eval_tokens: int,
    shard_cache_dir: str,
    data_sharding=None,
) -> float:
    """Compute exact vocab-size-independent validation bits-per-byte on the pinned val shard."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    tb_np = load_token_bytes(tokenizer_dir)
    if data_sharding is not None:
        tb = jax.device_put(tb_np, NamedSharding(data_sharding.mesh, P()))
    else:
        tb = jnp.asarray(tb_np, dtype=jnp.int32)

    @jax.jit
    def batch_totals(p, x, y, token_bytes):
        nll = eval_loss_fn(p, x, y).astype(jnp.float32)
        nb = token_bytes[y]
        mask = nb > 0
        nats = jnp.sum(jnp.where(mask, nll, 0.0), dtype=jnp.float32)
        n_bytes = jnp.sum(nb, dtype=jnp.int32)
        return nats, n_bytes

    steps = max(1, eval_tokens // (global_batch_size * seq_len))
    loader = StreamingDataLoader(
        tokenizer=tokenizer,
        dataset=dataset,
        split="val",
        batch_size=global_batch_size,
        seq_len=seq_len,
        grad_accum_steps=1,
        shard_cache_dir=shard_cache_dir,
        prefetch=2,
    )
    total_nats = 0.0
    total_bytes = 0
    try:
        for _ in range(steps):
            xs, ys, _ = next(loader)
            x_np, y_np = xs[0], ys[0]
            if data_sharding is not None:
                x = jax.device_put(x_np, data_sharding)
                y = jax.device_put(y_np, data_sharding)
            else:
                x = jnp.asarray(x_np)
                y = jnp.asarray(y_np)
            nats, n_bytes = batch_totals(params, x, y, tb)
            total_nats += float(nats)
            total_bytes += int(n_bytes)
    finally:
        loader.close()

    return total_nats / (math.log(2.0) * max(total_bytes, 1))
