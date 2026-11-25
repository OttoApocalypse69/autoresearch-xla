"""
Model architecture & training configurations for scaling autoresearch-xla to 1B+.

Any winning architectural recipe discovered by the proxy `autoresearch-xla` loop
can be saved to JSON (`ModelSpec.save_json`) and scaled directly to 100M / 350M / 1B
via `--arch-config` without changing a line of the FSDP engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    """Architecture specification — all dimensions should be multiples of 128 for TPU MXU efficiency."""

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 20
    n_embd: int = 2048
    n_head: int = 16
    n_kv_head: int = 4               # 4:1 GQA (kv_dim = 512)
    mlp_hidden: int = 5632           # ~2.75x for SwiGLU (or 8192 = 4x for ReLU^2)
    activation: str = "swiglu"       # "swiglu" | "relu2" | "gelu"
    window_pattern: str = "L"        # "L" (full), "SL", or "SSSL"
    use_qk_norm: bool = True
    use_value_embeds: bool = False   # ResFormer value embeddings (alternating/periodic)
    use_resid_lambdas: bool = True   # Per-layer residual + x0 skip scalars (nanochat)
    tie_embeddings: bool = False     # Tie wte & lm_head weights
    rope_base: float = 10000.0
    softcap: float = 15.0            # 0.0 disables logit softcapping
    z_loss_weight: float = 1e-4      # Auxiliary router/logit stability loss (0.0 disables)
    use_remat: bool = True           # Legacy bool (overridden when remat_policy != "auto")
    remat_policy: str = "save_attn"  # "none" | "save_all_compute" | "save_attn" | "save_qkv_attn" | "dots_with_no_batch_dims" | "dots" | "full"
    attn_impl: str = "auto"          # "auto" (Splash on TPU, cuDNN on Ampere+ GPU, XLA fallback) | "splash" | "cudnn" | "xla"
    loss_chunk_size: int = 2048      # Chunked linear CE token count (0 disables; 2048 cuts logits HBM by ~8-16x!)
    splash_block_size: int = 1024    # TPU Pallas SplashAttention VMEM tile size (auto-clamped to seq_len)
    matmul_quant: str = "none"       # Experimental linear GEMM quant: "none" | "int8" | "int8-all" | "fp8"
    scan_unroll: int = 1             # jax.lax.scan layer unroll factor (1 or 2 for pipeline overlap)

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        assert self.activation in ("swiglu", "relu2", "gelu"), f"Unsupported activation: {self.activation!r}"
        assert all(c in "SL" for c in self.window_pattern.upper()), f"Invalid window_pattern: {self.window_pattern!r}"
        valid_remat = ("none", "save_all_compute", "save_attn", "save_qkv_attn", "dots_with_no_batch_dims", "dots", "full")
        assert self.remat_policy in valid_remat, f"Invalid remat_policy: {self.remat_policy!r}"
        assert self.attn_impl in ("auto", "splash", "cudnn", "xla"), f"Invalid attn_impl: {self.attn_impl!r}"
        assert self.matmul_quant in ("none", "int8", "int8-all", "fp8"), f"Invalid matmul_quant: {self.matmul_quant!r}"
        assert self.scan_unroll >= 1, f"scan_unroll must be >= 1, got {self.scan_unroll}"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSpec":
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Curated Model Presets (all hidden dims divisible by 128 for TPU v5e/v6e MXU)
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, ModelSpec] = {
    # Tiny ~0.8M model for instant CPU & multi-device unit tests
    "smoke": ModelSpec(
        sequence_len=128,
        vocab_size=512,
        n_layer=4,
        n_embd=128,
        n_head=2,
        n_kv_head=1,
        mlp_hidden=256,
        activation="swiglu",
        window_pattern="SL",
        use_value_embeds=False,
        remat_policy="save_attn",
        loss_chunk_size=128,
    ),
    # ~126M proxy model (fast validation on Colab T4 / TPU v5e-1)
    "100m": ModelSpec(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=12,
        n_embd=768,
        n_head=6,
        n_kv_head=2,
        mlp_hidden=2048,
        activation="swiglu",
        window_pattern="L",
        use_value_embeds=False,
        remat_policy="save_attn",
        loss_chunk_size=2048,
    ),
    # ~367M mid-scale model (great for 1-session Colab A100 / v6e-1 sanity runs)
    "350m": ModelSpec(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=16,
        n_embd=1280,
        n_head=10,
        n_kv_head=2,
        mlp_hidden=3584,
        activation="swiglu",
        window_pattern="L",
        use_value_embeds=False,
        remat_policy="save_attn",
        loss_chunk_size=2048,
    ),
    # ~1.036B flagship base model (optimal for Kaggle TPU v5e-8 FSDP & Colab G4/A100/H100/v6e-1)
    "1b": ModelSpec(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=20,
        n_embd=2048,
        n_head=16,
        n_kv_head=4,
        mlp_hidden=5632,
        activation="swiglu",
        window_pattern="L",
        use_value_embeds=False,
        remat_policy="save_attn",
        loss_chunk_size=2048,
    ),
    # ~1.015B ReLU^2 + sliding-window nanochat-style variant
    "1b-relu2": ModelSpec(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=20,
        n_embd=2048,
        n_head=16,
        n_kv_head=4,
        mlp_hidden=8192,
        activation="relu2",
        window_pattern="SL",
        use_value_embeds=False,
        remat_policy="save_attn",
        loss_chunk_size=2048,
    ),
    # ~1.37B total / 1.036B active-FLOPs autoresearch winner (ResFormer Value Embeddings + SSSL window: -0.0967 BPB on mix-1b!)
    # Recommended optimizer flags from proxy sweep: --matrix-lr 0.04 --embedding-lr 0.4 --weight-decay 0.05
    "1b-ve": ModelSpec(
        sequence_len=2048,
        vocab_size=32768,
        n_layer=20,
        n_embd=2048,
        n_head=16,
        n_kv_head=4,
        mlp_hidden=5632,
        activation="swiglu",
        window_pattern="SSSL",
        use_value_embeds=True,
        remat_policy="save_all_compute",
        loss_chunk_size=2048,
    ),
}


def get_model_preset(name: str, arch_overrides: Optional[dict] = None) -> ModelSpec:
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from {list(MODEL_PRESETS)}")
    base = MODEL_PRESETS[name].to_dict()
    if arch_overrides:
        # Preserve preset scale dimensions unless explicitly overridden in arch_overrides
        base.update(arch_overrides)
    return ModelSpec.from_dict(base)


@dataclass
class TrainConfig:
    """Resumable WSD pretraining & FSDP runtime configuration."""

    # Data (single source, preset like "mix-1b" / "mix-cooldown" / "mix-code", or inline "src:w,...")
    dataset: str = "climbmix"
    total_tokens: int = 20_000_000_000  # 20B tokens (Chinchilla-optimal for 1B)
    eval_tokens: int = 4 * 524_288    # ~2.1M pinned val tokens per eval
    shard_cache_dir: str = "/tmp/autoresearch_shards"
    keep_shards: int = 2              # Rolling shard cap (~200 MB disk on Kaggle!)

    # Batching & Precision
    total_batch_size: int = 524_288   # 512K tokens/step (256 seqs @ 2048)
    per_device_batch: int = 4         # Microbatch sequences per chip (4 x 8 chips = 32 seqs)
    compute_dtype: str = "auto"       # "auto" | "bfloat16" | "float16" | "float32"
    param_dtype: str = "float32"      # "float32" (standard) or "bfloat16" (pure BF16 + stochastic rounding!)
    opt_dtype: str = "float32"        # "float32" (standard) or "bfloat16" (saves 50% opt HBM)
    use_fsdp: bool = True             # Shard params + opt_state across all devices

    # Optimizer (Muon for 2D/3D matrices + AdamW for embeddings/scalars)
    matrix_lr: float = 0.02           # Scaled cleanly for 1B width
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    scalar_lr: float = 0.2
    weight_decay: float = 0.1
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    muon_beta2: float = 0.95
    muon_ns_steps: int = 5
    muon_vmap: bool = False           # False = low-HBM lax.scan; True = batched vmap (great on v5e-8 / G4!)
    grad_clip: float = 1.0

    # WSD (Warmup -> Stable -> Decay) schedule for seamless multi-session pretraining!
    # Run as many 9h Kaggle / 12h Colab sessions as you want in `phase="stable"`,
    # then switch to `phase="decay"` for the final ~10% cooldown tokens.
    wsd_phase: str = "stable"         # "stable" | "decay"
    warmup_steps: int = 500
    decay_steps: int = 4000           # Used when wsd_phase == "decay"
    decay_start_step: Optional[int] = None  # Auto-set on entering decay phase
    min_lr_frac: float = 0.0

    # Session Controls, Evaluation & Checkpointing
    seed: int = 42
    max_steps: Optional[int] = None   # Optional step cap for this session
    max_hours: Optional[float] = 8.5  # Graceful save-and-exit before Kaggle 9h / Colab limit
    log_every: int = 10
    eval_every: int = 250
    sample_every: int = 500
    save_every: int = 500
    keep_last_checkpoints: int = 2

    # Persistence / Remote Sync
    checkpoint_dir: str = "./checkpoints/1b"
    sync_backend: str = "none"        # "none" | "local" | "hf" | "kaggle"
    remote_target: Optional[str] = None  # Drive path, HF repo_id, or Kaggle dataset_slug
    resume: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})
