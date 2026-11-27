"""
High-MFU resumable multi-chip FSDP pretraining engine for scaling autoresearch-xla to 1B+.

MFU engineering stack:
  - Official TPU XLA async collective fusion flags (`LIBTPU_INIT_ARGS`, set pre-JAX import)
  - Fused QKV & SwiGLU Gate/Up projections (128-multiple MXU/Tensor Core aligned)
  - Pallas TPU SplashAttention (fused backward kernel + shard_map FSDP) & cuDNN FlashAttention
  - Chunked linear cross-entropy (`--loss-chunk-size 2048`) slashing logits HBM by ~16x
  - Selective rematerialization (`--remat-policy none | dots_with_no_batch_dims | dots | full`)
  - True asynchronous step pipelining (zero host-device sync bubbles between steps!)
  - WSD (Warmup -> Stable -> Decay) schedule + disk-bounded streaming + HF/Drive/Kaggle sync

Examples:
    # 1B pretraining on Kaggle TPU v5e-8 with HuggingFace Hub auto-resume:
    python -m scale.pretrain --preset 1b --dataset climbmix \\
        --sync-backend hf --remote-target <user>/my-1b-base --max-hours 8.5

    # 1B MFU benchmark on Colab RTX PRO 6000 (G4) or TPU v5e-1 / v6e-1:
    python -m scale.pretrain --preset 1b --benchmark-only --remat-policy none --per-device-batch 8
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Set official XLA TPU / GPU performance flags BEFORE importing JAX!
# ---------------------------------------------------------------------------

_DEFAULT_TPU_XLA_FLAGS = (
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_ag_backward_pipelining=true "
    "--xla_tpu_enable_windowed_einsum_for_all_gather=true "
    "--xla_tpu_enable_windowed_einsum_for_reduce_scatter=true"
)
if not os.environ.get("LIBTPU_INIT_ARGS"):
    os.environ["LIBTPU_INIT_ARGS"] = _DEFAULT_TPU_XLA_FLAGS
if os.path.exists("/dev/nvidia0") and not os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"):
    # JAX defaults to 75% GPU VRAM; unlock 90% so larger microbatches & 1B models fit comfortably
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import functools
from collections import deque
import gc
import json
import math
from pathlib import Path
import signal
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from prepare import JAX_CACHE_DIR, detect_hardware, get_peak_memory_mb
from scale.attention import resolve_attn_backend
from scale.checkpoint import CheckpointManager
from scale.config import MODEL_PRESETS, ModelSpec, TrainConfig, get_model_preset
from scale.data import (
    DataCursor,
    StreamingDataLoader,
    ensure_tokenizer,
    evaluate_val_bpb,
    parse_dataset_mix,
)
from scale.fsdp import build_tree_shardings, create_fsdp_mesh, summarize_sharding
from scale.model import (
    count_params,
    estimate_flops_per_token,
    forward_loss_2d,
    forward_scalar_loss_and_z,
    generate_tokens,
    init_params,
    precompute_rotary_embeddings,
)
from scale.optimizer import (
    compute_muon_momentum,
    compute_weight_decay,
    compute_wsd_lrm,
    init_opt_state,
    optimizer_step,
)
from scale.quant import prequantize_model_weights

os.makedirs(JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

SAMPLE_PROMPT = "The key insight behind scaling neural networks is"


class TrainingInstabilityError(RuntimeError):
    """Raised when TrainingWatchdog detects divergence/regression to abort early and preserve TPU quota."""


class TrainingWatchdog:
    """
    Zero-overhead host-side instability detector & HF checkpoint protector:
      1. Hard blowup: NaN/Inf in loss, z_loss, or grad_norm; ce > 20 or z > 1000 after step 50.
      2. Pinned val_bpb regression: immediate abort if val_bpb jumps > +0.12 above best_val_bpb
         (or > +0.06 for 2 consecutive evals once best_val_bpb < 3.0).
      3. Sustained grad_norm explosion + rising train loss EMA: aborts if median grad_norm over
         5 consecutive log windows exceeds max(1.5, 4.0 * baseline_gnorm) AND loss EMA is > +0.30
         nats above its best EMA (immune to single noisy batches / shard boundaries!).
      4. Sustained train loss jump: aborts if loss EMA stays > +0.85 nats above best EMA for 5
         consecutive log windows.
      5. Checkpoint protection (`is_degraded`): prevents overwriting `latest/` on HF Hub with
         compromised weights when val_bpb or train loss has regressed.
    """

    def __init__(self, best_val_bpb: float = float("inf"), enabled: bool = True):
        self.enabled = enabled
        self.tripped = False
        self.best_val_bpb = best_val_bpb
        self.consec_val_regress = 0
        self.last_val_regressed = False
        self.loss_ema: Optional[float] = None
        self.best_loss_ema = float("inf")
        self.baseline_gnorm_ema: Optional[float] = None
        self.recent_gnorms: deque[float] = deque(maxlen=5)
        self.consec_loss_high = 0

    @property
    def is_degraded(self) -> bool:
        if not self.enabled:
            return False
        if self.tripped or self.last_val_regressed:
            return True
        if self.loss_ema is not None and math.isfinite(self.best_loss_ema):
            if self.loss_ema > self.best_loss_ema + 0.30:
                return True
        return False

    def _trip(self, reason: str) -> str:
        self.tripped = True
        return reason

    def check_step(self, step: int, ce_f: float, z_f: float, gnorm_f: float) -> Optional[str]:
        if not self.enabled:
            return None
        if not (math.isfinite(ce_f) and math.isfinite(z_f) and math.isfinite(gnorm_f)):
            return self._trip(f"Non-finite metric at step {step} (loss={ce_f}, z={z_f}, gnorm={gnorm_f})")
        if step >= 50 and ce_f > 20.0:
            return self._trip(f"Train loss blowup at step {step} (loss={ce_f:.3f} > 20.0)")
        if step >= 50 and z_f > 1000.0:
            return self._trip(f"Logit z-loss blowup at step {step} (z={z_f:.1f} > 1000.0)")

        self.loss_ema = ce_f if self.loss_ema is None else (0.2 * ce_f + 0.8 * self.loss_ema)
        if step >= 50 and self.loss_ema < self.best_loss_ema:
            self.best_loss_ema = self.loss_ema

        # Track calm baseline gnorm only while training is healthy
        if self.baseline_gnorm_ema is None:
            self.baseline_gnorm_ema = gnorm_f
        elif not math.isfinite(self.best_loss_ema) or self.loss_ema <= self.best_loss_ema + 0.15:
            self.baseline_gnorm_ema = 0.05 * gnorm_f + 0.95 * self.baseline_gnorm_ema

        self.recent_gnorms.append(gnorm_f)

        if step >= 200 and math.isfinite(self.best_loss_ema):
            if len(self.recent_gnorms) == 5:
                med_gn = sorted(self.recent_gnorms)[2]
                thresh = max(1.5, 4.0 * max(self.baseline_gnorm_ema, 0.1))
                if med_gn > thresh and self.loss_ema > self.best_loss_ema + 0.30:
                    return self._trip(
                        f"Sustained grad_norm explosion + loss regression at step {step}: "
                        f"median_gnorm={med_gn:.2f} > {thresh:.2f} (base={self.baseline_gnorm_ema:.2f}), "
                        f"loss_ema={self.loss_ema:.3f} vs best={self.best_loss_ema:.3f}"
                    )
            if self.loss_ema > self.best_loss_ema + 0.85:
                self.consec_loss_high += 1
                if self.consec_loss_high >= 5:
                    return self._trip(
                        f"Sustained train loss jump at step {step}: "
                        f"loss_ema={self.loss_ema:.3f} > best={self.best_loss_ema:.3f} + 0.85 for 5 windows"
                    )
            else:
                self.consec_loss_high = 0
        return None

    def check_eval(self, step: int, val_bpb: float) -> Optional[str]:
        if not self.enabled:
            if val_bpb < self.best_val_bpb:
                self.best_val_bpb = val_bpb
            return None
        if not math.isfinite(val_bpb):
            self.last_val_regressed = True
            return self._trip(f"Non-finite val_bpb={val_bpb} at step {step}")

        reason: Optional[str] = None
        if math.isfinite(self.best_val_bpb) and self.best_val_bpb < 3.0:
            delta = val_bpb - self.best_val_bpb
            self.last_val_regressed = delta > 0.05
            if delta > 0.12:
                reason = self._trip(
                    f"Severe val_bpb regression at step {step}: "
                    f"val_bpb={val_bpb:.6f} vs best={self.best_val_bpb:.6f} (+{delta:.4f} > +0.12)"
                )
            elif delta > 0.06:
                self.consec_val_regress += 1
                if self.consec_val_regress >= 2:
                    reason = self._trip(
                        f"Consecutive val_bpb regression at step {step}: "
                        f"val_bpb={val_bpb:.6f} vs best={self.best_val_bpb:.6f} (+{delta:.4f} for 2 evals)"
                    )
            elif delta <= 0.03:
                self.consec_val_regress = 0
        else:
            self.last_val_regressed = False

        if val_bpb < self.best_val_bpb:
            self.best_val_bpb = val_bpb
            self.consec_val_regress = 0
            self.last_val_regressed = False
        return reason


def _resolve_compute_dtype(tcfg: TrainConfig, hw_dtype_name: str):
    name = hw_dtype_name if tcfg.compute_dtype == "auto" else tcfg.compute_dtype
    mapping = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported compute_dtype: {name!r}")
    return name, mapping[name]


def _append_metric(metrics_path: Path, row: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _build_train_step(
    spec: ModelSpec,
    tcfg: TrainConfig,
    cos: jax.Array,
    sin: jax.Array,
    compute_dtype,
    ns_dtype,
    use_loss_scale: bool,
    grad_accum_steps: int,
    mesh,
    param_shardings,
    opt_shardings,
    step_sharding,
    replicated,
):
    def micro_loss_fn(p, x, y, scale, q_cache):
        total_scaled, ce, z_loss = forward_scalar_loss_and_z(
            p, x, y, cos, sin, spec, compute_dtype, scale=scale, mesh=mesh, q_cache=q_cache
        )
        return total_scaled, (ce, z_loss)

    grad_fn = jax.value_and_grad(micro_loss_fn, argnums=0, has_aux=True)

    @functools.partial(
        jax.jit,
        donate_argnums=(0, 1),
        in_shardings=(
            param_shardings,
            opt_shardings,
            step_sharding,
            step_sharding,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_shardings=(param_shardings, opt_shardings, replicated, replicated, replicated, replicated),
    )
    def train_step(p, st, x_steps, y_steps, lrm, muon_mom, muon_wd, loss_scale):
        # Megatron/MaxText mixed precision: if master weights are fp32, cast to bf16 ONCE before
        # micro-step accumulation (2x smaller FSDP AllGather/ReduceScatter & micro-grads).
        # If master weights are already bf16 (--param-dtype bfloat16 + SR), zero copy & zero cast!
        use_bf16_w = compute_dtype == jnp.bfloat16
        if use_bf16_w and tcfg.param_dtype != "bfloat16":
            p_work = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), p)
        else:
            p_work = p
        acc_dtype = jnp.bfloat16 if (tcfg.opt_dtype == "bfloat16" and use_bf16_w) else jnp.float32

        # Pre-quantize all linear weights ONCE per optimizer step before grad_accum scan:
        # eliminates 2*grad_accum redundant weight quantizations & halves weight HBM / FSDP AllGather!
        q_cache = prequantize_model_weights(p_work, spec.matmul_quant, tie_embeddings=spec.tie_embeddings)

        if grad_accum_steps == 1:
            (_, (mean_ce, mean_z)), g_raw = grad_fn(p_work, x_steps[0], y_steps[0], loss_scale, q_cache)
            grads = jax.tree_util.tree_map(lambda g: g / loss_scale, g_raw) if use_loss_scale else g_raw
        else:
            zero_grads = jax.tree_util.tree_map(lambda a: jnp.zeros_like(a, dtype=acc_dtype), p)

            def accum_body(carry, batch):
                g_acc, ce_acc, z_acc = carry
                xb, yb = batch
                (_, (ce_val, z_val)), g_micro = grad_fn(p_work, xb, yb, loss_scale, q_cache)
                g_acc = jax.tree_util.tree_map(lambda a, b: a + b.astype(acc_dtype), g_acc, g_micro)
                return (g_acc, ce_acc + ce_val, z_acc + z_val), None

            (g_sum, ce_sum, z_sum), _ = jax.lax.scan(
                accum_body,
                (zero_grads, jnp.float32(0.0), jnp.float32(0.0)),
                (x_steps, y_steps),
            )
            inv_accum = jnp.float32(1.0 / grad_accum_steps)
            inv_scale = inv_accum / loss_scale
            grads = jax.tree_util.tree_map(lambda g: (g.astype(jnp.float32) * inv_scale).astype(acc_dtype), g_sum)
            mean_ce = ce_sum * inv_accum
            mean_z = z_sum * inv_accum

        if use_loss_scale:
            leaves = jax.tree_util.tree_leaves(grads)
            all_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in leaves]))
            p_cand, st_cand, gnorm = optimizer_step(
                p, grads, st, lrm, muon_mom, muon_wd, spec, tcfg, ns_dtype, mesh=mesh
            )
            p_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), p_cand, p)
            st_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), st_cand, st)
            return p_next, st_next, mean_ce, mean_z, gnorm, all_finite

        p_next, st_next, gnorm = optimizer_step(
            p, grads, st, lrm, muon_mom, muon_wd, spec, tcfg, ns_dtype, mesh=mesh
        )
        return p_next, st_next, mean_ce, mean_z, gnorm, jnp.bool_(True)

    return train_step


def run_benchmark(spec: ModelSpec, tcfg: TrainConfig, warmup_steps: int = 2, timed_steps: int = 8) -> dict:
    """
    Pure MFU microbenchmark on real GPU/TPU with zero data-loader or disk overhead.
    Measures true compile time, steady-state async tok/s, MFU%, and peak HBM/VRAM.
    """
    hw = detect_hardware()
    dtype_name, compute_dtype = _resolve_compute_dtype(tcfg, hw.compute_dtype)
    param_dtype = jnp.bfloat16 if tcfg.param_dtype == "bfloat16" else jnp.float32
    opt_dtype = jnp.bfloat16 if tcfg.opt_dtype == "bfloat16" else jnp.float32
    use_loss_scale = compute_dtype == jnp.float16
    ns_dtype = jnp.bfloat16 if compute_dtype == jnp.bfloat16 else jnp.float32
    mesh = create_fsdp_mesh()
    attn_used = resolve_attn_backend(spec.attn_impl, mesh=mesh)
    replicated = NamedSharding(mesh, P())
    step_sharding = NamedSharding(mesh, P(None, "fsdp", None))

    global_micro_batch = tcfg.per_device_batch * hw.num_devices
    tokens_per_micro = global_micro_batch * spec.sequence_len
    total_batch_size = max(tcfg.total_batch_size, tokens_per_micro)
    if total_batch_size % tokens_per_micro != 0:
        total_batch_size = ((total_batch_size + tokens_per_micro - 1) // tokens_per_micro) * tokens_per_micro
    grad_accum_steps = total_batch_size // tokens_per_micro

    key = jax.random.PRNGKey(tcfg.seed)
    init_p_fn = functools.partial(init_params, spec=spec, param_dtype=param_dtype)
    abs_p = jax.eval_shape(init_p_fn, key)
    abs_o = jax.eval_shape(
        functools.partial(init_opt_state, opt_dtype=opt_dtype, gate_adamw=tcfg.gate_adamw, split_muon=tcfg.split_muon, spec=spec),
        abs_p,
    )
    param_shardings = build_tree_shardings(abs_p, mesh, use_fsdp=tcfg.use_fsdp)
    opt_shardings = build_tree_shardings(abs_o, mesh, use_fsdp=tcfg.use_fsdp)

    params = jax.jit(init_p_fn, out_shardings=param_shardings)(key)
    opt_state = jax.jit(
        functools.partial(init_opt_state, opt_dtype=opt_dtype, gate_adamw=tcfg.gate_adamw, split_muon=tcfg.split_muon, spec=spec),
        in_shardings=(param_shardings,),
        out_shardings=opt_shardings,
    )(params)
    cos, sin = precompute_rotary_embeddings(spec.sequence_len, spec.head_dim, base=spec.rope_base)
    cos = jax.device_put(cos, replicated)
    sin = jax.device_put(sin, replicated)

    p_counts = count_params(params)
    p_sh = summarize_sharding(params, mesh)
    o_sh = summarize_sharding(opt_state, mesh)
    flops_per_tok = estimate_flops_per_token(params, spec)
    peak_flops = hw.peak_tflops_total * 1e12

    muon_mode = "layer-sharded" if (tcfg.muon_layer_sharded and mesh.size > 1) else ("vmap" if tcfg.muon_vmap else "scan")
    print(
        f"[Bench] {hw.device_kind} x{hw.num_devices} ({hw.platform}, {dtype_name}, peak={hw.peak_tflops_total:.1f} TFLOPS) | "
        f"attn={attn_used} | remat={spec.remat_policy} | quant={spec.matmul_quant} | pallas={spec.use_pallas_kernels} | "
        f"unroll={spec.scan_unroll} | p={tcfg.param_dtype} | opt={tcfg.opt_dtype} | muon={muon_mode}"
    )
    print(
        f"[Bench] {p_counts['total']/1e6:.2f}M params | FSDP: {p_sh['per_device_mb']:.0f}MB p + {o_sh['per_device_mb']:.0f}MB opt/dev | "
        f"batch={total_batch_size:,} ({tcfg.per_device_batch}x{hw.num_devices}x{grad_accum_steps})"
    )

    train_step = _build_train_step(
        spec, tcfg, cos, sin, compute_dtype, ns_dtype, use_loss_scale,
        grad_accum_steps, mesh, param_shardings, opt_shardings, step_sharding, replicated,
    )

    # Pre-stage 2 random device batches to measure pure accelerator steady-state MFU
    rng = np.random.default_rng(0)
    x_dev = jax.device_put(rng.integers(0, spec.vocab_size, size=(grad_accum_steps, global_micro_batch, spec.sequence_len), dtype=np.int32), step_sharding)
    y_dev = jax.device_put(rng.integers(0, spec.vocab_size, size=(grad_accum_steps, global_micro_batch, spec.sequence_len), dtype=np.int32), step_sharding)
    lrm_d = jax.device_put(jnp.float32(1.0), replicated)
    mom_d = jax.device_put(jnp.float32(0.95), replicated)
    wd_d = jax.device_put(jnp.float32(tcfg.weight_decay), replicated)
    sc_d = jax.device_put(jnp.float32(32768.0 if use_loss_scale else 1.0), replicated)

    t_comp0 = time.perf_counter()
    for _ in range(warmup_steps):
        params, opt_state, ce_arr, _, _, _ = train_step(params, opt_state, x_dev, y_dev, lrm_d, mom_d, wd_d, sc_d)
        jax.block_until_ready(ce_arr)
    compile_s = time.perf_counter() - t_comp0

    # Optional real-silicon XProf hardware trace capture (after compile warmup!)
    if tcfg.profile_dir:
        os.makedirs(tcfg.profile_dir, exist_ok=True)
        print(f"[Profile] Capturing {tcfg.profile_steps} benchmark step(s) to {tcfg.profile_dir} ...", flush=True)
        with jax.profiler.trace(tcfg.profile_dir):
            for _ in range(max(1, tcfg.profile_steps)):
                params, opt_state, ce_arr, _, gnorm_arr, _ = train_step(
                    params, opt_state, x_dev, y_dev, lrm_d, mom_d, wd_d, sc_d
                )
            jax.block_until_ready(ce_arr)
        print(f"[Profile] XProf trace saved to {tcfg.profile_dir}", flush=True)

    # True async pipelined timed window!
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        params, opt_state, ce_arr, _, gnorm_arr, _ = train_step(params, opt_state, x_dev, y_dev, lrm_d, mom_d, wd_d, sc_d)
    last_loss = float(jax.block_until_ready(ce_arr))
    elapsed = time.perf_counter() - t0

    dt = elapsed / timed_steps
    tok_s = total_batch_size / dt
    mfu = 100.0 * flops_per_tok * tok_s / max(peak_flops, 1.0)
    peak_mb = get_peak_memory_mb()
    print(
        f"[Bench RESULT] compile={compile_s:.1f}s | step_dt={dt*1000:.1f}ms | "
        f"tok/s={tok_s:,.0f} | MFU={mfu:.2f}% | peak_mem={peak_mb:.1f}MB | loss={last_loss:.4f}",
        flush=True,
    )
    return {
        "compile_s": round(compile_s, 2),
        "step_ms": round(dt * 1000.0, 2),
        "tok_per_sec": round(tok_s, 1),
        "mfu_percent": round(mfu, 2),
        "peak_mem_mb": round(peak_mb, 1),
        "attn_backend": attn_used,
        "remat_policy": spec.remat_policy,
    }


def run_pretrain(spec: ModelSpec, tcfg: TrainConfig) -> dict:
    """Execute a resumable high-MFU FSDP pretraining session and return final summary metrics."""
    t_session_start = time.time()
    hw = detect_hardware()
    dtype_name, compute_dtype = _resolve_compute_dtype(tcfg, hw.compute_dtype)
    param_dtype = jnp.bfloat16 if tcfg.param_dtype == "bfloat16" else jnp.float32
    opt_dtype = jnp.bfloat16 if tcfg.opt_dtype == "bfloat16" else jnp.float32
    use_loss_scale = compute_dtype == jnp.float16
    ns_dtype = jnp.bfloat16 if compute_dtype == jnp.bfloat16 else jnp.float32
    mix_pairs = parse_dataset_mix(tcfg.dataset)
    mix_desc = ", ".join(f"{src}:{w*100:.0f}%" for src, w in mix_pairs)

    mesh = create_fsdp_mesh()
    attn_used = resolve_attn_backend(spec.attn_impl, mesh=mesh)
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("fsdp", None))
    step_sharding = NamedSharding(mesh, P(None, "fsdp", None))

    global_micro_batch = tcfg.per_device_batch * hw.num_devices
    tokens_per_micro = global_micro_batch * spec.sequence_len
    total_batch_size = max(tcfg.total_batch_size, tokens_per_micro)
    if total_batch_size % tokens_per_micro != 0:
        total_batch_size = ((total_batch_size + tokens_per_micro - 1) // tokens_per_micro) * tokens_per_micro
    grad_accum_steps = total_batch_size // tokens_per_micro

    ckpt_dir = Path(tcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        keep_last=tcfg.keep_last_checkpoints,
        sync_backend=tcfg.sync_backend,
        remote_target=tcfg.remote_target,
    )

    muon_mode = "layer-sharded" if (tcfg.muon_layer_sharded and mesh.size > 1) else ("vmap" if tcfg.muon_vmap else "scan")
    print(
        f"[Scale] Hardware: {hw.device_kind} x{hw.num_devices} "
        f"(platform={hw.platform}, compute={dtype_name}, param={tcfg.param_dtype}, opt={tcfg.opt_dtype}, "
        f"attn={attn_used}, remat={spec.remat_policy}, quant={spec.matmul_quant}, pallas={spec.use_pallas_kernels}, "
        f"muon={muon_mode}, fsdp={tcfg.use_fsdp}, peak={hw.peak_tflops_total:.1f} TFLOPS)"
    )
    print(f"[Scale] Dataset:  {tcfg.dataset} ({mix_desc})")

    if tcfg.resume and tcfg.sync_backend != "none" and tcfg.remote_target:
        try:
            ckpt_mgr.sync_pull()
        except Exception as exc:
            print(f"[Scale] No remote checkpoint found yet ({exc}); starting fresh.")

    tokenizer_dir = str(ckpt_dir / "tokenizer")
    tokenizer = ensure_tokenizer(
        tokenizer_dir=tokenizer_dir,
        vocab_size=spec.vocab_size,
        dataset=tcfg.dataset,
        shard_cache_dir=tcfg.shard_cache_dir,
    )
    assert tokenizer.get_vocab_size() == spec.vocab_size, (
        f"Tokenizer vocab ({tokenizer.get_vocab_size()}) != spec.vocab_size ({spec.vocab_size})"
    )

    key = jax.random.PRNGKey(tcfg.seed)
    init_p_fn = functools.partial(init_params, spec=spec, param_dtype=param_dtype)
    abstract_params = jax.eval_shape(init_p_fn, key)
    abstract_opt = jax.eval_shape(
        functools.partial(init_opt_state, opt_dtype=opt_dtype, gate_adamw=tcfg.gate_adamw, split_muon=tcfg.split_muon, spec=spec),
        abstract_params,
    )
    param_shardings = build_tree_shardings(abstract_params, mesh, use_fsdp=tcfg.use_fsdp)
    opt_shardings = build_tree_shardings(abstract_opt, mesh, use_fsdp=tcfg.use_fsdp)

    step = 0
    tokens_seen = 0
    best_val_bpb = float("inf")
    cursor = DataCursor()

    restored = None
    if tcfg.resume:
        restored = ckpt_mgr.load_latest(
            params_template=abstract_params,
            opt_template=abstract_opt,
            param_shardings=param_shardings,
            opt_shardings=opt_shardings,
            pull_remote=False,
        )

    if restored is not None:
        params, opt_state, saved_state = restored
        step = int(saved_state.get("step", 0))
        tokens_seen = int(saved_state.get("tokens_seen", step * total_batch_size))
        best_val_bpb = float(saved_state.get("best_val_bpb", float("inf")))
        cursor = DataCursor.from_dict(saved_state.get("data_cursor"))
        if tcfg.wsd_phase == "decay" and tcfg.decay_start_step is None:
            saved_decay_start = saved_state.get("decay_start_step")
            tcfg.decay_start_step = int(saved_decay_start) if saved_decay_start is not None else step
        print(
            f"[Scale] Resumed at step={step:,} | tokens={tokens_seen/1e9:.3f}B | "
            f"best_val_bpb={best_val_bpb:.4f} | cursor=shard {cursor.shard_idx} rg {cursor.row_group_idx} ep {cursor.epoch}"
        )
    else:
        params = jax.jit(init_p_fn, out_shardings=param_shardings)(key)
        opt_state = jax.jit(
            functools.partial(init_opt_state, opt_dtype=opt_dtype, gate_adamw=tcfg.gate_adamw, split_muon=tcfg.split_muon, spec=spec),
            in_shardings=(param_shardings,),
            out_shardings=opt_shardings,
        )(params)
        if tcfg.wsd_phase == "decay" and tcfg.decay_start_step is None:
            tcfg.decay_start_step = tcfg.warmup_steps

    cos, sin = precompute_rotary_embeddings(spec.sequence_len, spec.head_dim, base=spec.rope_base)
    cos = jax.device_put(cos, replicated)
    sin = jax.device_put(sin, replicated)

    p_counts = count_params(params)
    p_shard_info = summarize_sharding(params, mesh)
    o_shard_info = summarize_sharding(opt_state, mesh)
    flops_per_tok = estimate_flops_per_token(params, spec)
    peak_flops = hw.peak_tflops_total * 1e12

    print(
        f"[Scale] Model: {p_counts['total']/1e6:.2f}M params "
        f"({spec.n_layer}L x {spec.n_embd}D, {spec.activation}, GQA {spec.n_head}:{spec.n_kv_head}, "
        f"win={spec.window_pattern}, ce_chunk={spec.loss_chunk_size})"
    )
    print(f"[Scale] FSDP params: {p_shard_info['total_mb']:.1f} MB total -> {p_shard_info['per_device_mb']:.1f} MB/dev ({p_shard_info['sharded_leaves']} sharded leaves)")
    print(f"[Scale] FSDP opt:    {o_shard_info['total_mb']:.1f} MB total -> {o_shard_info['per_device_mb']:.1f} MB/dev ({o_shard_info['sharded_leaves']} sharded leaves)")
    print(f"[Scale] Batching:    {total_batch_size:,} tok/step ({tcfg.per_device_batch} seq/dev x {hw.num_devices} dev x {grad_accum_steps} accum) | FLOPs/tok={flops_per_tok:.3e}")
    print(f"[Scale] WSD phase:   {tcfg.wsd_phase} (warmup={tcfg.warmup_steps}, decay_start={tcfg.decay_start_step}, decay_steps={tcfg.decay_steps})")

    train_step = _build_train_step(
        spec, tcfg, cos, sin, compute_dtype, ns_dtype, use_loss_scale,
        grad_accum_steps, mesh, param_shardings, opt_shardings, step_sharding, replicated,
    )

    def eval_loss_fn(p, x, y):
        return forward_loss_2d(p, x, y, cos, sin, spec, compute_dtype, return_z_loss=False, mesh=mesh)

    stop_requested = {"flag": False}

    def _handle_signal(_signum, _frame):
        print("\n[Scale] Termination signal received — finishing step and saving checkpoint...")
        stop_requested["flag"] = True

    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    loader = StreamingDataLoader(
        tokenizer=tokenizer,
        dataset=tcfg.dataset,
        split="train",
        batch_size=global_micro_batch,
        seq_len=spec.sequence_len,
        grad_accum_steps=grad_accum_steps,
        shard_cache_dir=tcfg.shard_cache_dir,
        keep_shards=tcfg.keep_shards,
        cursor=cursor,
        prefetch=3,
    )

    active_cursor = loader.cursor

    def make_trainer_state() -> dict:
        return {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_bpb": best_val_bpb if math.isfinite(best_val_bpb) else None,
            "wsd_phase": tcfg.wsd_phase,
            "decay_start_step": tcfg.decay_start_step,
            "data_cursor": active_cursor.to_dict(),
            "model_spec": spec.to_dict(),
            "train_config": tcfg.to_dict(),
        }

    session_steps = 0
    loss_scale = 32768.0 if use_loss_scale else 1.0
    good_scale_steps = 0
    last_saved_step = step
    last_val_bpb = best_val_bpb

    # Async pipelining window trackers (eliminates host-device sync bubbles between steps!)
    window_start = time.perf_counter()
    window_steps = 0
    profile_active = False
    watchdog = TrainingWatchdog(best_val_bpb=best_val_bpb, enabled=tcfg.watchdog_enabled)
    instability_reason: Optional[str] = None

    try:
        x_np, y_np, next_cur = next(loader)
        x_dev = jax.device_put(x_np, step_sharding)
        y_dev = jax.device_put(y_np, step_sharding)
        while not stop_requested["flag"]:
            if tcfg.max_steps is not None and session_steps >= tcfg.max_steps:
                print(f"\n[Scale] Reached max_steps={tcfg.max_steps} for this session.")
                break
            if tokens_seen >= tcfg.total_tokens:
                print(f"\n[Scale] Reached target total_tokens={tcfg.total_tokens:,}.")
                break
            if tcfg.wsd_phase == "decay" and tcfg.decay_start_step is not None:
                if step >= tcfg.decay_start_step + tcfg.decay_steps:
                    print(f"\n[Scale] Completed WSD decay phase ({tcfg.decay_steps} steps).")
                    break
            if tcfg.max_hours is not None and (time.time() - t_session_start) / 3600.0 >= tcfg.max_hours:
                print(f"\n[Scale] Reached max_hours={tcfg.max_hours:.2f}h safety limit.")
                break

            lrm = compute_wsd_lrm(step, tcfg)
            muon_mom = compute_muon_momentum(step)
            muon_wd = compute_weight_decay(step, tcfg)

            next_session_step = session_steps + 1
            if tcfg.profile_dir and next_session_step == tcfg.profile_start_step:
                os.makedirs(tcfg.profile_dir, exist_ok=True)
                print(
                    f"[Profile] Starting XProf trace at session step {next_session_step} "
                    f"for {tcfg.profile_steps} step(s) -> {tcfg.profile_dir}",
                    flush=True,
                )
                jax.profiler.start_trace(tcfg.profile_dir)
                profile_active = True

            cur = next_cur
            active_cursor = cur
            params, opt_state, ce_arr, z_arr, gnorm_arr, finite_arr = train_step(
                params,
                opt_state,
                x_dev,
                y_dev,
                jax.device_put(jnp.float32(lrm), replicated),
                jax.device_put(jnp.float32(muon_mom), replicated),
                jax.device_put(jnp.float32(muon_wd), replicated),
                jax.device_put(jnp.float32(loss_scale), replicated),
            )

            step += 1
            session_steps += 1
            window_steps += 1
            tokens_seen += total_batch_size

            if profile_active and session_steps == tcfg.profile_start_step + max(1, tcfg.profile_steps) - 1:
                jax.block_until_ready(ce_arr)
                jax.profiler.stop_trace()
                profile_active = False
                print(f"[Profile] Stopped XProf trace -> {tcfg.profile_dir}", flush=True)

            # Decide whether we need to synchronize this step (logging, fp16 loss scale, eval, save, or warmup)
            should_log = session_steps <= 2 or (tcfg.log_every > 0 and step % tcfg.log_every == 0)
            should_eval = tcfg.eval_every > 0 and step % tcfg.eval_every == 0
            should_sample = tcfg.sample_every > 0 and step % tcfg.sample_every == 0
            should_save = tcfg.save_every > 0 and step % tcfg.save_every == 0
            is_last_step = (
                (tcfg.max_steps is not None and session_steps >= tcfg.max_steps)
                or tokens_seen >= tcfg.total_tokens
                or (tcfg.wsd_phase == "decay" and tcfg.decay_start_step is not None and step >= tcfg.decay_start_step + tcfg.decay_steps)
                or (tcfg.max_hours is not None and (time.time() - t_session_start) / 3600.0 >= tcfg.max_hours)
            )

            # Overlap next host batch fetch + async H2D DMA with current TPU/GPU compute (skip on final step!)
            if not is_last_step and not stop_requested["flag"]:
                x_np, y_np, next_cur = next(loader)
                x_dev = jax.device_put(x_np, step_sharding)
                y_dev = jax.device_put(y_np, step_sharding)

            if should_log or use_loss_scale or should_eval or should_save or is_last_step:
                ce_f = float(jax.block_until_ready(ce_arr))
                now = time.perf_counter()

                if use_loss_scale:
                    finite_ok = bool(finite_arr)
                    if finite_ok:
                        good_scale_steps += 1
                        if good_scale_steps >= 200:
                            loss_scale = min(loss_scale * 2.0, 2.0**24)
                            good_scale_steps = 0
                    else:
                        loss_scale = max(loss_scale * 0.5, 1.0)
                        good_scale_steps = 0

                if math.isnan(ce_f) or math.isinf(ce_f):
                    instability_reason = f"Training diverged (loss={ce_f}) at step {step}"
                    raise TrainingInstabilityError(instability_reason)

                if should_log:
                    dt = (now - window_start) / max(window_steps, 1)
                    z_f = float(z_arr)
                    gnorm_f = float(gnorm_arr)
                    tok_s = total_batch_size / max(dt, 1e-6)
                    mfu = 100.0 * flops_per_tok * tok_s / max(peak_flops, 1.0)
                    tag = "compile" if session_steps == 1 else f"step {step:07d}"
                    print(
                        f"[{tag}] tok={tokens_seen/1e9:.4f}B | loss={ce_f:.4f} | z={z_f:.2f} | "
                        f"gnorm={gnorm_f:.2f} | lrm={lrm:.3f} | dt={dt*1000:.0f}ms | "
                        f"tok/s={tok_s:,.0f} | mfu={mfu:.1f}% | shard={cur.shard_idx} ep={cur.epoch}",
                        flush=True,
                    )
                    _append_metric(
                        metrics_path,
                        {
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "loss": round(ce_f, 6),
                            "z_loss": round(z_f, 6),
                            "grad_norm": round(gnorm_f, 4),
                            "lrm": round(lrm, 5),
                            "dt_ms": round(dt * 1000.0, 1),
                            "tok_per_sec": round(tok_s, 1),
                            "mfu_percent": round(mfu, 2),
                            "shard": cur.shard_idx,
                            "epoch": cur.epoch,
                        },
                    )
                    window_start = time.perf_counter()
                    window_steps = 0

                    wd_reason = watchdog.check_step(step, ce_f, z_f, gnorm_f)
                    if wd_reason is not None:
                        instability_reason = wd_reason
                        raise TrainingInstabilityError(wd_reason)

            if session_steps == 2:
                gc.collect()
                gc.disable()
            elif session_steps % 1000 == 0:
                gc.collect()

            if should_eval:
                gc.enable()
                val_bpb = evaluate_val_bpb(
                    eval_loss_fn=eval_loss_fn,
                    params=params,
                    tokenizer=tokenizer,
                    tokenizer_dir=tokenizer_dir,
                    dataset=tcfg.dataset,
                    global_batch_size=global_micro_batch,
                    seq_len=spec.sequence_len,
                    eval_tokens=tcfg.eval_tokens,
                    shard_cache_dir=tcfg.shard_cache_dir,
                    data_sharding=batch_sharding,
                )
                last_val_bpb = val_bpb
                improved = val_bpb < best_val_bpb
                if improved:
                    best_val_bpb = val_bpb
                print(
                    f"  >>> [Eval step {step}] val_bpb={val_bpb:.6f} "
                    f"(best={best_val_bpb:.6f}{' *NEW BEST*' if improved else ''})",
                    flush=True,
                )
                _append_metric(metrics_path, {"step": step, "tokens_seen": tokens_seen, "val_bpb": round(val_bpb, 6)})
                gc.disable()
                window_start = time.perf_counter()
                window_steps = 0

                wd_eval_reason = watchdog.check_eval(step, val_bpb)
                if wd_eval_reason is not None:
                    instability_reason = wd_eval_reason
                    raise TrainingInstabilityError(wd_eval_reason)

            if should_save:
                if watchdog.is_degraded:
                    print(
                        f"  >>> [Watchdog] Skipping checkpoint save at step {step} (run degraded) "
                        f"to preserve last healthy checkpoint at step {last_saved_step}!",
                        flush=True,
                    )
                else:
                    gc.enable()
                    ckpt_mgr.save(
                        step=step,
                        params=params,
                        opt_state=opt_state,
                        trainer_state=make_trainer_state(),
                        tokenizer_dir=tokenizer_dir,
                        metrics_path=metrics_path,
                    )
                    last_saved_step = step
                    gc.disable()
                window_start = time.perf_counter()
                window_steps = 0

            if should_sample:
                try:
                    prompt_ids = tokenizer.encode(SAMPLE_PROMPT, prepend=tokenizer.get_bos_token_id())
                    out_ids = generate_tokens(
                        params, prompt_ids, max_new_tokens=48, spec=spec, compute_dtype=compute_dtype,
                        temperature=0.8, top_k=40, seed=step, batch_sharding=batch_sharding,
                    )
                    sample_txt = tokenizer.decode(out_ids[1:]).replace("\n", " ")
                    print(f"  >>> [Sample step {step}] {sample_txt[:220]}", flush=True)
                except Exception as exc:
                    print(f"  >>> [Sample step {step}] skipped ({exc})", flush=True)
                window_start = time.perf_counter()
                window_steps = 0

    except TrainingInstabilityError as exc:
        print("\n" + "!" * 78, flush=True)
        print(f"[Watchdog] AUTO-ABORTING COMPROMISED RUN TO SAVE TPU QUOTA!", flush=True)
        print(f"  Reason:                 {exc}", flush=True)
        print(f"  Last healthy saved step:{last_saved_step} (best_val_bpb={best_val_bpb:.6f})", flush=True)
        print(f"  Remote HF latest/ left 100% UNTOUCHED (no corrupted checkpoint uploaded).", flush=True)
        print("!" * 78 + "\n", flush=True)
        raise
    except KeyboardInterrupt:
        print("\n[Scale] Interrupted by user — saving final checkpoint before exit...")
    finally:
        if profile_active:
            try:
                jax.profiler.stop_trace()
            except Exception:
                pass
            profile_active = False
        gc.enable()
        signal.signal(signal.SIGTERM, old_sigterm)
        loader.close()

    if session_steps > 0 and instability_reason is None:
        last_val_bpb = evaluate_val_bpb(
            eval_loss_fn=eval_loss_fn,
            params=params,
            tokenizer=tokenizer,
            tokenizer_dir=tokenizer_dir,
            dataset=tcfg.dataset,
            global_batch_size=global_micro_batch,
            seq_len=spec.sequence_len,
            eval_tokens=tcfg.eval_tokens,
            shard_cache_dir=tcfg.shard_cache_dir,
            data_sharding=batch_sharding,
        )
        if last_val_bpb < best_val_bpb:
            best_val_bpb = last_val_bpb
        _append_metric(metrics_path, {"step": step, "tokens_seen": tokens_seen, "val_bpb": round(last_val_bpb, 6)})

        if step > last_saved_step and not watchdog.is_degraded:
            ckpt_mgr.save(
                step=step,
                params=params,
                opt_state=opt_state,
                trainer_state=make_trainer_state(),
                tokenizer_dir=tokenizer_dir,
                metrics_path=metrics_path,
            )

    peak_mem_mb = get_peak_memory_mb()
    summary = {
        "step": step,
        "session_steps": session_steps,
        "tokens_seen": tokens_seen,
        "val_bpb": last_val_bpb if math.isfinite(last_val_bpb) else 0.0,
        "best_val_bpb": best_val_bpb if math.isfinite(best_val_bpb) else 0.0,
        "peak_mem_mb": round(peak_mem_mb, 1),
        "num_params": p_counts["total"],
        "accelerator": f"{hw.device_kind} x{hw.num_devices} ({hw.platform}, {dtype_name})",
        "elapsed_seconds": round(time.time() - t_session_start, 1),
    }
    print("---")
    print(f"step:            {summary['step']}")
    print(f"tokens_seen_M:   {summary['tokens_seen'] / 1e6:.2f}")
    print(f"val_bpb:         {summary['val_bpb']:.6f}")
    print(f"best_val_bpb:    {summary['best_val_bpb']:.6f}")
    print(f"peak_vram_mb:    {summary['peak_mem_mb']:.1f}")
    print(f"num_params_M:    {summary['num_params'] / 1e6:.2f}")
    print(f"accelerator:     {summary['accelerator']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']:.1f}")
    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-MFU resumable pure-JAX FSDP pretraining engine for 100M -> 1B+ models"
    )
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default="1b", help="Model scale preset")
    parser.add_argument("--arch-config", type=str, default=None, help="Optional JSON overrides from autoresearch winner")
    parser.add_argument(
        "--remat-policy",
        choices=["none", "save_all_compute", "save_attn", "save_qkv_attn", "dots_with_no_batch_dims", "dots", "full"],
        default=None,
    )
    parser.add_argument("--attn-impl", choices=["auto", "splash", "cudnn", "xla"], default=None)
    parser.add_argument("--loss-chunk-size", type=int, default=None, help="Chunked CE token count (0=off, 2048=default)")
    parser.add_argument("--splash-block-size", type=int, default=None, help="TPU SplashAttention tile size (512/1024)")
    parser.add_argument(
        "--matmul-quant",
        choices=["none", "int8", "int8-all", "fp8"],
        default=None,
        help="Experimental dynamic INT8/FP8 quantized linear GEMMs",
    )
    parser.add_argument("--scan-unroll", type=int, default=None, help="jax.lax.scan unroll factor (1 or 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="climbmix",
        help="Source (climbmix/codeparrot/finemath/cosmopedia/...), mix preset (mix-1b/mix-cooldown/mix-code), or inline 'src:w,...'",
    )
    parser.add_argument("--total-tokens", type=int, default=20_000_000_000, help="Target total training tokens")
    parser.add_argument("--total-batch-size", type=int, default=524_288, help="Global tokens per optimizer step")
    parser.add_argument("--per-device-batch", type=int, default=4, help="Microbatch sequences per chip")
    parser.add_argument("--eval-tokens", type=int, default=4 * 524_288, help="Validation tokens per eval")
    parser.add_argument("--compute-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--param-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Master weight precision (bfloat16 uses unbiased stochastic rounding and cuts weight HBM ~3x)",
    )
    parser.add_argument("--opt-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--muon-vmap", action="store_true", help="Batch Muon across layers via vmap (faster on v5e-8 / G4)")
    parser.add_argument("--no-layer-sharded-muon", action="store_true", help="Disable U-only layer-sharded Split-Muon on multi-chip FSDP")
    parser.add_argument("--pallas-kernels", action="store_true", help="Opt-in custom Pallas TPU Mosaic SwiGLU+Quant kernel")
    parser.add_argument("--no-pallas-kernels", action="store_true", help="Explicitly disable custom Pallas TPU Mosaic kernels")
    parser.add_argument("--pallas-cutce", action="store_true", help="Opt-in Zero-HBM-Logits BF16 Pallas CutCE (saves ~1 GB HBM)")
    parser.add_argument("--no-fsdp", action="store_true", help="Disable FSDP parameter sharding (replicate instead)")
    parser.add_argument("--phase", choices=["stable", "decay"], default="stable", help="WSD schedule phase")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=4000)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--scalar-lr", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--gate-adamw", action="store_true", help="Update small gate tensors (ve_gate, w_attn_gate) with AdamW instead of Muon")
    parser.add_argument("--split-muon", action="store_true", default=True, help="Qwen 3.8 Split-Muon (default ON): orthogonalize Q/K/V and gate/up independently")
    parser.add_argument("--no-split-muon", action="store_false", dest="split_muon", help="Disable Qwen 3.8 Split-Muon orthogonalization")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps for this session")
    parser.add_argument("--max-hours", type=float, default=8.5, help="Graceful exit wall-clock cap in hours")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2, help="Local checkpoints retained on disk (use 1 on Kaggle 19.5 GB disk)")
    parser.add_argument("--no-watchdog", action="store_true", help="Disable automatic divergence/regression abort & checkpoint protection watchdog")
    parser.add_argument("--profile-dir", type=str, default=None, help="Optional directory for real-silicon XProf jax.profiler trace capture")
    parser.add_argument("--profile-start-step", type=int, default=3, help="Session step (1-based) to begin XProf trace capture")
    parser.add_argument("--profile-steps", type=int, default=2, help="Number of steps to record in XProf trace")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/1b")
    parser.add_argument("--shard-cache-dir", type=str, default="/tmp/autoresearch_shards")
    parser.add_argument("--keep-shards", type=int, default=2, help="Max rolling shards kept on disk per source")
    parser.add_argument("--sync-backend", choices=["none", "local", "hf", "kaggle"], default="none")
    parser.add_argument("--remote-target", type=str, default=None, help="Drive dir, HF repo_id (supports auto/<repo>), or Kaggle dataset slug")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-only", action="store_true", help="Run pure steady-state MFU microbenchmark")
    parser.add_argument("--bench-steps", type=int, default=8, help="Timed steps for --benchmark-only")
    args = parser.parse_args()

    # Validate dataset / mix string immediately before allocating accelerators
    parse_dataset_mix(args.dataset)

    if args.arch_config:
        raw_arch = args.arch_config.strip()
        arch_overrides = json.loads(raw_arch) if raw_arch.startswith("{") else json.loads(Path(raw_arch).read_text(encoding="utf-8"))
    else:
        arch_overrides = {}
    if args.remat_policy is not None:
        arch_overrides["remat_policy"] = args.remat_policy
    if args.attn_impl is not None:
        arch_overrides["attn_impl"] = args.attn_impl
    if args.loss_chunk_size is not None:
        arch_overrides["loss_chunk_size"] = args.loss_chunk_size
    if args.splash_block_size is not None:
        arch_overrides["splash_block_size"] = args.splash_block_size
    if args.matmul_quant is not None:
        arch_overrides["matmul_quant"] = args.matmul_quant
    if args.scan_unroll is not None:
        arch_overrides["scan_unroll"] = args.scan_unroll
    if args.pallas_kernels:
        arch_overrides["use_pallas_kernels"] = True
    if args.no_pallas_kernels:
        arch_overrides["use_pallas_kernels"] = False
    if args.pallas_cutce:
        arch_overrides["use_pallas_cutce"] = True
        arch_overrides["use_pallas_kernels"] = True

    spec = get_model_preset(args.preset, arch_overrides=arch_overrides or None)

    tcfg = TrainConfig(
        dataset=args.dataset,
        total_tokens=args.total_tokens,
        eval_tokens=args.eval_tokens,
        shard_cache_dir=args.shard_cache_dir,
        keep_shards=args.keep_shards,
        total_batch_size=args.total_batch_size,
        per_device_batch=args.per_device_batch,
        compute_dtype=args.compute_dtype,
        param_dtype=args.param_dtype,
        opt_dtype=args.opt_dtype,
        muon_vmap=args.muon_vmap,
        muon_layer_sharded=not args.no_layer_sharded_muon,
        gate_adamw=args.gate_adamw,
        split_muon=args.split_muon,
        use_fsdp=not args.no_fsdp,
        matrix_lr=args.matrix_lr,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        wsd_phase=args.phase,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        max_steps=args.max_steps,
        max_hours=args.max_hours,
        log_every=args.log_every,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        keep_last_checkpoints=args.keep_last_checkpoints,
        watchdog_enabled=not args.no_watchdog,
        profile_dir=args.profile_dir,
        profile_start_step=args.profile_start_step,
        profile_steps=args.profile_steps,
        checkpoint_dir=args.checkpoint_dir,
        sync_backend=args.sync_backend,
        remote_target=args.remote_target,
        resume=not args.no_resume,
        seed=args.seed,
    )

    if args.benchmark_only:
        run_benchmark(spec, tcfg, timed_steps=args.bench_steps)
    else:
        run_pretrain(spec, tcfg)


if __name__ == "__main__":
    main()
