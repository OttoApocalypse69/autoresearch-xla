"""
Resumable multi-chip FSDP pretraining engine for scaling autoresearch-xla to 1B+.

Supports:
  - Kaggle 8-chip TPU v5e-8 (FSDP shards 1B weights+opt to ~1.15 GB/chip, ~100K tok/s)
  - Colab TPU v6e-1 / v5e-1 and GPUs (H100 / A100 / L4 / T4 / Kaggle 2xT4)
  - WSD (Warmup -> Stable -> Decay) schedule across unlimited 9h/12h sessions
  - Disk-bounded streaming ClimbMix shards with exact resumable DataCursor
  - Topology-agnostic checkpoints with Google Drive / HuggingFace Hub / Kaggle sync
  - Pluggable `--arch-config` JSON to scale winning `autoresearch-xla` discoveries

Examples:
    # 1B pretraining on Kaggle TPU v5e-8 with HuggingFace Hub auto-resume:
    python -m scale.pretrain --preset 1b --dataset climbmix \\
        --sync-backend hf --remote-target <user>/my-1b-base --max-hours 8.5

    # Final WSD cooldown decay phase (resumes latest checkpoint automatically):
    python -m scale.pretrain --preset 1b --phase decay --decay-steps 4000 \\
        --sync-backend hf --remote-target <user>/my-1b-base

    # Fast local / CPU / 8-device simulated FSDP smoke test:
    python -m scale.pretrain --preset smoke --dataset synthetic --max-steps 10
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from prepare import JAX_CACHE_DIR, detect_hardware, get_peak_memory_mb
from scale.checkpoint import CheckpointManager
from scale.config import MODEL_PRESETS, ModelSpec, TrainConfig, get_model_preset
from scale.data import (
    DataCursor,
    StreamingDataLoader,
    ensure_tokenizer,
    evaluate_val_bpb,
)
from scale.fsdp import build_tree_shardings, create_fsdp_mesh, summarize_sharding
from scale.model import (
    count_params,
    estimate_flops_per_token,
    forward_loss_2d,
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

os.makedirs(JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

SAMPLE_PROMPT = "The key insight behind scaling neural networks is"


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


def run_pretrain(spec: ModelSpec, tcfg: TrainConfig) -> dict:
    """Execute a resumable FSDP pretraining session and return final summary metrics."""
    t_session_start = time.time()
    hw = detect_hardware()
    dtype_name, compute_dtype = _resolve_compute_dtype(tcfg, hw.compute_dtype)
    opt_dtype = jnp.bfloat16 if tcfg.opt_dtype == "bfloat16" else jnp.float32
    use_loss_scale = compute_dtype == jnp.float16
    ns_dtype = jnp.bfloat16 if compute_dtype == jnp.bfloat16 else jnp.float32

    mesh = create_fsdp_mesh()
    replicated = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("fsdp", None))
    step_sharding = NamedSharding(mesh, P(None, "fsdp", None))

    # Validate batch divisibility across devices
    global_micro_batch = tcfg.per_device_batch * hw.num_devices
    tokens_per_micro = global_micro_batch * spec.sequence_len
    total_batch_size = tcfg.total_batch_size
    if total_batch_size < tokens_per_micro:
        total_batch_size = tokens_per_micro
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

    print(
        f"[Scale] Hardware: {hw.device_kind} x{hw.num_devices} "
        f"(platform={hw.platform}, compute={dtype_name}, opt={tcfg.opt_dtype}, "
        f"fsdp={tcfg.use_fsdp}, peak={hw.peak_tflops_total:.1f} TFLOPS)"
    )

    # 1. Pull remote checkpoint early so we restore the exact saved tokenizer on fresh VMs
    if tcfg.resume and tcfg.sync_backend != "none" and tcfg.remote_target:
        try:
            ckpt_mgr.sync_pull()
        except Exception as exc:
            print(f"[Scale] No remote checkpoint found yet ({exc}); starting fresh.")

    # 2. Load or train tokenizer (matching spec.vocab_size)
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

    # 3. Direct-sharded initialization (zero chip-0 OOM spike on 1B+ models!)
    key = jax.random.PRNGKey(tcfg.seed)
    abstract_params = jax.eval_shape(functools.partial(init_params, spec=spec), key)
    abstract_opt = jax.eval_shape(functools.partial(init_opt_state, opt_dtype=opt_dtype), abstract_params)
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
            pull_remote=False,  # already pulled above
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
        init_p_jit = jax.jit(functools.partial(init_params, spec=spec), out_shardings=param_shardings)
        init_o_jit = jax.jit(
            functools.partial(init_opt_state, opt_dtype=opt_dtype),
            in_shardings=(param_shardings,),
            out_shardings=opt_shardings,
        )
        params = init_p_jit(key)
        opt_state = init_o_jit(params)
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

    print(f"[Scale] Model: {p_counts['total']/1e6:.2f}M params ({spec.n_layer}L x {spec.n_embd}D, {spec.activation}, GQA {spec.n_head}:{spec.n_kv_head}, win={spec.window_pattern})")
    print(f"[Scale] FSDP params: {p_shard_info['total_mb']:.1f} MB total -> {p_shard_info['per_device_mb']:.1f} MB/dev ({p_shard_info['sharded_leaves']} sharded leaves)")
    print(f"[Scale] FSDP opt:    {o_shard_info['total_mb']:.1f} MB total -> {o_shard_info['per_device_mb']:.1f} MB/dev ({o_shard_info['sharded_leaves']} sharded leaves)")
    print(f"[Scale] Batching:    {total_batch_size:,} tok/step ({tcfg.per_device_batch} seq/dev x {hw.num_devices} dev x {grad_accum_steps} accum) | FLOPs/tok={flops_per_tok:.3e}")
    print(f"[Scale] WSD phase:   {tcfg.wsd_phase} (warmup={tcfg.warmup_steps}, decay_start={tcfg.decay_start_step}, decay_steps={tcfg.decay_steps})")

    # 4. Fused JIT Training Step with FSDP Sharding + Buffer Donation
    def micro_loss_fn(p, x, y, scale):
        nll, z_loss = forward_loss_2d(p, x, y, cos, sin, spec, compute_dtype, return_z_loss=True)
        ce = jnp.mean(nll, dtype=jnp.float32)
        total = ce + jnp.float32(spec.z_loss_weight) * z_loss
        return total * scale, (ce, z_loss)

    grad_fn = jax.value_and_grad(micro_loss_fn, has_aux=True)

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
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, p)

        def accum_body(carry, batch):
            g_acc, ce_acc, z_acc = carry
            xb, yb = batch
            (_, (ce_val, z_val)), g_micro = grad_fn(p, xb, yb, loss_scale)
            g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g_micro)
            return (g_acc, ce_acc + ce_val, z_acc + z_val), None

        (g_sum, ce_sum, z_sum), _ = jax.lax.scan(
            accum_body,
            (zero_grads, jnp.float32(0.0), jnp.float32(0.0)),
            (x_steps, y_steps),
        )
        inv_accum = jnp.float32(1.0 / grad_accum_steps)
        grads = jax.tree_util.tree_map(lambda g: g * (inv_accum / loss_scale), g_sum)
        mean_ce = ce_sum * inv_accum
        mean_z = z_sum * inv_accum

        if use_loss_scale:
            leaves = jax.tree_util.tree_leaves(grads)
            all_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in leaves]))
            p_cand, st_cand, gnorm = optimizer_step(p, grads, st, lrm, muon_mom, muon_wd, spec, tcfg, ns_dtype)
            p_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), p_cand, p)
            st_next = jax.tree_util.tree_map(lambda a, b: jnp.where(all_finite, a, b), st_cand, st)
            return p_next, st_next, mean_ce, mean_z, gnorm, all_finite
        else:
            p_next, st_next, gnorm = optimizer_step(p, grads, st, lrm, muon_mom, muon_wd, spec, tcfg, ns_dtype)
            return p_next, st_next, mean_ce, mean_z, gnorm, jnp.bool_(True)

    def eval_loss_fn(p, x, y):
        return forward_loss_2d(p, x, y, cos, sin, spec, compute_dtype, return_z_loss=False)

    # 5. Graceful SIGTERM handler (saves checkpoint if Kaggle/Colab sends preemption signal)
    stop_requested = {"flag": False}

    def _handle_signal(_signum, _frame):
        print("\n[Scale] Termination signal received — finishing step and saving checkpoint...")
        stop_requested["flag"] = True

    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    # 6. Streaming Data Loader & Main Training Loop
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
        prefetch=2,
    )

    def make_trainer_state() -> dict:
        return {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_bpb": best_val_bpb if math.isfinite(best_val_bpb) else None,
            "wsd_phase": tcfg.wsd_phase,
            "decay_start_step": tcfg.decay_start_step,
            "data_cursor": loader.cursor.to_dict(),
            "model_spec": spec.to_dict(),
            "train_config": tcfg.to_dict(),
        }

    session_steps = 0
    loss_scale = 32768.0 if use_loss_scale else 1.0
    good_scale_steps = 0
    last_saved_step = step
    last_val_bpb = best_val_bpb

    try:
        x_np, y_np, _cur = next(loader)
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

            t0 = time.perf_counter()
            x_dev = jax.device_put(x_np, step_sharding)
            y_dev = jax.device_put(y_np, step_sharding)

            lrm = compute_wsd_lrm(step, tcfg)
            muon_mom = compute_muon_momentum(step)
            muon_wd = compute_weight_decay(step, tcfg)

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

            # Overlap host shard packing with accelerator execution
            x_np, y_np, cur = next(loader)

            ce_f = float(jax.block_until_ready(ce_arr))
            z_f = float(z_arr)
            gnorm_f = float(gnorm_arr)
            finite_ok = bool(finite_arr)
            dt = time.perf_counter() - t0

            if use_loss_scale:
                if finite_ok:
                    good_scale_steps += 1
                    if good_scale_steps >= 200:
                        loss_scale = min(loss_scale * 2.0, 2.0**24)
                        good_scale_steps = 0
                else:
                    loss_scale = max(loss_scale * 0.5, 1.0)
                    good_scale_steps = 0

            if math.isnan(ce_f) or math.isinf(ce_f):
                raise RuntimeError(f"Training diverged (loss={ce_f}) at step {step}")

            step += 1
            session_steps += 1
            tokens_seen += total_batch_size

            tok_s = total_batch_size / max(dt, 1e-6)
            mfu = 100.0 * flops_per_tok * tok_s / max(peak_flops, 1.0)

            if session_steps <= 2 or (tcfg.log_every > 0 and step % tcfg.log_every == 0):
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

            if session_steps == 2:
                gc.collect()
                gc.disable()
            elif session_steps % 1000 == 0:
                gc.collect()

            # Periodic validation BPB
            if tcfg.eval_every > 0 and step % tcfg.eval_every == 0:
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

            # Periodic qualitative sample generation
            if tcfg.sample_every > 0 and step % tcfg.sample_every == 0:
                prompt_ids = tokenizer.encode(SAMPLE_PROMPT, prepend=tokenizer.get_bos_token_id())
                out_ids = generate_tokens(
                    params, prompt_ids, max_new_tokens=48, spec=spec, compute_dtype=compute_dtype,
                    temperature=0.8, top_k=40, seed=step, batch_sharding=batch_sharding,
                )
                sample_txt = tokenizer.decode(out_ids[1:]).replace("\n", " ")
                print(f"  >>> [Sample step {step}] {sample_txt[:220]}", flush=True)

            # Periodic checkpoint + remote sync
            if tcfg.save_every > 0 and step % tcfg.save_every == 0:
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

    except KeyboardInterrupt:
        print("\n[Scale] Interrupted by user — saving final checkpoint before exit...")
    finally:
        gc.enable()
        signal.signal(signal.SIGTERM, old_sigterm)
        loader.close()

    # Final validation & checkpoint save so zero work is lost at session end
    if session_steps > 0:
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

        if step > last_saved_step:
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
        description="Resumable pure-JAX FSDP pretraining engine for 100M -> 1B+ models on Colab & Kaggle"
    )
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default="1b", help="Model scale preset")
    parser.add_argument("--arch-config", type=str, default=None, help="Optional JSON overrides from autoresearch winner")
    parser.add_argument("--dataset", choices=["climbmix", "tinystories", "synthetic"], default="climbmix")
    parser.add_argument("--total-tokens", type=int, default=20_000_000_000, help="Target total training tokens")
    parser.add_argument("--total-batch-size", type=int, default=524_288, help="Global tokens per optimizer step")
    parser.add_argument("--per-device-batch", type=int, default=4, help="Microbatch sequences per chip")
    parser.add_argument("--eval-tokens", type=int, default=4 * 524_288, help="Validation tokens per eval")
    parser.add_argument("--compute-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--opt-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--no-fsdp", action="store_true", help="Disable FSDP parameter sharding (replicate instead)")
    parser.add_argument("--phase", choices=["stable", "decay"], default="stable", help="WSD schedule phase")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=4000)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps for this session")
    parser.add_argument("--max-hours", type=float, default=8.5, help="Graceful exit wall-clock cap in hours")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/1b")
    parser.add_argument("--shard-cache-dir", type=str, default="/tmp/autoresearch_shards")
    parser.add_argument("--sync-backend", choices=["none", "local", "hf", "kaggle"], default="none")
    parser.add_argument("--remote-target", type=str, default=None, help="Drive dir, HF repo_id, or Kaggle dataset slug")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arch_overrides = json.loads(Path(args.arch_config).read_text(encoding="utf-8")) if args.arch_config else None
    spec = get_model_preset(args.preset, arch_overrides=arch_overrides)

    tcfg = TrainConfig(
        dataset=args.dataset,
        total_tokens=args.total_tokens,
        eval_tokens=args.eval_tokens,
        shard_cache_dir=args.shard_cache_dir,
        total_batch_size=args.total_batch_size,
        per_device_batch=args.per_device_batch,
        compute_dtype=args.compute_dtype,
        opt_dtype=args.opt_dtype,
        use_fsdp=not args.no_fsdp,
        matrix_lr=args.matrix_lr,
        embedding_lr=args.embedding_lr,
        weight_decay=args.weight_decay,
        wsd_phase=args.phase,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        max_steps=args.max_steps,
        max_hours=args.max_hours,
        log_every=args.log_every,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        sync_backend=args.sync_backend,
        remote_target=args.remote_target,
        resume=not args.no_resume,
        seed=args.seed,
    )

    run_pretrain(spec, tcfg)


if __name__ == "__main__":
    main()
