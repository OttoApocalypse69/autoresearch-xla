"""
Orbax-free, topology-agnostic checkpointing & remote sync for 1B+ pretraining.

Why topology-agnostic flat tensors (safetensors / npz + JSON) instead of Orbax?
- Zero version breakage between Colab GPU (JAX 0.11/0.12), Colab TPU (JAX 0.7.2),
  and Kaggle TPU images.
- Resume seamlessly across *different* topologies: e.g. train 9h on Kaggle 8-chip
  TPU v5e-8 -> resume cooldown on Colab 1-chip TPU v6e-1 or H100!
- Final weights are saved in standard `model.safetensors` (when installed) or
  `model.npz`, immediately usable by JAX, PyTorch, or vLLM for post-training.

Supported remote sync backends:
  - "none"   : local checkpoint_dir only
  - "local"  : syncs latest checkpoint + tokenizer + metrics to Google Drive / persistent dir
  - "hf"     : private HuggingFace Hub repo (reads HF_TOKEN from env / Colab / Kaggle Secrets)
  - "kaggle" : private Kaggle Dataset via `kaggle datasets create/version`
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp

from scale.fsdp import shard_pytree


# ---------------------------------------------------------------------------
# Secret Discovery Helper (Env / Colab Secrets / Kaggle Secrets)
# ---------------------------------------------------------------------------

def _get_secret(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val.strip()
    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get(name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = UserSecretsClient().get_secret(name)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Flat Pytree <-> NumPy Dict Serialization (handles bfloat16 cleanly!)
# ---------------------------------------------------------------------------

def flatten_pytree(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Gather sharded JAX arrays onto host CPU and flatten nested dicts with '/' keys."""
    flat: dict[str, np.ndarray] = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_pytree(v, key))
        else:
            arr = jax.device_get(v)
            # NumPy handles bfloat16 cleanly via ml_dtypes, or preserve dtype metadata
            flat[key] = np.asarray(arr)
    return flat


def unflatten_into_template(flat: dict[str, np.ndarray], template: dict, prefix: str = "") -> dict:
    """Reconstruct nested dict matching `template` structure and target JAX dtypes."""
    out: dict[str, Any] = {}
    for k, v in template.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out[k] = unflatten_into_template(flat, v, key)
        else:
            if key not in flat:
                raise KeyError(f"Checkpoint missing expected key: {key!r}")
            target_dtype = v.dtype
            arr = flat[key]
            out[k] = jnp.asarray(arr, dtype=target_dtype)
    return out


def _save_arrays(path_no_ext: Path, arrays: dict[str, np.ndarray], prefer_safetensors: bool = False) -> Path:
    """Save flat array dict to `.safetensors` (if preferred & supported) or uncompressed `.npz`."""
    has_bf16 = any(str(arr.dtype) == "bfloat16" for arr in arrays.values())
    if prefer_safetensors and not has_bf16:
        st_path = path_no_ext.with_suffix(".safetensors")
        try:
            from safetensors.numpy import save_file  # type: ignore

            save_file(arrays, str(st_path))
            return st_path
        except Exception:
            if st_path.exists():
                st_path.unlink(missing_ok=True)

    # Standard .npz handles float32, float16, int32; store bfloat16 safely as uint16 view + tag
    packed: dict[str, np.ndarray] = {}
    bf16_keys: list[str] = []
    for k, arr in arrays.items():
        if str(arr.dtype) == "bfloat16":
            packed[k] = arr.view(np.uint16)
            bf16_keys.append(k)
        else:
            packed[k] = arr
    if bf16_keys:
        packed["__bf16_keys__"] = np.array(bf16_keys, dtype=object)

    npz_path = path_no_ext.with_suffix(".npz")
    # Save uncompressed for maximum I/O speed on 1B+ models (saving 6.8 GB takes ~3-5s vs 60s+ compressed!)
    np.savez(str(npz_path), **packed)
    return npz_path


def _load_arrays(path_no_ext: Path) -> dict[str, np.ndarray]:
    st_path = path_no_ext.with_suffix(".safetensors")
    if st_path.exists():
        from safetensors.numpy import load_file  # type: ignore

        return dict(load_file(str(st_path)))

    npz_path = path_no_ext.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Neither {st_path} nor {npz_path} exists.")

    with np.load(str(npz_path), allow_pickle=True) as data:
        bf16_keys = set(data["__bf16_keys__"].tolist()) if "__bf16_keys__" in data.files else set()
        out: dict[str, np.ndarray] = {}
        for k in data.files:
            if k == "__bf16_keys__":
                continue
            arr = data[k]
            if k in bf16_keys:
                out[k] = jnp.asarray(arr).view(jnp.bfloat16)
            else:
                out[k] = arr
    return out


# ---------------------------------------------------------------------------
# Checkpoint Manager with Local / Google Drive / HF Hub / Kaggle Sync
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last: int = 2,
        sync_backend: str = "none",
        remote_target: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self.sync_backend = sync_backend
        self.remote_target = remote_target
        self._hf_tokenizer_synced = False

    def _step_dir(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step:07d}"

    def _latest_pointer(self) -> Path:
        return self.checkpoint_dir / "latest.json"

    def find_latest_local(self) -> Optional[Path]:
        ptr = self._latest_pointer()
        if ptr.exists():
            info = json.loads(ptr.read_text(encoding="utf-8"))
            cand = self.checkpoint_dir / info["step_dir"]
            if cand.is_dir():
                return cand
        step_dirs = sorted(p for p in self.checkpoint_dir.glob("step_*") if p.is_dir())
        return step_dirs[-1] if step_dirs else None

    def save(
        self,
        step: int,
        params: dict,
        opt_state: dict,
        trainer_state: dict,
        tokenizer_dir: Optional[str | Path] = None,
        metrics_path: Optional[str | Path] = None,
    ) -> Path:
        """Atomically save a topology-agnostic checkpoint, prune old ones, and sync remotely."""
        import time

        t0 = time.perf_counter()
        target_dir = self._step_dir(step)
        tmp_dir = self.checkpoint_dir / f".tmp_step_{step:07d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        flat_params = flatten_pytree(params)
        _save_arrays(tmp_dir / "model", flat_params, prefer_safetensors=True)
        del flat_params

        flat_opt = flatten_pytree(opt_state)
        _save_arrays(tmp_dir / "opt_state", flat_opt, prefer_safetensors=False)
        del flat_opt

        payload = {
            **trainer_state,
            "step": int(step),
            "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        (tmp_dir / "trainer_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        tmp_dir.rename(target_dir)

        # Persist tokenizer once at checkpoint root so fresh VMs restore identically
        if tokenizer_dir is not None:
            tok_src = Path(tokenizer_dir)
            tok_dst = self.checkpoint_dir / "tokenizer"
            if tok_src.is_dir() and tok_src.resolve() != tok_dst.resolve():
                if tok_dst.exists():
                    shutil.rmtree(tok_dst)
                shutil.copytree(tok_src, tok_dst)

        self._latest_pointer().write_text(
            json.dumps({"step": int(step), "step_dir": target_dir.name}, indent=2) + "\n",
            encoding="utf-8",
        )
        self._prune_old()

        size_gb = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file()) / (1024**3)
        print(f"[Checkpoint] Saved {target_dir.name} ({size_gb:.2f} GB) in {time.perf_counter() - t0:.1f}s", flush=True)

        if self.sync_backend != "none" and self.remote_target:
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    self.sync_push(target_dir, metrics_path=metrics_path)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    wait_s = 5 * (attempt + 1)
                    print(
                        f"[Sync:{self.sync_backend}] Push attempt {attempt + 1}/3 failed ({exc}); "
                        + (f"retrying in {wait_s}s..." if attempt < 2 else "keeping local checkpoint and continuing."),
                        flush=True,
                    )
                    if attempt < 2:
                        time.sleep(wait_s)
            if last_err is not None:
                print(f"[Sync:{self.sync_backend}] WARNING: remote sync failed for {target_dir.name} (local copy is safe).", flush=True)

        return target_dir

    def _prune_old(self) -> None:
        step_dirs = sorted(p for p in self.checkpoint_dir.glob("step_*") if p.is_dir())
        for old in step_dirs[: -self.keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    def _resolve_hf_repo_id(self, api) -> str:
        assert self.remote_target is not None
        if self.remote_target.startswith("auto/"):
            user = api.whoami()["name"]
            return f"{user}/{self.remote_target.split('/', 1)[1]}"
        return self.remote_target

    def load_latest(
        self,
        params_template: dict,
        opt_template: dict,
        param_shardings,
        opt_shardings,
        pull_remote: bool = True,
    ) -> Optional[tuple[dict, dict, dict]]:
        """
        Restore the latest checkpoint (pulling from remote backend first if enabled)
        and re-shard every leaf onto `param_shardings` / `opt_shardings`.
        """
        if pull_remote and self.sync_backend != "none" and self.remote_target:
            try:
                self.sync_pull()
            except Exception as exc:
                print(f"[Checkpoint] Remote pull skipped/empty ({exc})")

        ckpt_dir = self.find_latest_local()
        if ckpt_dir is None:
            return None

        state = json.loads((ckpt_dir / "trainer_state.json").read_text(encoding="utf-8"))
        flat_params = _load_arrays(ckpt_dir / "model")
        flat_opt = _load_arrays(ckpt_dir / "opt_state")

        params_host = unflatten_into_template(flat_params, params_template)
        opt_host = unflatten_into_template(flat_opt, opt_template)

        params = shard_pytree(params_host, param_shardings)
        opt_state = shard_pytree(opt_host, opt_shardings)
        print(f"[Checkpoint] Restored step {state['step']} from {ckpt_dir}")
        return params, opt_state, state

    # -----------------------------------------------------------------------
    # Remote Sync Backends (Google Drive / Local, HuggingFace Hub, Kaggle Dataset)
    # -----------------------------------------------------------------------

    def sync_push(self, step_dir: Path, metrics_path: Optional[str | Path] = None) -> None:
        assert self.remote_target is not None
        if self.sync_backend == "local":
            dst_root = Path(self.remote_target)
            dst_root.mkdir(parents=True, exist_ok=True)
            dst_latest = dst_root / "latest"
            tmp_dst = dst_root / ".tmp_latest"
            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            shutil.copytree(step_dir, tmp_dst)
            if dst_latest.exists():
                shutil.rmtree(dst_latest)
            tmp_dst.rename(dst_latest)
            if (self.checkpoint_dir / "tokenizer").is_dir():
                tok_dst = dst_root / "tokenizer"
                if tok_dst.exists():
                    shutil.rmtree(tok_dst)
                shutil.copytree(self.checkpoint_dir / "tokenizer", tok_dst)
            if metrics_path and Path(metrics_path).exists():
                shutil.copy2(metrics_path, dst_root / "metrics.jsonl")
            print(f"[Sync:local] Synced {step_dir.name} -> {dst_root}", flush=True)

        elif self.sync_backend == "hf":
            import time
            from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore

            t0 = time.perf_counter()
            token = _get_secret("HF_TOKEN") or _get_secret("HUGGINGFACE_TOKEN")
            api = HfApi(token=token)
            repo_id = self._resolve_hf_repo_id(api)
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            ops = [
                CommitOperationAdd(
                    path_in_repo=f"latest/{p.relative_to(step_dir).as_posix()}",
                    path_or_fileobj=str(p),
                )
                for p in sorted(step_dir.rglob("*"))
                if p.is_file()
            ]
            tok_dir = self.checkpoint_dir / "tokenizer"
            sync_tok = tok_dir.is_dir() and not self._hf_tokenizer_synced
            if sync_tok:
                for p in sorted(tok_dir.rglob("*")):
                    if p.is_file():
                        ops.append(
                            CommitOperationAdd(
                                path_in_repo=f"tokenizer/{p.relative_to(tok_dir).as_posix()}",
                                path_or_fileobj=str(p),
                            )
                        )
            if metrics_path and Path(metrics_path).exists():
                ops.append(
                    CommitOperationAdd(
                        path_in_repo="metrics.jsonl",
                        path_or_fileobj=str(metrics_path),
                    )
                )
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=ops,
                commit_message=f"Checkpoint {step_dir.name}",
            )
            if sync_tok:
                self._hf_tokenizer_synced = True
            print(
                f"[Sync:hf] Uploaded {step_dir.name} ({len(ops)} files, 1 atomic commit) -> "
                f"hf://{repo_id}/latest in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

        elif self.sync_backend == "kaggle":
            stage = self.checkpoint_dir / ".kaggle_stage"
            if stage.exists():
                shutil.rmtree(stage)
            shutil.copytree(step_dir, stage)
            if (self.checkpoint_dir / "tokenizer").is_dir():
                shutil.copytree(self.checkpoint_dir / "tokenizer", stage / "tokenizer")
            if metrics_path and Path(metrics_path).exists():
                shutil.copy2(metrics_path, stage / "metrics.jsonl")
            meta = {
                "title": self.remote_target.split("/")[-1],
                "id": self.remote_target,
                "licenses": [{"name": "CC0-1.0"}],
            }
            (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            ver = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(stage), "-m", f"Checkpoint {step_dir.name}", "-r", "zip"],
                capture_output=True,
                text=True,
            )
            if ver.returncode != 0:
                subprocess.run(
                    ["kaggle", "datasets", "create", "-p", str(stage), "-r", "zip"],
                    check=True,
                )
            shutil.rmtree(stage, ignore_errors=True)
            print(f"[Sync:kaggle] Pushed {step_dir.name} -> kaggle://{self.remote_target}", flush=True)

    def sync_pull(self) -> bool:
        assert self.remote_target is not None
        if self.sync_backend == "local":
            src_root = Path(self.remote_target)
            src_latest = src_root / "latest"
            if not (src_latest / "trainer_state.json").exists():
                return False
            state = json.loads((src_latest / "trainer_state.json").read_text(encoding="utf-8"))
            step = int(state["step"])
            dst = self._step_dir(step)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_latest, dst)
            if (src_root / "tokenizer").is_dir():
                tok_dst = self.checkpoint_dir / "tokenizer"
                if tok_dst.exists():
                    shutil.rmtree(tok_dst)
                shutil.copytree(src_root / "tokenizer", tok_dst)
            if (src_root / "metrics.jsonl").is_file() and not (self.checkpoint_dir / "metrics.jsonl").exists():
                shutil.copy2(src_root / "metrics.jsonl", self.checkpoint_dir / "metrics.jsonl")
            self._latest_pointer().write_text(
                json.dumps({"step": step, "step_dir": dst.name}, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[Sync:local] Pulled step {step} from {src_root}", flush=True)
            return True

        if self.sync_backend == "hf":
            from huggingface_hub import HfApi, snapshot_download  # type: ignore

            token = _get_secret("HF_TOKEN") or _get_secret("HUGGINGFACE_TOKEN")
            api = HfApi(token=token)
            repo_id = self._resolve_hf_repo_id(api)
            dl_dir = self.checkpoint_dir / ".hf_pull"
            if dl_dir.exists():
                shutil.rmtree(dl_dir)
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(dl_dir),
                allow_patterns=["latest/*", "tokenizer/*", "metrics.jsonl"],
                token=token,
            )
            src_latest = dl_dir / "latest"
            if not (src_latest / "trainer_state.json").exists():
                shutil.rmtree(dl_dir, ignore_errors=True)
                return False
            state = json.loads((src_latest / "trainer_state.json").read_text(encoding="utf-8"))
            step = int(state["step"])
            dst = self._step_dir(step)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src_latest), str(dst))
            if (dl_dir / "tokenizer").is_dir():
                tok_dst = self.checkpoint_dir / "tokenizer"
                if tok_dst.exists():
                    shutil.rmtree(tok_dst)
                shutil.move(str(dl_dir / "tokenizer"), str(tok_dst))
            if (dl_dir / "metrics.jsonl").is_file() and not (self.checkpoint_dir / "metrics.jsonl").exists():
                shutil.move(str(dl_dir / "metrics.jsonl"), str(self.checkpoint_dir / "metrics.jsonl"))
            self._latest_pointer().write_text(
                json.dumps({"step": step, "step_dir": dst.name}, indent=2) + "\n", encoding="utf-8"
            )
            shutil.rmtree(dl_dir, ignore_errors=True)
            print(f"[Sync:hf] Pulled step {step} from hf://{repo_id}", flush=True)
            return True

        if self.sync_backend == "kaggle":
            dl_dir = self.checkpoint_dir / ".kaggle_pull"
            if dl_dir.exists():
                shutil.rmtree(dl_dir)
            dl_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.remote_target, "-p", str(dl_dir), "--unzip"],
                check=True,
            )
            if not (dl_dir / "trainer_state.json").exists():
                return False
            state = json.loads((dl_dir / "trainer_state.json").read_text(encoding="utf-8"))
            step = int(state["step"])
            dst = self._step_dir(step)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(dl_dir, dst)
            if (dl_dir / "tokenizer").is_dir():
                tok_dst = self.checkpoint_dir / "tokenizer"
                if tok_dst.exists():
                    shutil.rmtree(tok_dst)
                shutil.copytree(dl_dir / "tokenizer", tok_dst)
            self._latest_pointer().write_text(
                json.dumps({"step": step, "step_dir": dst.name}, indent=2) + "\n", encoding="utf-8"
            )
            shutil.rmtree(dl_dir, ignore_errors=True)
            print(f"[Sync:kaggle] Pulled step {step} from kaggle://{self.remote_target}")
            return True

        return False
