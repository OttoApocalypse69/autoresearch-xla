"""
Autonomous research loop for autoresearch-xla (Google Colab, Kaggle, and local).

Works identically with GPUs (T4, 2xT4, P100, L4, A100, H100) and TPUs (v2-8, v3-8,
v5e-1, v5e-8, v6e-1). Critically, `agent.py` NEVER imports JAX in the parent process,
so every `train.py` run executes in a clean isolated subprocess with zero Cloud TPU
`libtpu_lockfile` conflicts and full VRAM/HBM cleanup between experiments.

Supported LLM providers (pure `requests`, zero SDK bloat):
  - gemini        (Google AI Studio — GEMINI_API_KEY)
  - anthropic     (Claude Messages API — ANTHROPIC_API_KEY)
  - openai        (OpenAI Chat Completions — OPENAI_API_KEY)
  - openrouter    (OpenRouter — OPENROUTER_API_KEY)
  - openai-compat (Any OpenAI-compatible endpoint via --base-url: Ollama, vLLM, Groq, etc.)

Usage:
    python agent.py --provider gemini --tag nov24-tpu --max-experiments 50
    python agent.py --provider anthropic --model claude-sonnet-4-6 --sync-dir /content/drive/MyDrive/autoresearch
    python agent.py --provider openrouter --model google/gemini-2.5-pro --max-hours 8.5
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

# ---------------------------------------------------------------------------
# Provider Defaults & Secret Discovery (Colab / Kaggle / Env)
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS = {
    "gemini": {
        "env": "GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "model": "gpt-4.1",
        "base_url": "https://api.openai.com/v1",
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "openai-compat": {
        "env": "LLM_API_KEY",
        "model": "default",
        "base_url": "http://localhost:11434/v1",
    },
}

RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


def get_secret(name: str) -> Optional[str]:
    """Read a secret from environment, Google Colab userdata, or Kaggle UserSecretsClient."""
    val = os.environ.get(name)
    if val:
        return val.strip()

    # Google Colab Secrets
    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get(name)
        if val:
            return str(val).strip()
    except Exception:
        pass

    # Kaggle Notebook Secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = UserSecretsClient().get_secret(name)
        if val:
            return str(val).strip()
    except Exception:
        pass

    return None


def resolve_provider_and_keys(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[str, list[str], str, Optional[str]]:
    """Resolve provider name, API key list (supports comma-separated rotation), model, and base URL."""
    if provider == "auto":
        for cand in ("gemini", "anthropic", "openai", "openrouter"):
            if get_secret(PROVIDER_DEFAULTS[cand]["env"]):
                provider = cand
                break
        else:
            raise RuntimeError(
                "No LLM API key found. Set GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "or OPENROUTER_API_KEY (via env var, Colab Secrets, or Kaggle Secrets)."
            )

    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider {provider!r}. Choose from {list(PROVIDER_DEFAULTS)}")

    cfg = PROVIDER_DEFAULTS[provider]
    raw_key = api_key or get_secret(cfg["env"]) or get_secret("LLM_API_KEY")
    if not raw_key and provider != "openai-compat":
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            f"Set {cfg['env']} (env var, Colab Secrets, or Kaggle Secrets) or pass --api-key."
        )
    keys = [k.strip() for k in (raw_key or "not-needed").split(",") if k.strip()]
    resolved_model = model or cfg["model"]
    resolved_url = base_url or cfg.get("base_url")
    return provider, keys, resolved_model, resolved_url


# ---------------------------------------------------------------------------
# Multi-Provider LLM Client (Pure Requests + Retry + Key Rotation)
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal, dependency-free multi-provider LLM client with backoff & key rotation."""

    def __init__(
        self,
        provider: str,
        api_keys: list[str],
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout: int = 180,
    ):
        self.provider = provider
        self.api_keys = api_keys
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._key_idx = 0

    def _next_key(self) -> str:
        key = self.api_keys[self._key_idx % len(self.api_keys)]
        self._key_idx += 1
        return key

    def complete(self, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            key = self._next_key()
            try:
                if self.provider == "gemini":
                    return self._call_gemini(key, system_prompt, user_prompt)
                if self.provider == "anthropic":
                    return self._call_anthropic(key, system_prompt, user_prompt)
                return self._call_openai_compat(key, system_prompt, user_prompt)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                body = exc.response.text[:400] if exc.response is not None else ""
                last_err = RuntimeError(f"HTTP {status}: {body}")
                if status in (429, 500, 502, 503, 504):
                    wait = min(2**attempt * 2.0, 60.0)
                    print(f"  [LLM] Transient HTTP {status}, retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                raise last_err from exc
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_err = exc
                wait = min(2**attempt * 2.0, 30.0)
                print(f"  [LLM] Error ({exc}), retrying in {wait:.0f}s...")
                time.sleep(wait)
        raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_err}")

    def _call_gemini(self, api_key: str, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(p["text"] for p in parts if "text" in p)

    def _call_anthropic(self, api_key: str, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return "\n".join(b["text"] for b in data["content"] if b.get("type") == "text")

    def _call_openai_compat(self, api_key: str, system_prompt: str, user_prompt: str) -> str:
        assert self.base_url is not None
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/autoresearch-xla"
            headers["X-Title"] = "autoresearch-xla"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Proposal Parsing, Safe Patch Application & Static Guardrails
# ---------------------------------------------------------------------------

@dataclass
class ProposedEdit:
    hypothesis: str
    description: str
    blocks: list[tuple[str, str]]  # list of (search_str, replace_str)


_TAG_RE = lambda tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
_DIFF_RE = re.compile(
    r"<<<<<<<\s*SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)


def parse_llm_proposal(text: str) -> ProposedEdit:
    """Extract hypothesis, description, and SEARCH/REPLACE diff blocks from LLM response."""
    hyp_m = _TAG_RE("hypothesis").search(text)
    desc_m = _TAG_RE("description").search(text)
    hypothesis = hyp_m.group(1).strip() if hyp_m else ""
    description = desc_m.group(1).strip() if desc_m else "experimental change"
    # Sanitize single-line description for TSV & git commit
    description = re.sub(r"[\t\r\n]+", " ", description).strip()[:120] or "experimental change"

    blocks = [(m.group(1), m.group(2)) for m in _DIFF_RE.finditer(text)]
    if not blocks:
        raise ValueError(
            "No valid SEARCH/REPLACE blocks found. Use exact markers:\n"
            "<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE"
        )
    return ProposedEdit(hypothesis=hypothesis, description=description, blocks=blocks)


def apply_edits_safely(original_code: str, blocks: list[tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks sequentially and validate Python syntax & eval integrity."""
    updated = original_code
    for idx, (search, replace) in enumerate(blocks, start=1):
        count = updated.count(search)
        if count == 0:
            preview = search.splitlines()[0][:80] if search.splitlines() else ""
            raise ValueError(
                f"Edit #{idx} SEARCH block not found verbatim in train.py (starts with: {preview!r})."
            )
        if count > 1:
            raise ValueError(
                f"Edit #{idx} SEARCH block matched {count} times in train.py; include more surrounding context."
            )
        updated = updated.replace(search, replace, 1)

    if updated == original_code:
        raise ValueError("Edits produced zero changes to train.py.")

    # Syntax validation
    try:
        tree = ast.parse(updated, filename="train.py")
    except SyntaxError as exc:
        raise ValueError(f"Edited train.py has a Python SyntaxError at line {exc.lineno}: {exc.msg}") from exc

    # Guardrail: verify evaluate_bpb from prepare.py is still imported & called
    has_eval_call = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "evaluate_bpb")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_bpb")
        )
        for node in ast.walk(tree)
    )
    if not has_eval_call:
        raise ValueError("Guardrail violation: train.py must still call prepare.evaluate_bpb().")

    return updated


# ---------------------------------------------------------------------------
# Git Helpers, Subprocess Runner & Metric Parser
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    ok: bool
    val_bpb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_M: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    depth: int = 0
    accelerator: str = ""
    error_tail: str = ""
    timed_out: bool = False


def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_git_branch(repo_dir: Path, tag: str) -> str:
    """Initialize git repo if needed (with Colab/Kaggle-safe identity) and checkout experiment branch."""
    if not (repo_dir / ".git").exists():
        _git(repo_dir, "init")
    # Fresh Colab/Kaggle VMs don't have git user.name / user.email configured
    if _git(repo_dir, "config", "user.name", check=False).returncode != 0:
        _git(repo_dir, "config", "user.name", "autoresearch-agent")
    if _git(repo_dir, "config", "user.email", check=False).returncode != 0:
        _git(repo_dir, "config", "user.email", "autoresearch@local")

    # Make initial commit if repo has no commits yet
    if _git(repo_dir, "rev-parse", "--verify", "HEAD", check=False).returncode != 0:
        _git(repo_dir, "add", "-A")
        _git(repo_dir, "commit", "-m", "initial commit")

    branch = f"autoresearch/{tag}"
    existing = _git(repo_dir, "rev-parse", "--verify", branch, check=False).returncode == 0
    if existing:
        _git(repo_dir, "checkout", branch)
    else:
        _git(repo_dir, "checkout", "-b", branch)
    return branch


def current_commit(repo_dir: Path) -> str:
    return _git(repo_dir, "rev-parse", "--short=7", "HEAD").stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_hardware_subprocess(repo_dir: Path, python_exe: str) -> str:
    """
    Detect GPU/TPU in a short-lived subprocess so the parent agent process NEVER
    imports JAX or holds a Cloud TPU device lock!
    """
    code = (
        "from prepare import detect_hardware; h = detect_hardware(); "
        "print(f'{h.device_kind} x{h.num_devices} "
        "(platform={h.platform}, dtype={h.compute_dtype}, "
        "hbm={h.hbm_per_device_gb:.1f}GB/dev, peak={h.peak_tflops_total:.1f} TFLOPS, "
        "default_depth={h.default_depth}, default_batch={h.default_total_batch_size})')"
    )
    res = subprocess.run(
        [python_exe, "-c", code],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip().splitlines()[-1]
    return f"unknown accelerator (probe stderr: {res.stderr.strip()[:200]})"


def parse_run_log(log_text: str) -> RunMetrics:
    """Parse Karpathy-format summary metrics from run.log."""
    metrics: dict[str, str] = {}
    for line in log_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in {
                "val_bpb",
                "training_seconds",
                "total_seconds",
                "peak_vram_mb",
                "mfu_percent",
                "total_tokens_M",
                "num_steps",
                "num_params_M",
                "depth",
                "accelerator",
            }:
                metrics[k] = v.strip()

    if "val_bpb" not in metrics:
        tail = "\n".join(log_text.splitlines()[-50:])
        return RunMetrics(ok=False, error_tail=tail)

    try:
        val_bpb = float(metrics["val_bpb"])
        if not (val_bpb > 0.0 and val_bpb < 1e6):
            raise ValueError(f"Non-finite or invalid val_bpb: {val_bpb}")
        return RunMetrics(
            ok=True,
            val_bpb=val_bpb,
            training_seconds=float(metrics.get("training_seconds", 0.0)),
            total_seconds=float(metrics.get("total_seconds", 0.0)),
            peak_vram_mb=float(metrics.get("peak_vram_mb", 0.0)),
            mfu_percent=float(metrics.get("mfu_percent", 0.0)),
            total_tokens_M=float(metrics.get("total_tokens_M", 0.0)),
            num_steps=int(float(metrics.get("num_steps", 0))),
            num_params_M=float(metrics.get("num_params_M", 0.0)),
            depth=int(float(metrics.get("depth", 0))),
            accelerator=metrics.get("accelerator", ""),
        )
    except Exception as exc:
        tail = "\n".join(log_text.splitlines()[-50:])
        return RunMetrics(ok=False, error_tail=f"Parse error: {exc}\n{tail}")


def run_training_subprocess(
    repo_dir: Path,
    python_exe: str,
    timeout_s: int,
    log_path: Path,
) -> RunMetrics:
    """Run `train.py` in an isolated process group, capturing all logs and enforcing timeout."""
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [python_exe, "train.py"],
            cwd=str(repo_dir),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                proc.wait(timeout=5)
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            raise

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        tail = "\n".join(log_text.splitlines()[-30:])
        return RunMetrics(
            ok=False,
            timed_out=True,
            error_tail=f"TIMEOUT after {timeout_s}s\n{tail}",
        )
    if proc.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-50:])
        return RunMetrics(ok=False, error_tail=f"Exit code {proc.returncode}\n{tail}")

    return parse_run_log(log_text)


# ---------------------------------------------------------------------------
# Results TSV, Rich JSONL Logging, Sync & Plotting
# ---------------------------------------------------------------------------

def load_results_tsv(tsv_path: Path) -> list[dict]:
    if not tsv_path.exists():
        return []
    rows = []
    for i, line in enumerate(tsv_path.read_text(encoding="utf-8").splitlines()):
        if i == 0 and line.startswith("commit\t"):
            continue
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            rows.append(
                {
                    "commit": parts[0],
                    "val_bpb": float(parts[1]),
                    "memory_gb": float(parts[2]),
                    "status": parts[3],
                    "description": parts[4],
                }
            )
    return rows


def append_result_tsv(
    tsv_path: Path, commit: str, val_bpb: float, peak_vram_mb: float, status: str, description: str
) -> None:
    if not tsv_path.exists():
        tsv_path.write_text(RESULTS_HEADER, encoding="utf-8")
    desc_clean = re.sub(r"[\t\r\n]+", " ", description).strip()
    mem_gb = peak_vram_mb / 1024.0 if peak_vram_mb > 0 else 0.0
    line = f"{commit}\t{val_bpb:.6f}\t{mem_gb:.1f}\t{status}\t{desc_clean}\n"
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(line)


def append_jsonl(jsonl_path: Path, record: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def plot_progress(tsv_path: Path, out_png: Path) -> bool:
    """Generate Karpathy-style autoresearch progress staircase chart."""
    rows = load_results_tsv(tsv_path)
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    valid = [(i, r) for i, r in enumerate(rows) if r["status"] in ("keep", "discard") and r["val_bpb"] > 0]
    crashes = [i for i, r in enumerate(rows) if r["status"] == "crash"]
    kept = [(i, r) for i, r in valid if r["status"] == "keep"]
    disc = [(i, r) for i, r in valid if r["status"] == "discard"]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if disc:
        ax.scatter(
            [i for i, _ in disc],
            [r["val_bpb"] for _, r in disc],
            color="#94a3b8",
            alpha=0.55,
            s=36,
            label=f"Discarded ({len(disc)})",
            zorder=2,
        )
    if kept:
        kx = [i for i, _ in kept]
        ky = [r["val_bpb"] for _, r in kept]
        ax.step(kx, ky, where="post", color="#22c55e", linewidth=2.2, zorder=3)
        ax.scatter(kx, ky, color="#22c55e", edgecolor="#052e16", s=64, label=f"Kept ({len(kept)})", zorder=4)
        for idx_pos, (x_i, r) in enumerate(kept):
            if idx_pos == 0 or idx_pos == len(kept) - 1 or len(kept) <= 10 or idx_pos % max(1, len(kept) // 8) == 0:
                short_lbl = r["description"][:28] + ("…" if len(r["description"]) > 28 else "")
                ax.annotate(
                    short_lbl,
                    (x_i, r["val_bpb"]),
                    textcoords="offset points",
                    xytext=(5, 7),
                    fontsize=7.5,
                    color="#dcfce7",
                    alpha=0.92,
                )
    if crashes and valid:
        ymax = max(r["val_bpb"] for _, r in valid)
        ax.scatter(
            crashes,
            [ymax] * len(crashes),
            marker="x",
            color="#ef4444",
            s=45,
            label=f"Crash ({len(crashes)})",
            zorder=3,
        )

    best_bpb = min((r["val_bpb"] for _, r in kept), default=0.0)
    base_bpb = kept[0][1]["val_bpb"] if kept else 0.0
    ax.set_title(
        f"autoresearch-xla Progress | Experiments: {len(rows)} | "
        f"Baseline: {base_bpb:.4f} → Best: {best_bpb:.4f} val_bpb",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Experiment #", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("val_bpb (lower is better)", color="#cbd5e1", fontsize=10)
    ax.tick_params(colors="#cbd5e1")
    ax.grid(color="#334155", linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.legend(facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", loc="upper right")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def sync_to_dir(repo_dir: Path, sync_dir: Optional[Path]) -> None:
    if sync_dir is None:
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("results.tsv", "experiments.jsonl", "train.py", "best_train.py", "progress.png", "program.md"):
        src = repo_dir / fname
        if src.exists():
            shutil.copy2(src, sync_dir / fname)


def restore_from_sync_dir(repo_dir: Path, sync_dir: Optional[Path]) -> bool:
    """Restore previous best state from Google Drive / Kaggle persistence directory if present."""
    if sync_dir is None or not (sync_dir / "results.tsv").exists():
        return False
    restored = False
    for fname in ("results.tsv", "experiments.jsonl", "best_train.py"):
        src = sync_dir / fname
        if src.exists():
            shutil.copy2(src, repo_dir / fname)
            restored = True
    best_src = sync_dir / "best_train.py"
    if best_src.exists():
        shutil.copy2(best_src, repo_dir / "train.py")
    return restored


# ---------------------------------------------------------------------------
# Prompt Builder (Context-Window-Safe Across 100+ Iterations)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous machine learning researcher running Karpathy-style autoresearch in pure JAX/XLA on Google Colab / Kaggle GPUs and TPUs.

Your single objective is to achieve the LOWEST possible `val_bpb` (validation bits-per-byte) within the fixed wall-clock training budget by editing ONLY `train.py`.

Strict output format — you MUST respond using these exact tags:
<hypothesis>
Concise 1-3 sentence scientific hypothesis explaining why this modification should improve val_bpb on this specific accelerator.
</hypothesis>
<description>
Single concise line (<= 80 chars, no tabs) summarizing the change for results.tsv and git commit.
</description>
<edit>
<<<<<<< SEARCH
exact contiguous lines currently in train.py
=======
replacement lines
>>>>>>> REPLACE
</edit>

Rules:
- You may include 1 to 4 <edit> blocks if needed.
- Every SEARCH block MUST match `train.py` character-for-character (including indentation).
- Do NOT modify `prepare.py`, do NOT add third-party imports, and do NOT bypass `evaluate_bpb()`.
- Keep array shapes static inside `train_step`, preserve SPMD multi-device sharding, and respect the Simplicity Criterion.
- Propose ONE coherent, testable experimental idea per iteration (not 5 unrelated changes at once).
"""


def build_research_prompt(
    program_md: str,
    train_code: str,
    rows: list[dict],
    hardware_desc: str,
    extra_feedback: str = "",
) -> str:
    kept = [r for r in rows if r["status"] == "keep"]
    best_bpb = min((r["val_bpb"] for r in kept), default=float("inf"))
    n_keep = len(kept)
    n_disc = sum(1 for r in rows if r["status"] == "discard")
    n_crash = sum(1 for r in rows if r["status"] == "crash")

    # Compact memory: ALL kept improvements + last 25 experiments so context stays bounded even at run 200
    kept_lines = "\n".join(
        f"  {r['commit']} | val_bpb={r['val_bpb']:.6f} | mem={r['memory_gb']:.1f}GB | {r['description']}"
        for r in kept
    )
    recent_lines = "\n".join(
        f"  #{i + 1:03d} [{r['status']:7s}] {r['commit']} | val_bpb={r['val_bpb']:.6f} | mem={r['memory_gb']:.1f}GB | {r['description']}"
        for i, r in list(enumerate(rows))[-25:]
    )

    parts = [
        "## Hardware Environment",
        hardware_desc,
        "",
        "## Research Program (`program.md`)",
        program_md.strip(),
        "",
        f"## Experiment History ({len(rows)} total: {n_keep} kept, {n_disc} discarded, {n_crash} crashed | current best val_bpb = {best_bpb:.6f})",
        "### Winning Lineage (kept commits):",
        kept_lines or "  (none yet)",
        "",
        "### Recent Experiments (avoid repeating failed ideas):",
        recent_lines or "  (none yet)",
        "",
        "## Current `train.py`",
        "```python",
        train_code,
        "```",
    ]
    if extra_feedback:
        parts.extend(["", "## Feedback From Previous Attempt (fix this)", extra_feedback])
    parts.extend(["", "Propose the next highest-leverage experiment for `train.py`."])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Autonomous Research Loop
# ---------------------------------------------------------------------------

def run_autoresearch_loop(
    repo_dir: Path,
    tag: str,
    llm_complete: Callable[[str, str], str],
    python_exe: str = sys.executable,
    max_experiments: int = 100,
    max_hours: Optional[float] = None,
    run_timeout: Optional[int] = None,
    sync_dir: Optional[Path] = None,
    resume: bool = True,
    fix_crashes: bool = True,
    git_push: bool = False,
) -> list[dict]:
    """Execute the full Karpathy ratchet loop autonomously."""
    repo_dir = repo_dir.resolve()
    train_path = repo_dir / "train.py"
    prepare_path = repo_dir / "prepare.py"
    program_path = repo_dir / "program.md"
    tsv_path = repo_dir / "results.tsv"
    jsonl_path = repo_dir / "experiments.jsonl"
    log_path = repo_dir / "run.log"
    png_path = repo_dir / "progress.png"

    assert train_path.exists() and prepare_path.exists(), f"Missing train.py or prepare.py in {repo_dir}"
    prepare_hash = file_sha256(prepare_path)

    # 1. Restore saved progress from Google Drive / Kaggle sync_dir if resuming
    if resume and sync_dir is not None:
        if restore_from_sync_dir(repo_dir, sync_dir):
            print(f"[Agent] Restored previous session state from {sync_dir}")

    # 2. Setup git branch
    branch = ensure_git_branch(repo_dir, tag)
    if _git(repo_dir, "status", "--porcelain", "train.py").stdout.strip():
        _git(repo_dir, "add", "train.py")
        _git(repo_dir, "commit", "-m", "restore best train.py from sync dir")

    # 3. Probe hardware in isolated subprocess (TPU lock-safe!)
    hardware_desc = probe_hardware_subprocess(repo_dir, python_exe)
    print(f"[Agent] Branch:    {branch} (HEAD={current_commit(repo_dir)})")
    print(f"[Agent] Hardware:  {hardware_desc}")
    if sync_dir:
        print(f"[Agent] Sync dir:  {sync_dir}")

    # Compute safe default timeout based on TIME_BUDGET env var
    time_budget_s = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "300"))
    timeout_s = run_timeout or max(time_budget_s * 2 + 180, 600)
    print(f"[Agent] Timeout:   {timeout_s}s per experiment (TIME_BUDGET={time_budget_s}s)")

    t_loop_start = time.time()
    rows = load_results_tsv(tsv_path)
    kept_bpbs = [r["val_bpb"] for r in rows if r["status"] == "keep" and r["val_bpb"] > 0]
    best_bpb = min(kept_bpbs) if kept_bpbs else float("inf")

    # 4. Run Baseline if no results exist yet
    if not rows:
        print("\n=== Experiment 0: Baseline ===")
        commit = current_commit(repo_dir)
        res = run_training_subprocess(repo_dir, python_exe, timeout_s, log_path)
        if not res.ok:
            print(f"[Agent] Baseline FAILED!\n{res.error_tail}")
            append_result_tsv(tsv_path, commit, 0.0, 0.0, "crash", "baseline failed")
            append_jsonl(jsonl_path, {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(), "iter": 0, "commit": commit, "status": "crash", "description": "baseline", **asdict(res)})
            sync_to_dir(repo_dir, sync_dir)
            raise RuntimeError("Baseline training run failed; inspect run.log before running the agent loop.")

        best_bpb = res.val_bpb
        append_result_tsv(tsv_path, commit, res.val_bpb, res.peak_vram_mb, "keep", "baseline")
        append_jsonl(
            jsonl_path,
            {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "iter": 0,
                "commit": commit,
                "status": "keep",
                "description": "baseline",
                "hypothesis": "Unmodified baseline",
                **asdict(res),
            },
        )
        shutil.copy2(train_path, repo_dir / "best_train.py")
        plot_progress(tsv_path, png_path)
        sync_to_dir(repo_dir, sync_dir)
        print(
            f"[Baseline] val_bpb={res.val_bpb:.6f} | peak_mem={res.peak_vram_mb/1024:.2f}GB | "
            f"mfu={res.mfu_percent:.1f}% | steps={res.num_steps} | tokens={res.total_tokens_M:.1f}M"
        )
        rows = load_results_tsv(tsv_path)

    # 5. Autonomous Ratchet Loop
    completed_this_session = 0
    while completed_this_session < max_experiments:
        if max_hours is not None:
            elapsed_h = (time.time() - t_loop_start) / 3600.0
            if elapsed_h >= max_hours:
                print(f"\n[Agent] Reached max_hours={max_hours:.2f}h safety limit. Cleanly stopping.")
                break

        # Verify immutable prepare.py integrity
        if file_sha256(prepare_path) != prepare_hash:
            raise RuntimeError("Guardrail triggered: prepare.py was modified externally!")

        iter_num = len(rows)
        completed_this_session += 1
        print(f"\n=== Experiment #{iter_num} (best val_bpb={best_bpb:.6f}) ===")

        program_md = program_path.read_text(encoding="utf-8") if program_path.exists() else ""
        base_code = train_path.read_text(encoding="utf-8")

        # Propose and validate edit (with up to 2 edit-format retries)
        proposal: Optional[ProposedEdit] = None
        new_code: Optional[str] = None
        edit_feedback = ""
        for edit_try in range(3):
            prompt = build_research_prompt(program_md, base_code, rows, hardware_desc, edit_feedback)
            try:
                raw_resp = llm_complete(SYSTEM_PROMPT, prompt)
                proposal = parse_llm_proposal(raw_resp)
                new_code = apply_edits_safely(base_code, proposal.blocks)
                break
            except Exception as exc:
                edit_feedback = f"Attempt {edit_try + 1} rejected: {exc}\nFix the SEARCH/REPLACE block carefully."
                print(f"  [Edit retry {edit_try + 1}/3] {exc}")

        if proposal is None or new_code is None:
            print("  [Agent] Could not produce a valid edit after 3 tries; skipping iteration.")
            continue

        print(f"  [Idea] {proposal.description}")
        if proposal.hypothesis:
            print(f"  [Hypothesis] {proposal.hypothesis[:200]}")

        # Commit candidate change
        train_path.write_text(new_code, encoding="utf-8")
        _git(repo_dir, "add", "train.py")
        _git(repo_dir, "commit", "-m", proposal.description)
        cand_commit = current_commit(repo_dir)

        # Execute training subprocess
        res = run_training_subprocess(repo_dir, python_exe, timeout_s, log_path)

        # Optional 1-shot crash self-repair (for trivial typos/shape bugs)
        if not res.ok and not res.timed_out and fix_crashes:
            print(f"  [Crash] Trying 1 automated bug-fix pass...")
            fix_feedback = (
                f"Your previous edit ('{proposal.description}') crashed during execution.\n"
                f"Error tail from run.log:\n```\n{res.error_tail[-2000:]}\n```\n"
                "Fix the bug in train.py while preserving your experimental idea (or simplify it)."
            )
            try:
                fix_prompt = build_research_prompt(program_md, new_code, rows, hardware_desc, fix_feedback)
                fix_resp = llm_complete(SYSTEM_PROMPT, fix_prompt)
                fix_prop = parse_llm_proposal(fix_resp)
                fixed_code = apply_edits_safely(new_code, fix_prop.blocks)
                train_path.write_text(fixed_code, encoding="utf-8")
                _git(repo_dir, "add", "train.py")
                _git(repo_dir, "commit", "--amend", "-m", f"{proposal.description} (fixed)")
                cand_commit = current_commit(repo_dir)
                res = run_training_subprocess(repo_dir, python_exe, timeout_s, log_path)
            except Exception as fix_exc:
                print(f"  [Crash fix skipped] {fix_exc}")

        # Ratchet decision: keep vs discard vs crash
        if not res.ok:
            status = "crash"
            print(f"  [CRASH] Reverting {cand_commit} -> HEAD~1")
            _git(repo_dir, "reset", "--hard", "HEAD~1")
        elif res.val_bpb < best_bpb:
            delta = best_bpb - res.val_bpb
            best_bpb = res.val_bpb
            status = "keep"
            shutil.copy2(train_path, repo_dir / "best_train.py")
            print(
                f"  [KEEP] val_bpb={res.val_bpb:.6f} (improved by -{delta:.6f}!) | "
                f"mem={res.peak_vram_mb/1024:.2f}GB | mfu={res.mfu_percent:.1f}% | steps={res.num_steps}"
            )
            if git_push:
                _git(repo_dir, "push", "-u", "origin", branch, check=False)
        else:
            status = "discard"
            print(
                f"  [DISCARD] val_bpb={res.val_bpb:.6f} vs best={best_bpb:.6f} | "
                f"mem={res.peak_vram_mb/1024:.2f}GB | Reverting {cand_commit}"
            )
            _git(repo_dir, "reset", "--hard", "HEAD~1")

        # Persist logs, plot, and sync to Google Drive / Kaggle output
        append_result_tsv(tsv_path, cand_commit, res.val_bpb, res.peak_vram_mb, status, proposal.description)
        append_jsonl(
            jsonl_path,
            {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "iter": iter_num,
                "commit": cand_commit,
                "status": status,
                "description": proposal.description,
                "hypothesis": proposal.hypothesis,
                "edits": [{"search": s, "replace": r} for s, r in proposal.blocks],
                **asdict(res),
            },
        )
        rows = load_results_tsv(tsv_path)
        plot_progress(tsv_path, png_path)
        sync_to_dir(repo_dir, sync_dir)

    print(f"\n[Agent] Finished session. Best val_bpb = {best_bpb:.6f} across {len(rows)} total experiments.")
    return rows


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous research agent loop for autoresearch-xla (Colab & Kaggle GPU/TPU)"
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "anthropic", "openai", "openrouter", "openai-compat"],
        default="auto",
        help="LLM API provider (default: auto-detect from API key secrets/env)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override default LLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key (or comma-separated keys for rotation)")
    parser.add_argument("--base-url", type=str, default=None, help="Custom base URL for openai-compat provider")
    parser.add_argument(
        "--tag",
        type=str,
        default=_dt.datetime.now().strftime("%b%d").lower(),
        help="Experiment branch tag (creates/resumes autoresearch/<tag>)",
    )
    parser.add_argument("--max-experiments", type=int, default=100, help="Maximum experiments this session")
    parser.add_argument("--max-hours", type=float, default=None, help="Wall-clock safety limit in hours (e.g. 8.5 for Kaggle)")
    parser.add_argument("--timeout", type=int, default=None, help="Per-experiment subprocess timeout in seconds")
    parser.add_argument("--sync-dir", type=str, default=None, help="Persistence directory (e.g. Google Drive / Kaggle working)")
    parser.add_argument("--no-resume", action="store_true", help="Do not restore previous state from --sync-dir")
    parser.add_argument("--no-fix-crashes", action="store_true", help="Disable 1-shot automated crash repair")
    parser.add_argument("--git-push", action="store_true", help="Push branch to git remote 'origin' on every kept win")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature")
    parser.add_argument("--repo-dir", type=str, default=".", help="Path to autoresearch-xla repo")
    args = parser.parse_args()

    provider, keys, model, base_url = resolve_provider_and_keys(
        args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url
    )
    print(f"[Agent] Provider:  {provider} | Model: {model} | Keys: {len(keys)}")

    client = LLMClient(
        provider=provider,
        api_keys=keys,
        model=model,
        base_url=base_url,
        temperature=args.temperature,
    )
    sync_path = Path(args.sync_dir) if args.sync_dir else None

    run_autoresearch_loop(
        repo_dir=Path(args.repo_dir),
        tag=args.tag,
        llm_complete=client.complete,
        max_experiments=args.max_experiments,
        max_hours=args.max_hours,
        run_timeout=args.timeout,
        sync_dir=sync_path,
        resume=not args.no_resume,
        fix_crashes=not args.no_fix_crashes,
        git_push=args.git_push,
    )


if __name__ == "__main__":
    main()
