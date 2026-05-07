"""
auto_optimizer.py — Antigravity Autoresearcher
===============================================
Iterative LLM-driven mutation loop for the Antigravity RL trading agent.

Providers supported:
  ollama   — local, free, OpenAI-compatible (default)
  azure    — enterprise Azure OpenAI
  anthropic — Claude via Anthropic SDK (best reasoning for complex mutations)
  gemini   — Google Gemini via OpenAI-compatible endpoint

Improvements over v1:
  - Fixed all syntax and indentation errors from original
  - Dataclass config + full environment variable override
  - Four provider support including Anthropic/Claude and Gemini
  - Rotating backup store (keeps last N checkpoints, not just last 1)
  - Experiment history: every iteration written to experiments/history.jsonl
  - Early stopping on score stagnation (configurable patience)
  - Resume: reads best score from history.jsonl on restart
  - Robust code block extraction: handles fenced, labelled, and unlabelled blocks
  - Adaptive temperature: warms up after stagnation to escape local optima
  - Structured rotating log file + console output
  - Graceful SIGINT: saves history and best score before exit
  - Multi-asset data support: passes all available parquet files as context
  - Quality gate: reads data/quality_report.json, aborts on failed assets
  - System prompt includes ANTIGRAVITY_HARVESTER_HANDOVER.md if present

Usage:
    python auto_optimizer.py
    python auto_optimizer.py --provider anthropic --model claude-opus-4-6 --iterations 100
    python auto_optimizer.py --provider ollama --model gemma3:4b --resume
    python auto_optimizer.py --provider azure --iterations 30 --patience 10

Environment variables (override config):
    LLM_PROVIDER, MODEL_NAME, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    ANTHROPIC_API_KEY, GEMINI_API_KEY, OLLAMA_HOST
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    # Provider: 'ollama' | 'azure' | 'anthropic' | 'gemini'
    provider: str = "ollama"

    # Model names per provider:
    #   ollama     → deepseek-coder-v2, qwen2.5-coder, gemma3:4b
    #   azure      → your deployment name (e.g. "gpt-4o-deployment")
    #   anthropic  → claude-opus-4-6, claude-sonnet-4-6
    #   gemini     → gemini-2.0-flash, gemini-1.5-pro
    model: str = "deepseek-coder-v2"

    # Files the LLM is asked to mutate
    target_files: list[str] = field(default_factory=lambda: [
        "core/features.py",
        "core/agent.py",
    ])

    # Training command — must print FITNESS_SCORE: <float> to stdout
    train_command: list[str] = field(default_factory=lambda: [
        "python", "train.py",
        "--data", "data/BTC_USDT_15m.parquet", "data/ETH_USDT_15m.parquet", "data/SOL_USDT_15m.parquet",
        "--timesteps", "50000",
    ])

    # Loop control
    iterations: int = 50
    patience: int = 8           # stop if no improvement for this many iterations
    timeout_seconds: int = 600  # per training run

    # LLM sampling
    temperature: float = 0.7
    temperature_stagnation: float = 1.1   # used after patience/2 stagnant iters
    max_tokens: int = 6000

    # Backup rotation — keep last N snapshots
    backup_dir: str = ".autoresearch_backups"
    backup_keep: int = 5

    # Experiment history
    history_dir: str = "experiments"
    history_file: str = "history.jsonl"

    # Logging
    log_dir: str = "logs"

    # Handover doc (injected into system prompt if present)
    handover_doc: str = "ANTIGRAVITY_HARVESTER_HANDOVER.md"

    # Ollama host override
    ollama_host: str = "http://localhost:11434"


def config_from_env(cfg: OptimizerConfig) -> OptimizerConfig:
    """Override config fields from environment variables."""
    cfg.provider = os.getenv("LLM_PROVIDER", cfg.provider)
    cfg.model = os.getenv("MODEL_NAME", cfg.model)
    cfg.ollama_host = os.getenv("OLLAMA_HOST", cfg.ollama_host)
    return cfg


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    fh = RotatingFileHandler(
        Path(log_dir) / "optimizer.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log = logging.getLogger("antigravity.optimizer")
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ─────────────────────────────────────────────
#  Provider client factory
# ─────────────────────────────────────────────

def build_client(cfg: OptimizerConfig, log: logging.Logger):
    provider = cfg.provider.lower()

    if provider == "ollama":
        from openai import OpenAI
        log.info("Provider: Ollama  host=%s  model=%s", cfg.ollama_host, cfg.model)
        return OpenAI(base_url=f"{cfg.ollama_host}/v1", api_key="ollama"), "openai_compat"

    elif provider == "azure":
        from openai import AzureOpenAI
        key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not key or not endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for Azure provider."
            )
        log.info("Provider: Azure OpenAI  endpoint=%s  deployment=%s", endpoint, cfg.model)
        return AzureOpenAI(
            api_key=key,
            api_version="2024-12-01-preview",
            azure_endpoint=endpoint,
        ), "openai_compat"

    elif provider == "anthropic":
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set for Anthropic provider.")
        log.info("Provider: Anthropic  model=%s", cfg.model)
        return anthropic.Anthropic(api_key=key), "anthropic"

    elif provider == "gemini":
        from openai import OpenAI
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY must be set for Gemini provider.")
        log.info("Provider: Gemini  model=%s", cfg.model)
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=key,
        ), "openai_compat"

    else:
        raise ValueError(f"Unknown provider: '{cfg.provider}'. Choose: ollama, azure, anthropic, gemini")


# ─────────────────────────────────────────────
#  Experiment history
# ─────────────────────────────────────────────

class History:
    def __init__(self, history_dir: str, history_file: str):
        Path(history_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(history_dir) / history_file

    def append(self, record: dict):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def best_score(self) -> Optional[float]:
        if not self.path.exists():
            return None
        best = None
        with open(self.path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    s = rec.get("score")
                    if s is not None and (best is None or s > best):
                        best = s
                except json.JSONDecodeError:
                    continue
        return best

    def stagnant_count(self, best: float) -> int:
        """Count trailing iterations with no improvement."""
        if not self.path.exists():
            return 0
        records = []
        with open(self.path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        count = 0
        for rec in reversed(records):
            s = rec.get("score")
            if s is not None and s >= best:
                break
            count += 1
        return count


# ─────────────────────────────────────────────
#  Backup manager (rotating)
# ─────────────────────────────────────────────

class BackupManager:
    def __init__(self, backup_dir: str, keep: int, target_files: list[str]):
        self.root = Path(backup_dir)
        self.keep = keep
        self.targets = target_files
        self.root.mkdir(parents=True, exist_ok=True)

    def _slot(self, n: int) -> Path:
        p = self.root / f"slot_{n:03d}"
        p.mkdir(exist_ok=True)
        return p

    def save(self, iteration: int):
        slot = self._slot(iteration % self.keep)
        for f in self.targets:
            src = Path(f)
            if src.exists():
                shutil.copy(src, slot / src.name)
        # Write slot metadata
        with open(slot / "meta.json", "w") as mf:
            json.dump({"iteration": iteration, "saved_at": datetime.now(timezone.utc).isoformat()}, mf)

    def restore_latest(self):
        slots = sorted(self.root.glob("slot_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not slots:
            raise FileNotFoundError("No backup slots found — cannot restore.")
        slot = slots[0]
        for f in self.targets:
            src_name = Path(f).name
            backed_up = slot / src_name
            if backed_up.exists():
                shutil.copy(backed_up, f)


# ─────────────────────────────────────────────
#  Fitness evaluation
# ─────────────────────────────────────────────

def run_evaluation(cfg: OptimizerConfig, log: logging.Logger) -> float:
    log.info("Running walk-forward evaluation: %s", " ".join(cfg.train_command))
    try:
        result = subprocess.run(
            cfg.train_command,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
        )
        # Accept FITNESS_SCORE or SHARPE or LOG_WEALTH as score signals
        for pattern in [
            r"FITNESS_SCORE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"SHARPE:\s*([-+]?\d*\.?\d+)",
            r"LOG_WEALTH:\s*([-+]?\d*\.?\d+)",
        ]:
            match = re.search(pattern, result.stdout)
            if match:
                score = float(match.group(1))
                if not math.isfinite(score):
                    log.warning("Degenerate score (%s) — zero trades or reward explosion. Reverting.", match.group(1))
                    return -999.0
                log.info("Fitness score: %.6f", score)
                return score

        log.error("No fitness score found in output. stdout:\n%s", result.stdout[-500:])
        if result.returncode != 0:
            log.error("stderr:\n%s", result.stderr[-300:])
        return -999.0

    except subprocess.TimeoutExpired:
        log.error("Training timed out after %ds — likely infinite loop introduced.", cfg.timeout_seconds)
        return -999.0
    except Exception as exc:
        log.error("Execution failed: %s", exc)
        return -999.0


# ─────────────────────────────────────────────
#  Code block extraction (robust)
# ─────────────────────────────────────────────

def extract_code_blocks(text: str) -> list[str]:
    """
    Extract Python code blocks from LLM output.
    Handles:
      ```python ... ```
      ```py ... ```
      ``` ... ```  (unlabelled)
    Returns list of extracted code strings (whitespace stripped).
    """
    patterns = [
        r"```(?:python|py)\n(.*?)```",   # labelled
        r"```\n(.*?)```",                # unlabelled
    ]
    blocks = []
    for pat in patterns:
        found = re.findall(pat, text, re.DOTALL)
        for b in found:
            b = b.strip()
            if b and b not in blocks:
                blocks.append(b)
        if len(blocks) >= 2:
            break

    return blocks


# ─────────────────────────────────────────────
#  LLM mutation
# ─────────────────────────────────────────────

def build_system_prompt(cfg: OptimizerConfig, n_features: int = 0) -> str:
    base = f"""You are an elite quantitative researcher optimising a Reinforcement Learning trading system called Antigravity.

Your objective: maximise the Out-of-Sample Logarithmic Wealth Utility (FITNESS_SCORE) as evaluated by Walk-Forward Optimisation. Train window = first 18 months. Eval window = final 6 months (unseen).

CRITICAL CONSTRAINT: features.py MUST output EXACTLY {n_features} feature columns. Do not add or remove features. Only modify their mathematical definitions.

Rules:
1. Return EXACTLY two Python code blocks in order: first core/features.py, then core/agent.py.
2. Wrap each in ```python ... ``` fences.
3. Use Polars (SIMD-accelerated) for all feature engineering. No pandas in features.py.
4. Do not introduce look-ahead bias.
5. Keep mathematical edge simple and robust.
6. Do not change the FITNESS_SCORE printing format in train.py.
7. If you cannot improve, return the files unchanged rather than breaking them.
"""
    # Inject handover doc if available
    hdoc = Path(cfg.handover_doc)
    if hdoc.exists():
        base += f"\n\n--- DATA PIPELINE REFERENCE ---\n{hdoc.read_text()}\n--- END REFERENCE ---\n"
    return base


def get_mutation(
    client,
    client_type: str,
    cfg: OptimizerConfig,
    current_score: float,
    file_contents: dict[str, str],
    temperature: float,
    log: logging.Logger,
    n_features: int = 0,
) -> str:
    system_prompt = build_system_prompt(cfg, n_features)

    files_block = "\n\n".join(
        f"### {path}\n```python\n{code}\n```"
        for path, code in file_contents.items()
    )

    # Read the exact feature list from the last known-good backup
    backup_features = Path(cfg.backup_dir) / "slot_000" / "features.py"
    canonical_cols = re.findall(r'\.alias\("(\w+)"\)', file_contents.get("core/features.py", ""))
    if not canonical_cols:
        canonical_cols = re.findall(r'"(\w+)"', file_contents.get("core/features.py", ""))

    RESERVED = {"timestamp", "open", "high", "low", "close", "volume"}
    canonical_cols = [c for c in canonical_cols if c not in RESERVED][:n_features]
    col_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(canonical_cols))

    user_prompt = f"""Current out-of-sample FITNESS_SCORE: {current_score:.6f}

HARD CONSTRAINTS -- READ BEFORE WRITING ANY CODE:
1. features.py must output EXACTLY {n_features} feature columns -- no more, no less.
2. These are the ONLY valid feature column names. Do not rename, add, or remove any:
{col_list}
3. NEVER use these reserved base column names as feature names: timestamp, open, high, low, close, volume.
4. agent.py may ONLY reference feature columns listed above -- no other column names.
5. Violations will cause a runtime crash and score -999.

{files_block}

Improve the mathematical definitions of the existing features only.
Output core/features.py first, then core/agent.py.
"""

    log.info("Requesting mutation from %s (%s) temp=%.2f", cfg.provider, cfg.model, temperature)

    if client_type == "anthropic":
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    else:  # openai_compat (ollama, azure, gemini)
        # GPT-5 / o-series use max_completion_tokens; older models use max_tokens
        token_param = (
            "max_completion_tokens"
            if any(x in cfg.model.lower() for x in ("gpt-5", "o1", "o3", "o4"))
            else "max_tokens"
        )
        response = client.chat.completions.create(
            model=cfg.model,
            **{token_param: cfg.max_tokens},
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def apply_mutation(
    llm_response: str,
    target_files: list[str],
    log: logging.Logger,
) -> bool:
    blocks = extract_code_blocks(llm_response)

    if len(blocks) < 2:
        log.error(
            "Expected 2 code blocks, got %d. LLM output snippet:\n%s",
            len(blocks),
            llm_response[:400],
        )
        return False

    for path, code in zip(target_files, blocks):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(code)
        log.debug("Wrote %d chars -> %s", len(code), path)

    return True


# ─────────────────────────────────────────────
#  Data quality gate
# ─────────────────────────────────────────────

def check_data_quality(log: logging.Logger) -> bool:
    report_path = Path("data/quality_report.json")
    if not report_path.exists():
        log.warning("No quality_report.json found — skipping data quality gate. Run data_harvester.py first.")
        return True
    with open(report_path) as f:
        reports = json.load(f)
    failures = [r["symbol"] for r in reports if not r.get("passed", True)]
    if failures:
        log.error("Data quality gate FAILED for: %s — re-run data_harvester.py", failures)
        return False
    log.info("Data quality gate PASSED (%d assets)", len(reports))
    return True


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args(cfg: OptimizerConfig) -> OptimizerConfig:
    p = argparse.ArgumentParser(
        description="Antigravity Autoresearcher — LLM mutation loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--provider", default=cfg.provider,
                   choices=["ollama", "azure", "anthropic", "gemini"])
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--iterations", type=int, default=cfg.iterations)
    p.add_argument("--patience", type=int, default=cfg.patience,
                   help="Stop after this many consecutive non-improving iterations")
    p.add_argument("--timeout", type=int, default=cfg.timeout_seconds,
                   help="Per-run training timeout in seconds")
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--max-tokens", type=int, default=cfg.max_tokens)
    p.add_argument("--resume", action="store_true",
                   help="Load best score from history.jsonl and continue")
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Skip data quality check (not recommended)")
    args = p.parse_args()

    cfg.provider = args.provider
    cfg.model = args.model
    cfg.iterations = args.iterations
    cfg.patience = args.patience
    cfg.timeout_seconds = args.timeout
    cfg.temperature = args.temperature
    cfg.max_tokens = args.max_tokens
    return cfg, args.resume, args.no_quality_gate


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cfg = OptimizerConfig()
    cfg = config_from_env(cfg)
    cfg, resume, skip_quality = parse_args(cfg)

    log = setup_logging(cfg.log_dir)
    history = History(cfg.history_dir, cfg.history_file)
    backups = BackupManager(cfg.backup_dir, cfg.backup_keep, cfg.target_files)

    # Graceful SIGINT
    shutdown = {"flag": False}
    def _sigint(sig, frame):
        log.warning("SIGINT received — finishing current iteration then exiting.")
        shutdown["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    log.info("=== ANTIGRAVITY AUTORESEARCHER START ===")
    log.info("Provider: %s | Model: %s | Iterations: %d | Patience: %d",
             cfg.provider, cfg.model, cfg.iterations, cfg.patience)

    # Data quality gate
    if not skip_quality and not check_data_quality(log):
        sys.exit(1)

    # Read feature count from existing model observation space
    n_features = 15  # update this if you retrain from scratch with a different count

    # Build LLM client
    client, client_type = build_client(cfg, log)

    # Baseline score
    if resume:
        best_score = history.best_score()
        if best_score is not None:
            log.info("Resuming — loaded best score from history: %.6f", best_score)
        else:
            log.info("No history found — running baseline evaluation.")
            best_score = run_evaluation(cfg, log)
    else:
        best_score = run_evaluation(cfg, log)
        log.info("BASELINE SCORE: %.6f", best_score)
        history.append({"iteration": 0, "score": best_score, "event": "baseline"})

    stagnant = 0
    temperature = cfg.temperature

    for i in range(1, cfg.iterations + 1):
        if shutdown["flag"]:
            log.info("Shutting down cleanly at iteration %d.", i)
            break

        log.info("-- ITERATION %d/%d  best=%.6f  stagnant=%d --",
                 i, cfg.iterations, best_score, stagnant)

        # Adaptive temperature: heat up after patience/2 stagnant iters
        if stagnant >= cfg.patience // 2:
            temperature = cfg.temperature_stagnation
            log.info("Stagnation detected — raising temperature to %.2f", temperature)
        else:
            temperature = cfg.temperature

        # Early stopping
        if stagnant >= cfg.patience:
            log.info("Early stopping: %d consecutive non-improving iterations.", cfg.patience)
            break

        # Read current files
        file_contents = {}
        missing = False
        for path in cfg.target_files:
            p = Path(path)
            if not p.exists():
                log.error("Target file missing: %s", path)
                missing = True
                break
            file_contents[path] = p.read_text()
        if missing:
            break

        # Backup before mutation
        backups.save(i)

        # Get LLM mutation
        try:
            llm_response = get_mutation(
                client, client_type, cfg, best_score, file_contents, temperature, log, n_features=n_features
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            history.append({"iteration": i, "score": None, "event": "llm_error", "error": str(exc)})
            stagnant += 1
            continue

        # Apply mutation
        if not apply_mutation(llm_response, cfg.target_files, log):
            backups.restore_latest()
            history.append({"iteration": i, "score": None, "event": "parse_error"})
            stagnant += 1
            continue

        # Evaluate
        new_score = run_evaluation(cfg, log)

        if new_score > best_score:
            delta = new_score - best_score
            log.info("IMPROVEMENT: %.6f -> %.6f (delta +%.6f)", best_score, new_score, delta)
            best_score = new_score
            stagnant = 0
            history.append({"iteration": i, "score": new_score, "event": "improvement", "delta": delta})
        else:
            log.info("No improvement (%.6f <= %.6f) - reverting.", new_score, best_score)
            backups.restore_latest()
            stagnant += 1
            history.append({"iteration": i, "score": new_score, "event": "revert"})

    log.info("=== RESEARCH CONCLUDED ===")
    log.info("Best score achieved: %.6f", best_score)
    log.info("Experiment history: %s/%s", cfg.history_dir, cfg.history_file)
    log.info(
        "Autoresearcher constraint reminder:\n"
        "  Train on first 18 months. Evaluate on final 6 months only.\n"
        "  Optimise for Sharpe Ratio AND Log-Wealth Utility. No look-ahead bias."
    )


if __name__ == "__main__":
    main()
