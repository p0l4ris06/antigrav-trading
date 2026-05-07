"""
data_harvester.py — Antigravity RL Data Pipeline
=================================================
Production-grade data ingestion for the Antigravity autoresearcher.

Improvements over v1:
  - Structured logging (file + console, rotating)
  - Checkpoint/resume: picks up mid-pull if interrupted
  - Exponential backoff with jitter on all network calls
  - yfinance MultiIndex column flattening (breaking change in yfinance ≥0.2.18)
  - Post-fetch data validation: gap detection, outlier z-score, min-row enforcement
  - JSON manifest tracking every asset fetched (rows, date range, schema hash)
  - Alignment utility: forward-fills ETF gaps across crypto weekends
  - CLI interface: override assets, timeframes, days_back at runtime
  - Data quality report printed + saved after every run
  - Graceful SIGINT handling — flushes manifest before exit

Usage:
    python data_harvester.py
    python data_harvester.py --days 365 --output raw_data
    python data_harvester.py --crypto BTC/USDT ETH/USDT --etfs SPY QQQ
    python data_harvester.py --align  # also writes aligned/ subfolder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import ccxt
import polars as pl
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class HarvesterConfig:
    output_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = ".checkpoints"
    crypto_timeframe: str = "15m"
    etf_timeframe: str = "1h"
    days_back: int = 730
    max_retries: int = 6
    base_backoff: float = 2.0       # seconds; doubles each retry
    backoff_jitter: float = 0.3     # ± 30% jitter
    min_rows_crypto: int = 50_000   # fail validation below this
    min_rows_etf: int = 2_000
    gap_threshold_minutes: int = 30  # flag gaps larger than this (crypto)
    gap_threshold_etf_hours: int = 4  # flag ETF gaps larger than this
    outlier_z_score: float = 8.0    # flag candles with z-score above this
    crypto_assets: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    etf_assets: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IBIT"])
    align_output: bool = False


# ─────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "harvester.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )

    file_handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger("antigravity.harvester")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ─────────────────────────────────────────────
#  Manifest — tracks every file produced
# ─────────────────────────────────────────────

class Manifest:
    """JSON sidecar recording provenance of every parquet file."""

    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "manifest.json"
        self.records: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def update(self, key: str, meta: dict):
        self.records[key] = {**meta, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def already_done(self, key: str, days_back: int) -> bool:
        """Return True if this asset was fetched recently (within 1 day)."""
        if key not in self.records:
            return False
        rec = self.records[key]
        updated = datetime.fromisoformat(rec.get("updated_at", "2000-01-01T00:00:00+00:00"))
        age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
        return age_hours < 24 and rec.get("days_back", 0) >= days_back


# ─────────────────────────────────────────────
#  Checkpoint — resume interrupted pulls
# ─────────────────────────────────────────────

class Checkpoint:
    """Stores the last successfully fetched 'since' timestamp per symbol."""

    def __init__(self, checkpoint_dir: str, key: str):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        safe_key = key.replace("/", "_").replace(":", "_")
        self.path = Path(checkpoint_dir) / f"{safe_key}.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def get_since(self) -> Optional[int]:
        return self._data.get("since_ms")

    def get_rows(self) -> list:
        return self._data.get("rows", [])

    def save(self, since_ms: int, rows: list):
        self._data = {"since_ms": since_ms, "rows": rows, "saved_at": time.time()}
        with open(self.path, "w") as f:
            json.dump(self._data, f)

    def clear(self):
        if self.path.exists():
            self.path.unlink()


# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────

@dataclass
class ValidationReport:
    symbol: str
    rows: int
    start: str
    end: str
    gaps_found: int
    outliers_found: int
    passed: bool
    notes: list[str] = field(default_factory=list)

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"  [{status}] {self.symbol}: {self.rows:,} rows | {self.start} -> {self.end}",
            f"         gaps={self.gaps_found} | outliers={self.outliers_found}",
        ]
        for note in self.notes:
            lines.append(f"         [WARN] {note}")
        return "\n".join(lines)


def validate_dataframe(
    df: pl.DataFrame,
    symbol: str,
    gap_threshold_minutes: int,
    outlier_z: float,
    min_rows: int,
) -> ValidationReport:
    notes = []
    passed = True

    # Row count
    if df.height < min_rows:
        notes.append(f"Only {df.height:,} rows — expected ≥ {min_rows:,}")
        passed = False

    # Date range
    start = str(df["timestamp"].min())
    end = str(df["timestamp"].max())

    # Gap detection
    timestamps = df["timestamp"].sort().cast(pl.Int64)
    diffs = timestamps.diff().drop_nulls()
    gap_ms = gap_threshold_minutes * 60 * 1_000
    gaps = (diffs > gap_ms).sum()
    if gaps > 0:
        notes.append(f"{gaps} gap(s) exceeding {gap_threshold_minutes}min detected")

    # Outlier detection on close price (z-score)
    close = df["close"].cast(pl.Float64)
    mean = close.mean()
    std = close.std()
    if std and std > 0:
        outliers = ((close - mean).abs() / std > outlier_z).sum()
        if outliers > 0:
            notes.append(f"{outliers} outlier candle(s) with |z| > {outlier_z}")
    else:
        outliers = 0

    # Null check
    nulls = df.null_count().to_series().sum()
    if nulls > 0:
        notes.append(f"{nulls} null values remain after cleaning")
        passed = False

    return ValidationReport(
        symbol=symbol,
        rows=df.height,
        start=start,
        end=end,
        gaps_found=int(gaps),
        outliers_found=int(outliers),
        passed=passed,
        notes=notes,
    )


# ─────────────────────────────────────────────
#  Schema fingerprint (for manifest)
# ─────────────────────────────────────────────

def schema_hash(df: pl.DataFrame) -> str:
    schema_str = str(df.schema)
    return hashlib.md5(schema_str.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────
#  Core harvester
# ─────────────────────────────────────────────

class AntigravityHarvester:

    def __init__(self, config: HarvesterConfig, logger: logging.Logger):
        self.cfg = config
        self.log = logger
        self.manifest = Manifest(config.output_dir)
        self.reports: list[ValidationReport] = []
        self._shutdown = False

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.log.info("AntigravityHarvester initialised | output=%s", config.output_dir)

    # ── Retry decorator logic ──────────────────

    def _with_retry(self, fn, *args, label="call", **kwargs):
        for attempt in range(self.cfg.max_retries):
            if self._shutdown:
                raise KeyboardInterrupt
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as exc:
                if attempt == self.cfg.max_retries - 1:
                    raise
                wait = self.cfg.base_backoff * (2 ** attempt)
                jitter = wait * self.cfg.backoff_jitter * (random.random() * 2 - 1)
                wait = max(0.5, wait + jitter)
                self.log.warning(
                    "[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                    label, attempt + 1, self.cfg.max_retries, exc, wait
                )
                time.sleep(wait)

    # ── Crypto ────────────────────────────────

    def fetch_crypto(self, symbol: str):
        key = f"{symbol.replace('/', '_')}_{self.cfg.crypto_timeframe}"
        self.log.info("-- CRYPTO  %s (%s) --", symbol, self.cfg.crypto_timeframe)

        checkpoint = Checkpoint(self.cfg.checkpoint_dir, key)
        resumed_rows = checkpoint.get_rows()

        # Determine start
        if checkpoint.get_since() and resumed_rows:
            since = checkpoint.get_since()
            all_ohlcv = [list(r) for r in resumed_rows]
            self.log.info("Resuming from checkpoint: %s rows cached", len(all_ohlcv))
        else:
            since = self.exchange.parse8601(
                (datetime.now(timezone.utc) - timedelta(days=self.cfg.days_back)).isoformat()
            )
            all_ohlcv = []

        now_ms = self.exchange.milliseconds()

        while since < now_ms:
            if self._shutdown:
                break

            batch = self._with_retry(
                self.exchange.fetch_ohlcv,
                symbol,
                self.cfg.crypto_timeframe,
                since,
                1000,
                label=symbol,
            )
            if not batch:
                break

            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1
            checkpoint.save(since, all_ohlcv)

            self.log.debug(
                "  %s -> fetched up to %s (%d total rows)",
                symbol,
                self.exchange.iso8601(batch[-1][0]),
                len(all_ohlcv),
            )
            time.sleep(0.35)

        if not all_ohlcv:
            self.log.error("No data returned for %s", symbol)
            return

        df = pl.DataFrame(
            all_ohlcv,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        df = (
            df.with_columns([
                pl.col("timestamp")
                  .cast(pl.Datetime(time_unit="ms"))
                  .dt.replace_time_zone("UTC"),
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
            ])
            .unique(subset=["timestamp"])
            .sort("timestamp")
            .drop_nulls()
        )

        report = validate_dataframe(
            df, symbol,
            gap_threshold_minutes=self.cfg.gap_threshold_minutes,
            outlier_z=self.cfg.outlier_z_score,
            min_rows=self.cfg.min_rows_crypto,
        )
        self.reports.append(report)
        self.log.info("%s", report)

        self._save_parquet(df, key)
        self.manifest.update(key, {
            "symbol": symbol,
            "timeframe": self.cfg.crypto_timeframe,
            "days_back": self.cfg.days_back,
            "rows": df.height,
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
            "schema_hash": schema_hash(df),
            "source": "binance/ccxt",
        })
        checkpoint.clear()

    # ── ETFs ──────────────────────────────────

    def fetch_etfs(self, tickers: list[str]):
        self.log.info("-- ETFs  %s (%s) --", tickers, self.cfg.etf_timeframe)

        for ticker in tickers:
            if self._shutdown:
                break
            key = f"{ticker}_{self.cfg.etf_timeframe}"
            self.log.info("Pulling ETF: %s", ticker)

            try:
                raw = self._with_retry(
                    yf.download,
                    ticker,
                    period=f"{self.cfg.days_back}d",
                    interval=self.cfg.etf_timeframe,
                    progress=False,
                    auto_adjust=True,   # removes Adj Close, adjusts OHLC
                    label=ticker,
                )
            except Exception as exc:
                self.log.error("Failed to download %s: %s", ticker, exc)
                continue

            if raw is None or (hasattr(raw, "empty") and raw.empty):
                self.log.warning("No data returned for %s", ticker)
                continue

            raw.reset_index(inplace=True)

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower().strip() for col in raw.columns]
            else:
                raw.columns = [c.lower().strip().replace(" ", "_") for c in raw.columns]

            # Robustly find the timestamp column regardless of what yfinance calls it
            ts_col = next(
                (c for c in raw.columns if c in ("datetime", "date", "index", "timestamp")),
                None
            )
            if ts_col is None:
                self.log.error("Cannot find timestamp column for %s. Columns: %s", ticker, list(raw.columns))
                continue
            raw.rename(columns={ts_col: "timestamp"}, inplace=True)

            # Drop any residual adj_close / adj close columns
            drop_cols = [c for c in raw.columns if "adj" in c.lower()]
            raw.drop(columns=drop_cols, inplace=True, errors="ignore")

            # Ensure expected columns exist
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in raw.columns:
                    self.log.error("Missing column '%s' for %s — skipping", col, ticker)
                    break
            else:
                df = pl.from_pandas(raw[["timestamp", "open", "high", "low", "close", "volume"]])

                # Timezone: yfinance returns tz-aware for intraday
                if df["timestamp"].dtype == pl.Utf8:
                    df = df.with_columns(pl.col("timestamp").str.to_datetime())

                df = (
                    df.with_columns([
                        pl.col("timestamp").dt.convert_time_zone("UTC")
                        if df["timestamp"].dtype.is_(pl.Datetime("us", "America/New_York"))
                        else pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
                        pl.col("open").cast(pl.Float32),
                        pl.col("high").cast(pl.Float32),
                        pl.col("low").cast(pl.Float32),
                        pl.col("close").cast(pl.Float32),
                        pl.col("volume").cast(pl.Float32),
                    ])
                    .unique(subset=["timestamp"])
                    .sort("timestamp")
                    .drop_nulls()
                )

                report = validate_dataframe(
                    df, ticker,
                    gap_threshold_minutes=self.cfg.gap_threshold_etf_hours * 60,
                    outlier_z=self.cfg.outlier_z_score,
                    min_rows=self.cfg.min_rows_etf,
                )
                self.reports.append(report)
                self.log.info("%s", report)

                self._save_parquet(df, key)
                self.manifest.update(key, {
                    "symbol": ticker,
                    "timeframe": self.cfg.etf_timeframe,
                    "days_back": self.cfg.days_back,
                    "rows": df.height,
                    "start": str(df["timestamp"].min()),
                    "end": str(df["timestamp"].max()),
                    "schema_hash": schema_hash(df),
                    "source": "yahoo_finance/yfinance",
                })

    # ── Alignment ─────────────────────────────

    def build_aligned_dataset(self):
        """
        Forward-fill ETF data across weekends to match crypto 24/7 timeline.
        Writes aligned parquet files to data/aligned/.

        The Antigravity autoresearcher system prompt should reference these
        files when training on mixed crypto+ETF features. Using forward-filled
        ETF data during market-closed periods prevents the RL agent hallucinating
        on null feature rows.
        """
        aligned_dir = Path(self.cfg.output_dir) / "aligned"
        aligned_dir.mkdir(exist_ok=True)
        self.log.info("-- Building aligned dataset -> %s --", aligned_dir)

        # Load a crypto anchor for timestamps (BTC is always available)
        anchor_path = Path(self.cfg.output_dir) / f"BTC_USDT_{self.cfg.crypto_timeframe}.parquet"
        if not anchor_path.exists():
            self.log.error("Cannot align: missing BTC anchor at %s", anchor_path)
            return

        anchor = pl.read_parquet(anchor_path).select("timestamp")

        for ticker in self.cfg.etf_assets:
            src = Path(self.cfg.output_dir) / f"{ticker}_{self.cfg.etf_timeframe}.parquet"
            if not src.exists():
                self.log.warning("Skipping alignment for %s — parquet not found", ticker)
                continue

            etf = pl.read_parquet(src)

            # Join on nearest timestamp (asof join), then forward-fill
            merged = (
                anchor
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .sort("timestamp")
                .join_asof(
                    etf.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).sort("timestamp"),
                    on="timestamp",
                    strategy="backward",
                )
                .with_columns([
                    pl.col(c).forward_fill()
                    for c in ["open", "high", "low", "close", "volume"]
                ])
            )

            out_path = aligned_dir / f"{ticker}_aligned.parquet"
            merged.write_parquet(out_path)
            self.log.info(
                "Aligned %s: %d rows written to %s",
                ticker, merged.height, out_path
            )

    # ── I/O ──────────────────────────────────

    def _save_parquet(self, df: pl.DataFrame, key: str):
        path = Path(self.cfg.output_dir) / f"{key}.parquet"
        df.write_parquet(path, compression="zstd", compression_level=3)
        size_kb = path.stat().st_size // 1024
        self.log.info("Wrote %d rows -> %s (%d KB)", df.height, path, size_kb)

    # ── Quality report ────────────────────────

    def print_quality_report(self):
        passed = sum(1 for r in self.reports if r.passed)
        total = len(self.reports)
        print("\n" + "=" * 60)
        print(f"  ANTIGRAVITY DATA QUALITY REPORT  ({passed}/{total} passed)")
        print("=" * 60)
        for r in self.reports:
            print(r)
        print("=" * 60)

        # Save JSON version alongside manifest
        report_path = Path(self.cfg.output_dir) / "quality_report.json"
        with open(report_path, "w") as f:
            json.dump(
                [
                    {
                        "symbol": r.symbol,
                        "rows": r.rows,
                        "start": r.start,
                        "end": r.end,
                        "gaps_found": r.gaps_found,
                        "outliers_found": r.outliers_found,
                        "passed": r.passed,
                        "notes": r.notes,
                    }
                    for r in self.reports
                ],
                f,
                indent=2,
            )
        self.log.info("Quality report saved -> %s", report_path)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> HarvesterConfig:
    cfg = HarvesterConfig()
    p = argparse.ArgumentParser(
        description="Antigravity RL Data Harvester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", default=cfg.output_dir, help="Output directory for parquet files")
    p.add_argument("--days", type=int, default=cfg.days_back, help="Days of history to pull")
    p.add_argument("--crypto", nargs="+", default=cfg.crypto_assets, help="Crypto pairs (e.g. BTC/USDT)")
    p.add_argument("--etfs", nargs="+", default=cfg.etf_assets, help="ETF tickers (e.g. SPY QQQ)")
    p.add_argument("--crypto-tf", default=cfg.crypto_timeframe, help="Crypto candle timeframe")
    p.add_argument("--etf-tf", default=cfg.etf_timeframe, help="ETF candle timeframe")
    p.add_argument("--align", action="store_true", help="Also write aligned/ forward-filled dataset")
    p.add_argument("--retries", type=int, default=cfg.max_retries, help="Max retries per request")
    args = p.parse_args()

    cfg.output_dir = args.output
    cfg.days_back = args.days
    cfg.crypto_assets = args.crypto
    cfg.etf_assets = args.etfs
    cfg.crypto_timeframe = args.crypto_tf
    cfg.etf_timeframe = args.etf_tf
    cfg.align_output = args.align
    cfg.max_retries = args.retries
    return cfg


# ─────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cfg = parse_args()
    logger = setup_logging(cfg.log_dir)
    harvester = AntigravityHarvester(cfg, logger)

    # Graceful SIGINT: flush manifest then exit cleanly
    def _handle_sigint(sig, frame):
        logger.warning("SIGINT received — flushing manifest and exiting cleanly")
        harvester._shutdown = True
        harvester.manifest._flush()
        harvester.print_quality_report()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info("=== Antigravity Harvest Start ===")
    logger.info("Assets: crypto=%s  etfs=%s", cfg.crypto_assets, cfg.etf_assets)
    logger.info("Window: %d days back  |  crypto_tf=%s  etf_tf=%s", cfg.days_back, cfg.crypto_timeframe, cfg.etf_timeframe)

    # 1. Crypto
    for asset in cfg.crypto_assets:
        if not harvester._shutdown:
            harvester.fetch_crypto(asset)

    # 2. ETFs
    if not harvester._shutdown:
        harvester.fetch_etfs(cfg.etf_assets)

    # 3. Optional alignment layer
    if cfg.align_output and not harvester._shutdown:
        harvester.build_aligned_dataset()

    # 4. Report
    harvester.print_quality_report()
    logger.info("=== Antigravity Harvest Complete ===")
    logger.info(
        "Autoresearcher system prompt constraint:\n"
        '  "Read parquet files from data/ using polars.read_parquet(). '
        "Train PPO on first 18 months. Evaluate strictly on final 6 months. "
        'Optimise for Sharpe Ratio and Log-Wealth Utility. No look-ahead bias."'
    )


if __name__ == "__main__":
    main()
