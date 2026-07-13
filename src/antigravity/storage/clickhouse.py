"""
Async ClickHouse client wrapper.

Provides batched tick insertion, zero-copy Arrow-to-Polars query results,
and connection lifecycle management. All I/O is non-blocking.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient
import polars as pl
import structlog

from antigravity.config import settings

logger = structlog.get_logger(__name__)

# Path to the DDL file for schema bootstrapping
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class ClickHouseManager:
    """
    High-performance async ClickHouse interface.

    Design:
        - Uses clickhouse-connect AsyncClient (HTTP interface)
        - Batches inserts to avoid per-row overhead
        - Returns Polars DataFrames via Arrow for zero-copy reads
        - Async insert mode for server-side buffering
    """

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    @classmethod
    async def create(cls) -> "ClickHouseManager":
        """Factory: create a connected manager instance."""
        cfg = settings.clickhouse
        client = await clickhouse_connect.get_async_client(
            host=cfg.host,
            port=cfg.port,
            username=cfg.user,
            password=cfg.password,
            database=cfg.database,
            connect_timeout=3,
            send_receive_timeout=3,
            settings={
                "async_insert": 1,
                "wait_for_async_insert": 0,
            },
        )
        logger.info(
            "clickhouse.connected",
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
        )
        return cls(client)

    async def close(self) -> None:
        """Close the underlying connection."""
        await self._client.close()
        logger.info("clickhouse.disconnected")

    # ------------------------------------------------------------------
    # Schema Management
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Execute the DDL statements to bootstrap tables and views."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        # Split on semicolons and execute each statement
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    await self._client.command(stmt)
                except Exception as exc:
                    logger.warning(
                        "clickhouse.schema_statement_skipped",
                        error=str(exc),
                        statement=stmt[:120],
                    )
        logger.info("clickhouse.schema_ensured")

    # ------------------------------------------------------------------
    # Tick Insertion
    # ------------------------------------------------------------------

    async def insert_tick_batch(self, ticks: list[dict[str, Any]]) -> None:
        """
        Insert a batch of tick dicts into the ticks table.

        Each dict must contain keys matching the ticks table columns:
            symbol, timestamp, bid_price, ask_price, bid_size, ask_size,
            last_price, last_size, trade_id
        """
        if not ticks:
            return

        columns = [
            "symbol",
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "last_price",
            "last_size",
            "trade_id",
        ]
        data = [[tick.get(col) for col in columns] for tick in ticks]

        await self._client.insert(
            table="ticks",
            data=data,
            column_names=columns,
        )
        logger.debug("clickhouse.ticks_inserted", count=len(ticks))

    # ------------------------------------------------------------------
    # Queries → Polars DataFrames
    # ------------------------------------------------------------------

    async def query_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 1_000_000,
    ) -> pl.DataFrame:
        """
        Fetch raw ticks for a symbol in [start, end] as a Polars DataFrame.
        Uses Arrow transport for zero-copy deserialization.
        """
        query = """
            SELECT *
            FROM ticks
            WHERE symbol = {symbol:String}
              AND timestamp >= {start:DateTime64(6, 'UTC')}
              AND timestamp <= {end:DateTime64(6, 'UTC')}
            ORDER BY timestamp
            LIMIT {limit:UInt64}
        """
        result = await self._client.query_arrow(
            query,
            parameters={"symbol": symbol, "start": start, "end": end, "limit": limit},
        )
        return pl.from_arrow(result)

    async def query_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch pre-aggregated 1-minute OHLCV candles from the materialized view.
        """
        query = """
            SELECT
                symbol,
                window_start,
                argMinMerge(open) AS open,
                maxMerge(high) AS high,
                minMerge(low) AS low,
                argMaxMerge(close) AS close,
                sumMerge(volume) AS volume
            FROM ohlcv_1m_store
            WHERE symbol = {symbol:String}
              AND window_start >= {start:DateTime('UTC')}
              AND window_start <= {end:DateTime('UTC')}
            GROUP BY symbol, window_start
            ORDER BY window_start
        """
        result = await self._client.query_arrow(
            query,
            parameters={"symbol": symbol, "start": start, "end": end},
        )
        return pl.from_arrow(result)

    async def query_recent_ticks(
        self,
        symbol: str,
        n_ticks: int = 10_000,
    ) -> pl.DataFrame:
        """Fetch the N most recent ticks for a symbol."""
        query = """
            SELECT *
            FROM ticks
            WHERE symbol = {symbol:String}
            ORDER BY timestamp DESC
            LIMIT {n:UInt64}
        """
        result = await self._client.query_arrow(
            query,
            parameters={"symbol": symbol, "n": n_ticks},
        )
        # Reverse to chronological order
        df = pl.from_arrow(result)
        return df.reverse() if df.height > 0 else df

    async def export_parquet(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        path: str | Path,
    ) -> Path:
        """
        Export tick data to a Parquet file for shadow fork training.
        Returns the path to the written file.
        """
        df = await self.query_ticks(symbol, start, end)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out, compression="zstd")
        logger.info(
            "clickhouse.parquet_exported",
            path=str(out),
            rows=df.height,
        )
        return out


@asynccontextmanager
async def get_clickhouse() -> AsyncIterator[ClickHouseManager]:
    """Context manager for ClickHouse lifecycle."""
    manager = await ClickHouseManager.create()
    try:
        yield manager
    finally:
        await manager.close()
