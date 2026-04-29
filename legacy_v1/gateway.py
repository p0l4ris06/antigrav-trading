"""
ANTIGRAV GATEWAY: PHASE 1.1
===========================
winloop-backed FastAPI Gateway for Windows High-Performance I/O.
WebSocket Pooling and Bounded Queue Backpressure.
"""

import asyncio
import logging
import json
import winloop # High-performance event loop for Windows
from fastapi import FastAPI, WebSocket
from typing import List
from datetime import datetime

# Enforce winloop for Cython-based libuv event management
asyncio.set_event_loop_policy(winloop.EventLoopPolicy())

app = FastAPI(title="ANTIGRAV_GATEWAY")
# Bounded Queue (Phase 1.1): Strict Backpressure at 50,000 items
data_queue = asyncio.Queue(maxsize=50000)

@app.on_event("startup")
async def startup_event():
    logging.info("GATEWAY >> winloop Event Loop Active (Windows libuv).")
    asyncio.create_task(ingestion_consumer())

async def ingestion_consumer():
    """Consumes ticks from the bounded queue for inference/persistence."""
    while True:
        try:
            item = await data_queue.get()
            # Logic: Pass to Feature Factory & Decision Core
            data_queue.task_done()
        except Exception as e:
            logging.error(f"GATEWAY >> Consumer Err: {e}")

@app.websocket("/ws/v1/ticks")
async def tick_stream(websocket: WebSocket):
    """L2/L3 Order Book Delta Ingestion."""
    await websocket.accept()
    logging.info("GATEWAY >> WebSocket Ingestion Channel Open.")
    try:
        while True:
            raw_data = await websocket.receive_text()
            tick = json.loads(raw_data)
            
            # Strict Backpressure Check
            if not data_queue.full():
                await data_queue.put(tick)
            else:
                # Explicit Drop: Preventing OOM / Stale Execution
                logging.warning("GATEWAY >> Queue Saturated (50k). Dropping stale tick.")
                
    except Exception as e:
        logging.error(f"GATEWAY >> Connection Terminated: {e}")

if __name__ == "__main__":
    import uvicorn
    # Workers=1 to maintain deterministic order in single-asset streams
    uvicorn.run("gateway:app", host="0.0.0.0", port=8000, loop="asyncio") # uvicorn will use our policy
