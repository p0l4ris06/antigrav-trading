import asyncio, websockets, json

async def sniff():
    # Try simulated_feed first — this is likely the broadcast stream for the dashboard
    uri = "ws://localhost:8000/ws/simulated_feed"
    async with websockets.connect(uri) as ws:
        print("Connected to simulated_feed. Capturing 10 messages...\n")
        for _ in range(10):
            msg = await ws.recv()
            print(json.dumps(json.loads(msg), indent=2))
            print("---")

asyncio.run(sniff())
