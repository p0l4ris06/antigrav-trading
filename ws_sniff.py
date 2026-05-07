import asyncio, websockets, json

async def sniff():
    uri = "ws://localhost:8000/ws/live"
    async with websockets.connect(uri) as ws:
        print("Connected. Capturing 10 messages...\n")
        for _ in range(10):
            msg = await ws.recv()
            print(json.dumps(json.loads(msg), indent=2))
            print("---")

asyncio.run(sniff())
