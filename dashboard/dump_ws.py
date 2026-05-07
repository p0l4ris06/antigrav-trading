import asyncio
import websockets
import json

async def test():
    try:
        async with websockets.connect("ws://localhost:8000/ws/live") as websocket:
            print("Connected to ws://localhost:8000/ws/live")
            for i in range(10):
                msg = await websocket.recv()
                print(f"MSG {i}:", json.dumps(json.loads(msg), indent=2))
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
