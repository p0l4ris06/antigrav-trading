import urllib.request
req = urllib.request.Request(
    "http://localhost:8000/ws/live",
    headers={"Upgrade": "websocket", "Connection": "Upgrade",
             "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
             "Sec-WebSocket-Version": "13"}
)
try:
    urllib.request.urlopen(req)
except Exception as e:
    print(e)
