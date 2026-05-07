import { useEffect, useRef, useCallback, useState } from 'react';

type Handler = (msg: Record<string, unknown>) => void;

export function useWebSocket(url: string) {
  const ws = useRef<WebSocket | null>(null);
  const handlers = useRef<Set<Handler>>(new Set());
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    const socket = new WebSocket(url);
    ws.current = socket;
    socket.onopen = () => setConnected(true);
    socket.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        handlers.current.forEach(h => h(msg));
      } catch {}
    };
  }, [url]);

  useEffect(() => { connect(); return () => ws.current?.close(); }, [connect]);

  const subscribe = useCallback((handler: Handler) => {
    handlers.current.add(handler);
    return () => handlers.current.delete(handler);
  }, []);

  return { connected, subscribe };
}
