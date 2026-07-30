import { useEffect, useRef, useCallback, useState } from 'react';

type Handler = (msg: Record<string, unknown>) => void;

export function useWebSocket(url: string) {
  const ws = useRef<WebSocket | null>(null);
  const handlers = useRef<Set<Handler>>(new Set());
  const [connected, setConnected] = useState(false);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN || ws.current?.readyState === WebSocket.CONNECTING) return;
    
    try {
      const socket = new WebSocket(url);
      ws.current = socket;

      socket.onopen = () => {
        setConnected(true);
        reconnectAttempts.current = 0; // Reset attempts on successful connection
      };

      socket.onclose = () => {
        setConnected(false);
        ws.current = null;
        
        // Exponential backoff: min 1s, max 30s
        const backoff = Math.min(1000 * Math.pow(1.5, reconnectAttempts.current), 30000);
        reconnectAttempts.current++;
        
        if (reconnectTimeout.current !== null) {
          window.clearTimeout(reconnectTimeout.current);
        }
        reconnectTimeout.current = window.setTimeout(connect, backoff);
      };

      socket.onerror = (err) => {
        console.error('[useWebSocket] socket error on', url, err);
        socket.close();
      };

      socket.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          // Allow React to batch updates
          handlers.current.forEach(h => h(msg));
        } catch (err) {
          console.error("WebSocket message parse error:", err);
        }
      };
    } catch (err) {
      console.error("WebSocket connection error:", err);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeout.current !== null) {
        window.clearTimeout(reconnectTimeout.current);
      }
      ws.current?.close();
    };
  }, [connect]);

  const subscribe = useCallback((handler: Handler) => {
    handlers.current.add(handler);
    return () => {
      handlers.current.delete(handler);
    };
  }, []);

  return { connected, subscribe };
}
