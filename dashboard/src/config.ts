/**
 * API Configuration
 * 
 * This module provides centralized configuration for all API endpoints
 * and environment-specific settings.
 */

/**
 * Get the API base URL based on environment and current location
 * - In development (Vite ports 5173, 8080, 3000): uses http://localhost:8000
 * - In production: uses the configured environment variable or current origin
 * - Falls back to empty string for same-origin requests
 */
function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  
  if (envUrl) {
    return envUrl;
  }
  
  if (typeof window !== 'undefined') {
    const devPorts = ['5173', '8080', '3000'];
    const isLocalhost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
    if (isLocalhost || devPorts.includes(window.location.port)) {
      return 'http://localhost:8000';
    }
    return `${window.location.protocol}//${window.location.host}`;
  }
  
  return '';
}

/**
 * Get the WebSocket base URL
 */
function getWsBaseUrl(): string {
  const envUrl = import.meta.env.VITE_WS_BASE_URL;
  
  if (envUrl) {
    return envUrl.replace(/\/$/, '');
  }
  
  const apiUrl = getApiBaseUrl();
  if (apiUrl) {
    return apiUrl.replace(/^http/, 'ws').replace(/\/$/, '');
  }
  
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}`;
  }
  
  return '';
}

/**
 * API Configuration object
 */
export const apiConfig = {
  // Base URLs
  baseUrl: getApiBaseUrl(),
  wsBaseUrl: getWsBaseUrl(),
  
  // API Endpoints
  endpoints: {
    // Status and monitoring
    status: '/api/status',
    prices: '/api/prices',
    
    // Control endpoints
    control: {
      pause: '/api/control/pause',
      resume: '/api/control/resume',
    },
    
    // WebSocket endpoints
    ws: {
      live: '/ws/live',
    },
  },
  
  // Request configuration
  request: {
    timeout: 10000, // 10 seconds
    retries: 3,
    retryDelay: 1000, // 1 second
  },
  
  // Poll intervals (in milliseconds)
  polling: {
    status: 1000, // Update status every 1 second
    prices: 1000, // Update prices every 1 second
  },
} as const;

/**
 * Build a full API URL from an endpoint path
 */
export function buildApiUrl(endpoint: string): string {
  const base = apiConfig.baseUrl || '';
  return `${base}${endpoint}`;
}

/**
 * Build a full WebSocket URL from an endpoint path
 */
export function buildWsUrl(endpoint: string): string {
  const base = apiConfig.wsBaseUrl || '';
  return `${base}${endpoint}`;
}

/**
 * Check if API is configured and accessible
 */
export function isApiConfigured(): boolean {
  return apiConfig.baseUrl !== '';
}

export default apiConfig;
