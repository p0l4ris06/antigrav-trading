# ============================================================
# Antigravity Trading — Multi-Stage Docker Build
# Stage 1: Build dashboard (Node)
# Stage 2: Runtime (Python + pre-built dashboard)
# ============================================================

# --- Stage 1: Dashboard Build ---
FROM node:20-alpine AS dashboard-build
WORKDIR /build
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# --- Stage 2: Python Runtime ---
FROM python:3.12-slim AS runtime
LABEL maintainer="Antigravity Systems"

# System deps for numpy/scipy wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    (pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "websockets>=13" \
        "clickhouse-connect>=0.8" \
        "polars>=1.0" \
        "numpy>=1.26" \
        "scipy>=1.14" \
        "scikit-learn>=1.5" \
        "gymnasium>=1.0" \
        "stable-baselines3>=2.4" \
        "pydantic>=2.9" \
        "pydantic-settings>=2.5" \
        "structlog>=24.4" \
        "uvloop>=0.21" \
        "opentelemetry-api>=1.27" \
        "opentelemetry-sdk>=1.27" \
        "opentelemetry-exporter-otlp-proto-http>=1.27" \
        "opentelemetry-instrumentation-fastapi>=0.48b0")

# Copy application code
COPY antigravity/ ./antigravity/

# Copy pre-built dashboard from stage 1
COPY --from=dashboard-build /build/dist ./dashboard/dist/

EXPOSE 8000

# Health check against the gateway
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "antigravity.gateway.server"]
