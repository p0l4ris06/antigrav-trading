# Production Multi-Stage Hardened Dockerfile for ANTIGRAV TRADING Gateway
FROM python:3.11.9-slim as base

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root unprivileged security user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Install UV package manager
RUN pip install uv

# Copy dependencies manifest & install
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv pip install -e . --system

# Switch to non-root user
USER appuser

# Expose FastAPI Gateway Port
EXPOSE 8000

ENV AG_HOST=0.0.0.0
ENV AG_PORT=8000
ENV AG_MIN_REVERSAL_AGE_MS=3000.0

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["python", "-m", "antigravity.gateway.server"]
