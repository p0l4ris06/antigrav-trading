# Multi-stage build

# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY . .
RUN pip install --user --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ src/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1
CMD ["antigrav", "serve", "--host", "0.0.0.0", "--port", "8000"]
