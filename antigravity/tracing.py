"""
OpenTelemetry Tracing — LangSmith Export.

Initializes the OTel TracerProvider with an OTLP/HTTP exporter pointed at
LangSmith's ingestion endpoint. Provides convenience decorators and context
managers for instrumenting subsystem operations.

Architecture:
    - TracerProvider with BatchSpanProcessor (async, non-blocking export)
    - OTLP/HTTP exporter → https://api.smith.langchain.com/otel
    - x-api-key header authentication
    - TraceIdRatioBased sampler for production volume control
    - Auto-instrumentation of FastAPI via opentelemetry-instrumentation-fastapi

Usage:
    from antigravity.tracing import init_tracing, get_tracer, trace_span

    # At startup
    init_tracing(app)  # instruments FastAPI + configures exporter

    # In subsystem code
    tracer = get_tracer("antigravity.features")
    with tracer.start_as_current_span("compute_features") as span:
        span.set_attribute("buffer_height", 5000)
        ...
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

import structlog

from antigravity.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = structlog.get_logger(__name__)

# Module-level tracer cache
_tracers: dict[str, Any] = {}
_initialized: bool = False


def init_tracing(app: "FastAPI | None" = None) -> bool:
    """
    Initialize OpenTelemetry with LangSmith OTLP export.

    Args:
        app: FastAPI application to auto-instrument (optional)

    Returns:
        True if tracing was successfully initialized, False otherwise.
    """
    global _initialized

    cfg = settings.telemetry

    if not cfg.enabled:
        logger.info("tracing.disabled", hint="Set AG_OTEL_ENABLED=true to enable")
        return False

    if not cfg.api_key:
        logger.warning("tracing.no_api_key", hint="Set AG_OTEL_API_KEY to your LangSmith key")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Resource identifies this service in LangSmith
        resource = Resource.create(
            {
                "service.name": cfg.service_name,
                "service.version": "0.1.0",
                "deployment.environment": "development",
                "langsmith.project": cfg.project_name,
            }
        )

        # Sampler controls trace volume
        sampler = TraceIdRatioBased(cfg.sample_rate)

        # TracerProvider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # OTLP/HTTP exporter → LangSmith
        exporter = OTLPSpanExporter(
            endpoint=f"{cfg.endpoint}/v1/traces",
            headers={
                "X-API-Key": cfg.api_key,
            },
        )

        # BatchSpanProcessor: async, non-blocking export
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI if provided
        if app is not None:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor.instrument_app(
                    app,
                    excluded_urls="health",  # don't trace liveness probes
                )
                logger.info("tracing.fastapi_instrumented")
            except Exception as exc:
                logger.warning("tracing.fastapi_instrumentation_failed", error=str(exc))

        _initialized = True
        logger.info(
            "tracing.initialized",
            endpoint=cfg.endpoint,
            project=cfg.project_name,
            service=cfg.service_name,
            sample_rate=cfg.sample_rate,
        )
        return True

    except ImportError as exc:
        logger.warning(
            "tracing.import_error",
            error=str(exc),
            hint="Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http",
        )
        return False
    except Exception as exc:
        logger.error("tracing.init_failed", error=str(exc))
        return False


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _initialized
    if not _initialized:
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        _initialized = False
        logger.info("tracing.shutdown")
    except Exception as exc:
        logger.warning("tracing.shutdown_error", error=str(exc))


def get_tracer(name: str = "antigravity") -> Any:
    """
    Get a named tracer instance.

    Returns a real OTel tracer if initialized, or a no-op tracer otherwise.
    Tracers are cached by name.
    """
    if name in _tracers:
        return _tracers[name]

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(name)
    except Exception:
        tracer = _NoOpTracer()

    _tracers[name] = tracer
    return tracer


def trace_span(
    name: str,
    tracer_name: str = "antigravity",
    attributes: dict[str, Any] | None = None,
):
    """
    Decorator to wrap a function in an OTel span.

    Usage:
        @trace_span("compute_features", tracer_name="antigravity.features")
        def compute_features(self):
            ...

        @trace_span("predict", attributes={"component": "rl_agent"})
        def predict(self, obs):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(tracer_name)
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(exc))
                    span.record_exception(exc)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(tracer_name)
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(exc))
                    span.record_exception(exc)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def span(name: str, tracer_name: str = "antigravity", **attributes: Any):
    """
    Context manager for manual span creation.

    Usage:
        with span("clickhouse_query", symbol="BTCUSDT", rows=1000):
            result = await ch.query_ticks(...)
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            s.set_attribute(k, _safe_attr(v))
        yield s


def _safe_attr(value: Any) -> str | int | float | bool:
    """Coerce an attribute value to an OTel-safe type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# No-Op Fallback (when OTel is not installed or disabled)
# ---------------------------------------------------------------------------
class _NoOpSpan:
    """Span stub that silently accepts all operations."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Tracer stub that returns no-op spans."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()
