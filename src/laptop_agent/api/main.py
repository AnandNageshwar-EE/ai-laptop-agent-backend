"""FastAPI application.

Deliberate choices:

* CORS uses an explicit origin allowlist, never ``*``. The Streamlit frontend is
  a separate service on its own origin, so this is the boundary between them.
* Unhandled exceptions return a generic message. Stack traces and exception text
  go to the redacting logger, never to the client — an error message is a
  classic information-disclosure channel.
* The agent is constructed at startup so the first request does not pay for it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..agent import get_agent, reset_agent
from ..config import get_settings
from ..security.logging import configure_logging, get_logger
from .routes import router

_logger = get_logger("laptop_agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    agent = get_agent()
    _logger.info(
        "api.startup",
        extra={
            "environment": settings.environment,
            "llm_mode": settings.llm_mode,
            "model": settings.traced_model_name,
            "tracing_enabled": settings.tracing_enabled,
            "prompt_caching_enabled": settings.prompt_caching_enabled,
            "marketplaces": [m.value for m in agent.registry.marketplaces],
        },
    )
    try:
        yield
    finally:
        agent.shutdown()
        reset_agent()
        _logger.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Laptop Shopping Agent",
        version="1.0.0",
        description=(
            "Laptop shopping agent with layered guardrails, deterministic pricing, "
            "prompt caching and LangSmith tracing."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Detail goes to the log; the client gets nothing actionable.
        _logger.error(
            "api.unhandled_exception",
            extra={"path": request.url.path, "error_type": type(exc).__name__},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error. Please try again."},
        )

    return app


app = create_app()
