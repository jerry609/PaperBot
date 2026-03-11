from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from paperbot.application.collaboration.message_schema import new_trace_id

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_CHAT_MAX_HISTORY = 20
TRACE_ID_HEADER = "X-Trace-Id"
GENERIC_INTERNAL_ERROR_MESSAGE = "Internal server error"
GENERIC_STREAM_ERROR_MESSAGE = "Request failed. Retry later or contact support with the trace_id."


def is_debug_enabled() -> bool:
    return os.getenv("PAPERBOT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_chat_max_history() -> int:
    raw_value = os.getenv("PAPERBOT_CHAT_MAX_HISTORY", str(DEFAULT_CHAT_MAX_HISTORY)).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_CHAT_MAX_HISTORY


def resolve_max_request_bytes() -> int:
    raw_value = os.getenv("PAPERBOT_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES)).strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES


def get_request_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", "")
    if trace_id:
        return str(trace_id)
    trace_id = request.headers.get(TRACE_ID_HEADER) or new_trace_id()
    request.state.trace_id = trace_id
    return trace_id


def public_exception_message(exc: Exception | str | None = None) -> str:
    if is_debug_enabled() and exc is not None:
        return str(exc)
    return GENERIC_INTERNAL_ERROR_MESSAGE


def json_error_response(
    request: Request,
    *,
    status_code: int,
    detail: str,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    trace_id = get_request_trace_id(request)
    payload: Dict[str, Any] = {"detail": detail, "trace_id": trace_id}
    if extra:
        payload.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={TRACE_ID_HEADER: trace_id},
    )


def json_internal_error_response(
    request: Request,
    *,
    exc: Exception,
    detail: str = GENERIC_INTERNAL_ERROR_MESSAGE,
    extra: Optional[Dict[str, Any]] = None,
    log_message: str = "Unhandled API error",
) -> JSONResponse:
    trace_id = get_request_trace_id(request)
    logger.exception(
        "%s trace_id=%s path=%s", log_message, trace_id, request.url.path, exc_info=exc
    )
    return json_error_response(
        request,
        status_code=500,
        detail=detail if not is_debug_enabled() else str(exc),
        extra=extra,
    )


class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get(TRACE_ID_HEADER) or new_trace_id()
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers.setdefault(TRACE_ID_HEADER, trace_id)
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        max_request_bytes = resolve_max_request_bytes()
        if max_request_bytes <= 0 or request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_request_bytes:
                    return json_error_response(
                        request,
                        status_code=413,
                        detail=f"Request body too large (max {max_request_bytes} bytes)",
                    )
            except ValueError:
                pass

        body = await request.body()
        if len(body) > max_request_bytes:
            return json_error_response(
                request,
                status_code=413,
                detail=f"Request body too large (max {max_request_bytes} bytes)",
            )

        return await call_next(request)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            rendered_detail = str(detail.get("detail") or detail)
        else:
            rendered_detail = str(detail)
        if exc.status_code >= 500:
            rendered_detail = public_exception_message(rendered_detail)
        return json_error_response(
            request,
            status_code=exc.status_code,
            detail=rendered_detail,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning(
            "Request validation failed trace_id=%s path=%s errors=%s",
            get_request_trace_id(request),
            request.url.path,
            exc.errors(),
        )
        return json_error_response(
            request,
            status_code=422,
            detail="Request validation failed",
            extra={"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return json_internal_error_response(request, exc=exc)


def install_api_error_handling(app: FastAPI) -> None:
    app.add_middleware(TraceIDMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    register_exception_handlers(app)
