import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import auth, health, memory
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    ForbiddenError,
    MemoryOSError,
    ResourceNotFoundError,
)
from app.core.logging import logger
from app.embeddings.service import embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing MEMORYOS backend engine...")
    # Pre-warm embedding service
    try:
        embedding_service.embed_text("memoryos warm up")
        logger.info("Dense embedding model pre-warmed successfully.")
    except Exception as e:
        logger.error(f"Error pre-warming embedding model: {e}")

    yield

    logger.info("Shutting down MEMORYOS backend engine...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="MEMORYOS: Intelligent Conversational Memory Engine (Convico AI Assessment)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID & Audit Logging Middleware
@app.middleware("http")
async def request_id_and_audit_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.time()

    response = await call_next(request)

    duration_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)

    # Safe access logging (never log Authorization header or query tokens)
    logger.info(
        f"{request.method} {request.url.path} returned {response.status_code} in {duration_ms}ms",
        extra={"request_id": req_id},
    )
    return response


# Domain Exception Handlers
@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": "authentication_error", "message": exc.message},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(ForbiddenError)
async def forbidden_error_handler(request: Request, exc: ForbiddenError):
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": "forbidden", "message": exc.message},
    )


@app.exception_handler(ResourceNotFoundError)
async def not_found_error_handler(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "not_found", "message": exc.message},
    )


@app.exception_handler(MemoryOSError)
async def domain_error_handler(request: Request, exc: MemoryOSError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "domain_error",
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.method} {request.url.path}: {exc}")
    # Never leak internal tracebacks or secrets to clients
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected server error occurred.",
        },
    )


# Root required endpoints (direct compliance with Section 11 specifications)
app.include_router(health.router, tags=["Health"])
app.include_router(memory.router, tags=["Memory Engine"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Versioned API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    return {
        "engine": "MEMORYOS",
        "description": "Intelligent Conversational Memory Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "ui": "/app",
    }


# Mount Apple-styled Glassmorphism React frontend if built
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/app", StaticFiles(directory=frontend_dist, html=True), name="frontend")
