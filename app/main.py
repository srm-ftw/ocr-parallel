"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app import __version__
from app.api.endpoints import extractor, full_process, ocr
from app.api.router import router
from app.config import settings
from app.core.exceptions import (
    GCSException,
    GeminiAPIException,
    OCRException,
    PDFProcessingException,
    RateLimitException,
)
from app.core.logging_config import setup_logging
from app.models.common import ErrorResponse, HealthResponse
from app.services.extractor_service import ExtractorService
from app.services.full_process_orchestrator import FullProcessOrchestrator
from app.services.health_service import HealthService
from app.services.ocr_service import OCRService

# Setup logging
setup_logging()

logger = logging.getLogger(__name__)

# Global service instances
ocr_service: OCRService = None
extractor_service: ExtractorService = None
health_service: HealthService = None
full_process_orchestrator: FullProcessOrchestrator = None


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""

    def __init__(self, app, max_size_bytes: int):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: StarletteRequest, call_next):
        """Check request size before processing."""
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    logger.warning(
                        f"Request body too large: {size:,} bytes (max: {self.max_size_bytes:,} bytes) "
                        f"from {request.client.host if request.client else 'unknown'}"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content=ErrorResponse(
                            error="Request Entity Too Large",
                            detail=f"Request body size ({size:,} bytes) exceeds maximum allowed size ({self.max_size_bytes:,} bytes)",
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        ).model_dump(),
                    )
            except ValueError:
                # Invalid content-length header, let it through (will fail later if needed)
                pass

        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global ocr_service, extractor_service, health_service, full_process_orchestrator

    # Startup
    logger.info("Starting OCR endpoint service...")
    try:
        ocr_service = OCRService()
        extractor_service = ExtractorService()
        health_service = HealthService()
        full_process_orchestrator = FullProcessOrchestrator()
        ocr.set_ocr_service(ocr_service)
        extractor.set_extractor_service(extractor_service)
        full_process.set_orchestrator(full_process_orchestrator)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down OCR endpoint service...")
    if ocr_service:
        await ocr_service.close()
    if extractor_service:
        await extractor_service.close()
    if full_process_orchestrator:
        await full_process_orchestrator.close()
    logger.info("Services closed")


# Create FastAPI app
app = FastAPI(
    title="OCR Endpoint",
    description="API para procesar páginas de PDFs con OCR usando Gemini API",
    version=__version__,
    lifespan=lifespan,
)

# Request size limit middleware (must be before CORS)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_size_bytes=settings.max_request_size_bytes,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(OCRException)
async def ocr_exception_handler(request: Request, exc: OCRException):
    """Handle OCR-related exceptions."""
    logger.error(f"OCR exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="OCR Processing Error",
            detail=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


@app.exception_handler(GCSException)
async def gcs_exception_handler(request: Request, exc: GCSException):
    """Handle GCS-related exceptions."""
    logger.error(f"GCS exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="GCS Error",
            detail=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


@app.exception_handler(GeminiAPIException)
async def gemini_exception_handler(request: Request, exc: GeminiAPIException):
    """Handle Gemini API exceptions."""
    logger.error(f"Gemini API exception: {exc}")
    status_code = (
        status.HTTP_429_TOO_MANY_REQUESTS
        if isinstance(exc, RateLimitException)
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error="Gemini API Error",
            detail=str(exc),
            status_code=status_code,
        ).model_dump(),
    )


@app.exception_handler(PDFProcessingException)
async def pdf_exception_handler(request: Request, exc: PDFProcessingException):
    """Handle PDF processing exceptions."""
    logger.error(f"PDF processing exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="PDF Processing Error",
            detail=str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
        ).model_dump(),
    )


# Include routers
app.include_router(router)


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if health_service is None:
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            checks={"initialization": "error: services not initialized"},
        )

    health_data = await health_service.check_health()
    return HealthResponse(**health_data)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "OCR Endpoint",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.uvicorn_reload,
        workers=settings.uvicorn_workers if not settings.uvicorn_reload else 1,
        timeout_keep_alive=settings.uvicorn_timeout_keep_alive_seconds,
        timeout_graceful_shutdown=settings.uvicorn_timeout_graceful_shutdown_seconds,
        limit_concurrency=settings.uvicorn_limit_concurrency_value,
        backlog=settings.uvicorn_backlog,
    )

