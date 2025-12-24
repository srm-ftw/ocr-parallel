"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.endpoints import ocr
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
from app.services.health_service import HealthService
from app.services.ocr_service import OCRService

# Setup logging
setup_logging()

logger = logging.getLogger(__name__)

# Global service instances
ocr_service: OCRService = None
health_service: HealthService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global ocr_service, health_service

    # Startup
    logger.info("Starting OCR endpoint service...")
    try:
        ocr_service = OCRService()
        health_service = HealthService()
        ocr.set_ocr_service(ocr_service)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down OCR endpoint service...")
    if ocr_service:
        await ocr_service.close()
    logger.info("Services closed")


# Create FastAPI app
app = FastAPI(
    title="OCR Endpoint",
    description="API para procesar páginas de PDFs con OCR usando Gemini API",
    version=__version__,
    lifespan=lifespan,
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

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

