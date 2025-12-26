"""OCR endpoint."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import verify_api_key
from app.config import DEFAULT_GEMINI_MODEL, get_dynamic_model, get_model_limits, settings
from app.models.ocr import OCRRequest, OCRResponse
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OCR"])

# Global service instance (will be initialized in main)
ocr_service: OCRService = None


def set_ocr_service(service: OCRService):
    """Set the OCR service instance."""
    global ocr_service
    ocr_service = service


@router.post("/ocr/pages", response_model=OCRResponse)
async def process_ocr_pages(
    request: OCRRequest,
    model: Optional[str] = Query(
        None,
        description="Gemini model to use (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro'). Defaults to configured default.",
    ),
    _: str = Depends(verify_api_key),
) -> OCRResponse:
    """
    Process OCR for multiple pages in parallel.

    This endpoint:
    - Receives a list of page requests (bucket, file, page)
    - Deduplicates pages (same bucket/file/page processed only once)
    - Processes unique pages in parallel
    - Returns results in the same order as input
    
    Query Parameters:
    - model: Optional Gemini model name.
             - Use 'dynamic' for automatic selection: Pro for <threshold pages, Flash for >=threshold
             - Available models: 'gemini-2.0-flash', 'gemini-2.5-pro', 'gemini-2.5-flash', etc.
             - If not provided, uses default from configuration.
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR service not initialized")

    if not request.pages:
        return OCRResponse(results=[])

    # Determine model to use
    if model == "dynamic":
        # Calculate model dynamically based on batch size
        model_to_use = get_dynamic_model(
            len(request.pages),
            threshold=settings.dynamic_model_threshold,
            fast_model=settings.dynamic_model_fast,
            slow_model=settings.dynamic_model_slow,
        )
        logger.info(
            f"Dynamic model selection: {len(request.pages)} pages -> {model_to_use} "
            f"(threshold: {settings.dynamic_model_threshold}, "
            f"fast: {settings.dynamic_model_fast}, slow: {settings.dynamic_model_slow})"
        )
    elif model:
        # Validate provided model
        try:
            get_model_limits(model)
            model_to_use = model
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Use default
        model_to_use = DEFAULT_GEMINI_MODEL

    request_start_time = time.time()
    try:
        logger.info(
            f"Received OCR request for {len(request.pages)} pages "
            f"(model: {model_to_use})"
        )
        results = await ocr_service.process_pages_async(request.pages, model=model_to_use)
        request_elapsed = time.time() - request_start_time
        logger.info(
            f"Completed OCR request in {request_elapsed:.2f}s, "
            f"returning {len(results)} results"
        )
        return OCRResponse(results=results)

    except Exception as e:
        request_elapsed = time.time() - request_start_time
        logger.error(
            f"Error processing OCR request after {request_elapsed:.2f}s: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

