"""Extractor endpoint."""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from app.config import DEFAULT_GEMINI_MODEL, get_dynamic_model, get_model_limits, settings
from app.models.extractor import InmuebleRequest, InmuebleResponse
from app.services.extractor_service import ExtractorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Extractor"])

# Global service instance (will be initialized in main)
extractor_service: ExtractorService = None


def set_extractor_service(service: ExtractorService):
    """Set the extractor service instance."""
    global extractor_service
    extractor_service = service


@router.post("/extractor/inmuebles", response_model=List[InmuebleResponse])
async def extract_inmuebles(
    inmuebles: List[InmuebleRequest] = Body(..., description="Array de inmuebles a procesar"),
    model: Optional[str] = Query(
        None,
        description="Gemini model to use (e.g., 'gemini-2.0-flash', 'gemini-2.5-pro'). Defaults to configured default.",
    ),
) -> List[InmuebleResponse]:
    """
    Extract inmueble sections from OCR text using Gemini.

    This endpoint:
    - Receives a list of inmueble requests (nombre_inmueble, folio, texto_ocr_completo)
    - Processes each inmueble in parallel using Gemini
    - Returns results in the same order as input
    - Handles errors per item (doesn't fail entire batch)
    
    Query Parameters:
    - model: Optional Gemini model name. 
             - Use 'dynamic' for automatic selection: Pro for <threshold items, Flash for >=threshold
             - Available models: 'gemini-2.0-flash', 'gemini-2.5-pro', 'gemini-2.5-flash', etc.
             - If not provided, uses default from configuration.
    """
    if extractor_service is None:
        raise HTTPException(status_code=500, detail="Extractor service not initialized")

    if not inmuebles:
        return []

    # Determine model to use
    if model == "dynamic":
        # Calculate model dynamically based on batch size
        model_to_use = get_dynamic_model(
            len(inmuebles),
            threshold=settings.dynamic_model_threshold,
            fast_model=settings.dynamic_model_fast,
            slow_model=settings.dynamic_model_slow,
        )
        logger.info(
            f"Dynamic model selection: {len(inmuebles)} items -> {model_to_use} "
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
            f"Received extractor request for {len(inmuebles)} inmuebles "
            f"(model: {model_to_use})"
        )
        results = await extractor_service.process_inmuebles_async(inmuebles, model=model_to_use)
        request_elapsed = time.time() - request_start_time
        logger.info(
            f"Completed extractor request in {request_elapsed:.2f}s, "
            f"returning {len(results)} results"
        )
        return results

    except Exception as e:
        request_elapsed = time.time() - request_start_time
        logger.error(
            f"Error processing extractor request after {request_elapsed:.2f}s: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

