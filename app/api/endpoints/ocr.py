"""OCR endpoint."""

import logging

from fastapi import APIRouter, HTTPException

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
async def process_ocr_pages(request: OCRRequest) -> OCRResponse:
    """
    Process OCR for multiple pages in parallel.

    This endpoint:
    - Receives a list of page requests (bucket, file, page)
    - Deduplicates pages (same bucket/file/page processed only once)
    - Processes unique pages in parallel
    - Returns results in the same order as input
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR service not initialized")

    if not request.pages:
        return OCRResponse(results=[])

    try:
        logger.info(f"Received OCR request for {len(request.pages)} pages")
        results = await ocr_service.process_pages_async(request.pages)
        logger.info(f"Completed OCR request, returning {len(results)} results")
        return OCRResponse(results=results)

    except Exception as e:
        logger.error(f"Error processing OCR request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e

