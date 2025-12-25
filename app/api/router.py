"""Main API router."""

from fastapi import APIRouter

from app.api.endpoints import extractor, ocr

router = APIRouter()

# Include all endpoint routers
router.include_router(ocr.router)
router.include_router(extractor.router)

