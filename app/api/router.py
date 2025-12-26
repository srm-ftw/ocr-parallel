"""Main API router."""

from fastapi import APIRouter

from app.api.endpoints import extractor, full_process, ocr

router = APIRouter()

# Include all endpoint routers
router.include_router(ocr.router)
router.include_router(extractor.router)
router.include_router(full_process.router)

