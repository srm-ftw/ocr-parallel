"""Main API router."""

from fastapi import APIRouter

from app.api.endpoints import ocr

router = APIRouter()

# Include all endpoint routers
router.include_router(ocr.router)

