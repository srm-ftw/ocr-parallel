"""Common Pydantic models."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Mensaje de error")
    detail: Optional[str] = Field(None, description="Detalles adicionales del error")
    status_code: int = Field(..., description="Código de estado HTTP")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Invalid page number",
                "status_code": 400,
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la aplicación")
    checks: Optional[Dict[str, Any]] = Field(None, description="Resultados de checks adicionales")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "checks": {
                    "gcs": "ok",
                    "gemini": "ok",
                },
            }
        }

