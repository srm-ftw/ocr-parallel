"""Pydantic models for OCR endpoints."""

from typing import List

from pydantic import BaseModel, Field


class OCRPageRequest(BaseModel):
    """Request model for a single page OCR request."""

    bucket: str = Field(..., description="Nombre del bucket de GCS donde está almacenado el PDF")
    file: str = Field(..., description="Nombre del archivo PDF en el bucket")
    page: str = Field(..., description="Número de página a procesar (1-indexado)")

    class Config:
        json_schema_extra = {
            "example": {
                "bucket": "bucket-escrituras-smr",
                "file": "cropped_7_8_-1_602_cropped_2 1 P112295 IRIS 1DmSnictBLSUrwspqJitakdqXON28zCnb.pdf",
                "page": "1",
            }
        }


class OCRPageResponse(BaseModel):
    """Response model for a single page OCR result."""

    page: str = Field(..., description="Número de página procesada")
    texto_ocr: str = Field(..., description="Texto extraído por OCR para esa página")

    class Config:
        json_schema_extra = {
            "example": {
                "page": "1",
                "texto_ocr": "Texto extraído de la página...",
            }
        }


class OCRRequest(BaseModel):
    """Request model for batch OCR processing."""

    pages: List[OCRPageRequest] = Field(..., description="Lista de páginas a procesar")

    class Config:
        json_schema_extra = {
            "example": {
                "pages": [
                    {
                        "bucket": "bucket-escrituras-smr",
                        "file": "documento.pdf",
                        "page": "1",
                    }
                ]
            }
        }


class OCRResponse(BaseModel):
    """Response model for batch OCR processing."""

    results: List[OCRPageResponse] = Field(..., description="Lista de resultados de OCR")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "page": "1",
                        "texto_ocr": "Texto extraído...",
                    }
                ]
            }
        }

