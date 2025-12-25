"""Pydantic models for extractor endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class InmuebleRequest(BaseModel):
    """Request model for a single inmueble extraction request."""

    nombre_inmueble: str = Field(..., description="Nombre del inmueble")
    folio: str = Field(..., description="Folio de matrícula")
    texto_ocr_completo: str = Field(..., description="Texto OCR completo (puede incluir varias páginas)")
    paginas_original: Optional[str] = Field(None, description="Páginas originales (informativo)")
    paginas_con_offset: Optional[str] = Field(None, description="Páginas con offset (informativo)")

    class Config:
        json_schema_extra = {
            "example": {
                "nombre_inmueble": "LOTE UTIL 2",
                "folio": "307-102037",
                "texto_ocr_completo": "texto ocreado...",
                "paginas_original": "7-8",
                "paginas_con_offset": "1,2",
            }
        }


class InmuebleResponse(BaseModel):
    """Response model for a single inmueble extraction result."""

    text: str = Field(..., description="Texto literal extraído o 'Inmueble no encontrado'")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "TEXTO LITERAL EXTRAÍDO...",
            }
        }


class ExtractorRequest(BaseModel):
    """Request model for batch inmueble extraction."""

    inmuebles: List[InmuebleRequest] = Field(..., description="Lista de inmuebles a procesar")

    class Config:
        json_schema_extra = {
            "example": {
                "inmuebles": [
                    {
                        "nombre_inmueble": "LOTE UTIL 2",
                        "folio": "307-102037",
                        "texto_ocr_completo": "texto ocreado...",
                        "paginas_original": "7-8",
                        "paginas_con_offset": "1,2",
                    }
                ]
            }
        }


class ExtractorResponse(BaseModel):
    """Response model for batch inmueble extraction."""

    results: List[InmuebleResponse] = Field(..., description="Lista de resultados de extracción")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "TEXTO LITERAL EXTRAÍDO...",
                    }
                ]
            }
        }

