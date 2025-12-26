"""Pydantic models for full process pipeline (OCR + Extraction)."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# API Request Models
class InmuebleInput(BaseModel):
    """Input model for a single inmueble to extract."""

    inmueble: str = Field(..., description="Nombre del inmueble")
    folio: str = Field(..., description="Folio de matrícula")

    class Config:
        json_schema_extra = {
            "example": {
                "inmueble": "APARTAMENTO 102 -T3",
                "folio": "294-109668",
            }
        }


class EscrituraRequest(BaseModel):
    """Request model for processing a single escritura."""

    escritura_bucket: str = Field(..., description="GCS bucket name containing the escritura PDF")
    escritura_file_name: str = Field(..., description="File name of the escritura PDF in the bucket")
    inmuebles: List[InmuebleInput] = Field(..., description="Lista de inmuebles a extraer de esta escritura")

    class Config:
        json_schema_extra = {
            "example": {
                "escritura_bucket": "escrituras_publicas",
                "escritura_file_name": "escritura_109668_del_29_04_2025.pdf",
                "inmuebles": [
                    {"inmueble": "APARTAMENTO 102 -T3", "folio": "294-109668"},
                    {"inmueble": "APARTAMENTO 103 -T3", "folio": "294-109669"},
                ],
            }
        }


class FullProcessRequest(BaseModel):
    """Request model for full process endpoint."""

    escrituras: List[EscrituraRequest] = Field(..., description="Lista de escrituras a procesar")

    class Config:
        json_schema_extra = {
            "example": {
                "escrituras": [
                    {
                        "escritura_bucket": "escrituras_publicas",
                        "escritura_file_name": "escritura_109668.pdf",
                        "inmuebles": [{"inmueble": "APARTAMENTO 102 -T3", "folio": "294-109668"}],
                    }
                ]
            }
        }


# API Response Models
class FullProcessResponse(BaseModel):
    """Response model for full process endpoint (202 Accepted)."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="processing", description="Job status")
    output_bucket: str = Field(..., description="GCS bucket where results will be uploaded")
    output_file_name: str = Field(..., description="File name of the output JSON in the bucket")
    estimated_time_seconds: Optional[int] = Field(None, description="Estimated processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123",
                "status": "processing",
                "output_bucket": "output-bucket",
                "output_file_name": "results_abc123.json",
                "estimated_time_seconds": 120,
            }
        }


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status: processing, completed, or failed")
    output_bucket: Optional[str] = Field(None, description="GCS bucket with results")
    output_file_name: Optional[str] = Field(None, description="Output file name")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123",
                "status": "completed",
                "output_bucket": "output-bucket",
                "output_file_name": "results_abc123.json",
            }
        }


# Output Models (for final JSON file)
class InmuebleResult(BaseModel):
    """Result model for a single inmueble extraction."""

    inmueble: str = Field(..., description="Nombre del inmueble")
    folio: str = Field(..., description="Folio de matrícula")
    status: str = Field(..., description="Status: found, not_found, or error")
    paginas: Optional[str] = Field(None, description="Páginas donde se encontró el inmueble (comma-separated)")
    text: str = Field(..., description="Texto extraído o mensaje de error")
    localization_method: Optional[str] = Field(
        None,
        description="Método de localización: 'text-regex' (localizador textual con regex) o 'refinement-llm' (refinamiento LLM)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "inmueble": "APARTAMENTO 102 -T3",
                "folio": "294-109668",
                "status": "found",
                "paginas": "45,46,47",
                "text": "APARTAMENTO NÚMERO CIENTO DOS (102), TORRE TRES (T-3)...",
                "localization_method": "text-regex",
            }
        }


class EscrituraMetadata(BaseModel):
    """Metadata for escritura processing."""

    total_pages: int = Field(..., description="Total number of pages in the escritura")
    processing_time_seconds: float = Field(..., description="Total processing time")
    ocr_time_seconds: float = Field(..., description="Time spent on OCR")
    extraction_time_seconds: float = Field(..., description="Time spent on extraction")


class EscrituraResult(BaseModel):
    """Result model for a single escritura processing."""

    escritura_file_name: str = Field(..., description="Name of the processed escritura file")
    resultados: List[InmuebleResult] = Field(..., description="List of inmueble extraction results")
    metadata: EscrituraMetadata = Field(..., description="Processing metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "escritura_file_name": "escritura_109668.pdf",
                "resultados": [
                    {
                        "inmueble": "APARTAMENTO 102 -T3",
                        "folio": "294-109668",
                        "status": "found",
                        "paginas": "45,46,47",
                        "text": "APARTAMENTO NÚMERO CIENTO DOS...",
                    }
                ],
                "metadata": {
                    "total_pages": 150,
                    "processing_time_seconds": 45.2,
                    "ocr_time_seconds": 30.1,
                    "extraction_time_seconds": 15.1,
                },
            }
        }


class FullProcessOutput(BaseModel):
    """Final output structure for JSON file uploaded to GCS."""

    escrituras: List[EscrituraResult] = Field(..., description="List of escritura processing results")

    class Config:
        json_schema_extra = {
            "example": {
                "escrituras": [
                    {
                        "escritura_file_name": "escritura_109668.pdf",
                        "resultados": [...],
                        "metadata": {...},
                    }
                ]
            }
        }


# Internal Data Classes (not for API)
@dataclass
class PaginaOCR:
    """Internal representation of a single OCR'd page."""

    page: int  # 1-indexed page number
    texto: str  # OCR text content


@dataclass
class ChunkInfo:
    """Internal representation of a text chunk with overlap."""

    index: int  # Chunk index (0-based)
    start_page: int  # First page in chunk (1-indexed)
    end_page: int  # Last page in chunk (1-indexed)
    texto: str  # Concatenated text from all pages in chunk


@dataclass
class EscrituraOCRResult:
    """Internal result from full OCR processing of an escritura."""

    escritura_bucket: str
    escritura_file_name: str
    total_pages: int
    paginas: List[PaginaOCR]  # All pages with OCR text
    chunks: List[ChunkInfo]  # Chunks with overlap for localization


# Statistics Models
class ExtractionPhaseStats(BaseModel):
    """Statistics for extraction phases."""

    phase1_text_localization_time_seconds: float = Field(..., description="Time for text localization phase")
    phase2_llm_refinement_time_seconds: float = Field(..., description="Time for LLM refinement phase (0 if skipped)")
    phase3_extraction_time_seconds: float = Field(..., description="Time for extraction phase")
    phase3_preparation_time_seconds: float = Field(..., description="Time for building extraction requests")
    found_by_text_regex: int = Field(..., description="Number of inmuebles found by text-regex method")
    found_by_refinement_llm: int = Field(..., description="Number of inmuebles found by refinement-llm method")
    total_found: int = Field(..., description="Total number of inmuebles found")
    total_not_found: int = Field(..., description="Total number of inmuebles not found")


class EscrituraStats(BaseModel):
    """Statistics for a single escritura processing."""

    escritura_file_name: str = Field(..., description="Name of the processed escritura")
    total_pages: int = Field(..., description="Total pages in escritura")
    total_inmuebles: int = Field(..., description="Total inmuebles to extract")
    found_count: int = Field(..., description="Number of inmuebles found")
    not_found_count: int = Field(..., description="Number of inmuebles not found")
    ocr_time_seconds: float = Field(..., description="Time spent on OCR")
    extraction_time_seconds: float = Field(..., description="Time spent on extraction")
    total_processing_time_seconds: float = Field(..., description="Total processing time")
    extraction_stats: ExtractionPhaseStats = Field(..., description="Detailed extraction phase statistics")


class BatchStats(BaseModel):
    """Statistics for a complete batch processing."""

    job_id: str = Field(..., description="Job identifier")
    timestamp: str = Field(..., description="ISO timestamp of when batch completed")
    total_escrituras: int = Field(..., description="Total number of escrituras processed")
    total_inmuebles: int = Field(..., description="Total number of inmuebles across all escrituras")
    total_found: int = Field(..., description="Total inmuebles found across all escrituras")
    total_not_found: int = Field(..., description="Total inmuebles not found across all escrituras")
    total_processing_time_seconds: float = Field(..., description="Total batch processing time")
    total_ocr_time_seconds: float = Field(..., description="Total OCR time across all escrituras")
    total_extraction_time_seconds: float = Field(..., description="Total extraction time across all escrituras")
    localization_method_counts: Dict[str, int] = Field(
        ..., description="Count of inmuebles by localization method"
    )
    escritura_stats: List[EscrituraStats] = Field(..., description="Statistics per escritura")
    aggregated_extraction_stats: ExtractionPhaseStats = Field(
        ..., description="Aggregated extraction phase statistics across all escrituras"
    )

