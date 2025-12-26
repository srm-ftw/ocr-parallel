"""Full process endpoint for complete OCR + extraction pipeline."""

import logging
import time
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query

from app.api.dependencies import verify_api_key
from app.config import settings
from app.models.full_process import (
    EscrituraRequest,
    FullProcessRequest,
    FullProcessResponse,
    JobStatusResponse,
)
from app.services.full_process_orchestrator import FullProcessOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Full Process"])

# Global orchestrator instance (will be initialized in main)
orchestrator: FullProcessOrchestrator = None


def set_orchestrator(orch: FullProcessOrchestrator):
    """Set the orchestrator instance."""
    global orchestrator
    orchestrator = orch


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return f"job_{uuid.uuid4().hex[:8]}"


def estimate_processing_time(requests: List[EscrituraRequest]) -> int:
    """
    Estimate processing time in seconds.

    Rough estimate: 2 seconds per page for OCR + 1 second per inmueble for extraction.

    Args:
        requests: List of escritura requests

    Returns:
        Estimated time in seconds
    """
    total_inmuebles = sum(len(req.inmuebles) for req in requests)
    # Rough estimate: assume average 100 pages per escritura
    total_pages = len(requests) * 100

    # Estimate: 2s per page OCR + 1s per inmueble extraction
    estimated = (total_pages * 2) + (total_inmuebles * 1)
    return max(estimated, 60)  # Minimum 1 minute


@router.post("/full-process", status_code=202, response_model=FullProcessResponse)
async def full_process(
    request: FullProcessRequest = Body(..., description="Full process request with escrituras"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: str = Depends(verify_api_key),
) -> FullProcessResponse:
    """
    Start full processing pipeline (OCR + extraction) for escrituras.

    This endpoint:
    - Receives a list of escritura requests (bucket, file, inmuebles)
    - Returns 202 Accepted immediately with job info
    - Processes escrituras in background
    - Uploads results to configured output bucket

    The client should poll the status endpoint or wait for the file to appear in the bucket.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator service not initialized")

    if not request.escrituras:
        raise HTTPException(status_code=400, detail="No escrituras provided")

    # Validate output bucket is configured
    if not settings.output_bucket:
        raise HTTPException(
            status_code=500, detail="Output bucket not configured. Set OUTPUT_BUCKET in environment."
        )

    # Generate job ID and output file name
    job_id = generate_job_id()
    output_file_name = f"results_{job_id}.json"

    # Estimate processing time
    estimated_time = estimate_processing_time(request.escrituras)

    logger.info(
        f"Received full process request: {len(request.escrituras)} escrituras, "
        f"job_id={job_id}, output=gs://{settings.output_bucket}/{output_file_name}"
    )

    # Add background task
    background_tasks.add_task(
        orchestrator.process_batch,
        request.escrituras,
        settings.output_bucket,
        output_file_name,
    )

    return FullProcessResponse(
        job_id=job_id,
        status="processing",
        output_bucket=settings.output_bucket,
        output_file_name=output_file_name,
        estimated_time_seconds=estimated_time,
    )


@router.get("/full-process/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str = Path(..., description="Job identifier"),
    _: str = Depends(verify_api_key),
) -> JobStatusResponse:
    """
    Check the status of a full process job.

    Checks if the output file exists in the bucket to determine status.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator service not initialized")

    if not settings.output_bucket:
        raise HTTPException(
            status_code=500, detail="Output bucket not configured. Set OUTPUT_BUCKET in environment."
        )

    # Check if output file exists
    output_file_name = f"results_{job_id}.json"
    file_exists = await orchestrator.gcs_client.file_exists(
        settings.output_bucket, output_file_name
    )

    if file_exists:
        status = "completed"
    else:
        status = "processing"

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        output_bucket=settings.output_bucket,
        output_file_name=output_file_name if file_exists else None,
    )

