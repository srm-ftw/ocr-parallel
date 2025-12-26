"""Orchestrator for full process pipeline (OCR + Extraction)."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from app.clients.gcs_client import GCSClient
from app.config import settings
from app.core.exceptions import GCSException
from app.models.full_process import (
    BatchStats,
    EscrituraRequest,
    EscrituraResult,
    EscrituraStats,
    ExtractionPhaseStats,
    InmuebleInput,
)
from app.services.full_extractor_service import FullExtractorService
from app.services.full_ocr_service import FullOCRService

logger = logging.getLogger(__name__)


class FullProcessOrchestrator:
    """Orchestrator for coordinating OCR and extraction pipeline."""

    def __init__(self):
        """Initialize orchestrator with services."""
        self.full_ocr_service = FullOCRService()
        self.full_extractor_service = FullExtractorService()
        self.gcs_client = GCSClient()

    async def process_batch(
        self, requests: List[EscrituraRequest], output_bucket: str, output_file_name: str
    ) -> None:
        """
        Process a batch of escrituras and upload results to GCS.

        Args:
            requests: List of escritura processing requests
            output_bucket: GCS bucket name for output
            output_file_name: File name for output JSON
        """
        batch_start_time = time.time()
        job_id = output_file_name.replace("results_", "").replace(".json", "")
        
        logger.info(
            f"Starting batch processing: {len(requests)} escrituras, "
            f"output: gs://{output_bucket}/{output_file_name}"
        )

        # Process escrituras with controlled parallelism
        # Use semaphore to limit concurrent escrituras (to manage memory)
        max_concurrent = settings.full_process_max_concurrent_escrituras
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_escritura(request: EscrituraRequest) -> Tuple[EscrituraResult, EscrituraStats]:
            """Process a single escritura with semaphore control."""
            async with semaphore:
                return await self._process_escritura(request)

        # Create tasks for all escrituras
        tasks = [process_single_escritura(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results and errors
        final_results: List[EscrituraResult] = []
        escritura_stats_list: List[EscrituraStats] = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing escritura {requests[i].escritura_file_name}: {result}",
                    exc_info=True,
                )
                # Create error result
                error_result = self._create_error_result(requests[i], str(result))
                final_results.append(error_result)
                # Create error stats
                error_stats = EscrituraStats(
                    escritura_file_name=requests[i].escritura_file_name,
                    total_pages=0,
                    total_inmuebles=len(requests[i].inmuebles),
                    found_count=0,
                    not_found_count=len(requests[i].inmuebles),
                    ocr_time_seconds=0.0,
                    extraction_time_seconds=0.0,
                    total_processing_time_seconds=0.0,
                    extraction_stats=ExtractionPhaseStats(
                        phase1_text_localization_time_seconds=0.0,
                        phase2_llm_refinement_time_seconds=0.0,
                        phase3_preparation_time_seconds=0.0,
                        phase3_extraction_time_seconds=0.0,
                        found_by_text_regex=0,
                        found_by_refinement_llm=0,
                        total_found=0,
                        total_not_found=len(requests[i].inmuebles),
                    ),
                )
                escritura_stats_list.append(error_stats)
            else:
                result_obj, stats_obj = result
                final_results.append(result_obj)
                escritura_stats_list.append(stats_obj)

        # Build output structure (upload as direct list, not wrapped)
        output_data = [result.model_dump() for result in final_results]

        # Upload to GCS
        try:
            upload_start = time.time()
            await self.gcs_client.upload_json(output_bucket, output_file_name, output_data)
            upload_time = time.time() - upload_start

            batch_time = time.time() - batch_start_time
            logger.info(
                f"Batch processing complete in {batch_time:.2f}s "
                f"(upload: {upload_time:.2f}s): "
                f"gs://{output_bucket}/{output_file_name}"
            )
        except GCSException as e:
            logger.error(f"Failed to upload results to GCS: {e}", exc_info=True)
            raise

        # Build and upload statistics
        if settings.stats_bucket:
            try:
                batch_stats = self._build_batch_stats(
                    job_id, batch_start_time, requests, final_results, escritura_stats_list
                )
                stats_file_name = f"stats_{job_id}.json"
                # Use model_dump with mode='json' to ensure proper serialization of nested models
                stats_dict = batch_stats.model_dump(mode='json')
                await self.gcs_client.upload_json(
                    settings.stats_bucket, stats_file_name, stats_dict
                )
                logger.info(
                    f"Statistics uploaded to gs://{settings.stats_bucket}/{stats_file_name}"
                )
            except Exception as e:
                logger.error(f"Failed to upload statistics: {e}", exc_info=True)
                # Don't raise - stats failure shouldn't fail the batch

    async def _process_escritura(self, request: EscrituraRequest) -> Tuple[EscrituraResult, EscrituraStats]:
        """
        Process a single escritura: OCR + Extraction.

        Args:
            request: Escritura processing request

        Returns:
            Tuple of (EscrituraResult with extraction results and metadata, EscrituraStats)
        """
        escritura_start = time.time()
        logger.info(
            f"Processing escritura: {request.escritura_file_name} "
            f"({len(request.inmuebles)} inmuebles)"
        )

        try:
            # Step 1: Full OCR
            ocr_start = time.time()
            ocr_result = await self.full_ocr_service.process_escritura(
                request.escritura_bucket, request.escritura_file_name
            )
            ocr_time = time.time() - ocr_start

            logger.info(
                f"OCR complete for {request.escritura_file_name}: "
                f"{ocr_result.total_pages} pages in {ocr_time:.2f}s"
            )

            # Step 2: Extract inmuebles
            extract_start = time.time()
            inmueble_inputs = [
                InmuebleInput(inmueble=inm.inmueble, folio=inm.folio) for inm in request.inmuebles
            ]
            extraction_results, extraction_stats = await self.full_extractor_service.extract_inmuebles(
                ocr_result, inmueble_inputs
            )
            extract_time = time.time() - extract_start

            total_time = time.time() - escritura_start

            logger.info(
                f"Extraction complete for {request.escritura_file_name}: "
                f"{len(extraction_results)} inmuebles in {extract_time:.2f}s "
                f"(total: {total_time:.2f}s)"
            )

            # Build result
            from app.models.full_process import EscrituraMetadata

            result = EscrituraResult(
                escritura_file_name=request.escritura_file_name,
                resultados=extraction_results,
                metadata=EscrituraMetadata(
                    total_pages=ocr_result.total_pages,
                    processing_time_seconds=total_time,
                    ocr_time_seconds=ocr_time,
                    extraction_time_seconds=extract_time,
                ),
            )

            # Build stats
            found_count = sum(1 for r in extraction_results if r.status == "found")
            not_found_count = len(extraction_results) - found_count
            
            stats = EscrituraStats(
                escritura_file_name=request.escritura_file_name,
                total_pages=ocr_result.total_pages,
                total_inmuebles=len(request.inmuebles),
                found_count=found_count,
                not_found_count=not_found_count,
                ocr_time_seconds=ocr_time,
                extraction_time_seconds=extract_time,
                total_processing_time_seconds=total_time,
                extraction_stats=ExtractionPhaseStats(
                    phase1_text_localization_time_seconds=extraction_stats.phase1_text_localization_time_seconds,
                    phase2_llm_refinement_time_seconds=extraction_stats.phase2_llm_refinement_time_seconds,
                    phase3_preparation_time_seconds=extraction_stats.phase3_preparation_time_seconds,
                    phase3_extraction_time_seconds=extraction_stats.phase3_extraction_time_seconds,
                    found_by_text_regex=extraction_stats.found_by_text_regex,
                    found_by_refinement_llm=extraction_stats.found_by_refinement_llm,
                    total_found=extraction_stats.total_found,
                    total_not_found=extraction_stats.total_not_found,
                ),
            )

            return result, stats

        except Exception as e:
            logger.error(
                f"Error processing escritura {request.escritura_file_name}: {e}",
                exc_info=True,
            )
            # Re-raise to be handled by process_batch
            raise

    def _build_batch_stats(
        self,
        job_id: str,
        batch_start_time: float,
        requests: List[EscrituraRequest],
        final_results: List[EscrituraResult],
        escritura_stats_list: List[EscrituraStats],
    ) -> BatchStats:
        """
        Build aggregated batch statistics.

        Args:
            job_id: Job identifier
            batch_start_time: Batch start timestamp
            requests: Original requests
            final_results: Processing results
            escritura_stats_list: Statistics per escritura

        Returns:
            BatchStats with aggregated statistics
        """
        total_processing_time = time.time() - batch_start_time
        
        # Aggregate totals
        total_escrituras = len(requests)
        total_inmuebles = sum(len(req.inmuebles) for req in requests)
        total_found = sum(stats.found_count for stats in escritura_stats_list)
        total_not_found = sum(stats.not_found_count for stats in escritura_stats_list)
        total_ocr_time = sum(stats.ocr_time_seconds for stats in escritura_stats_list)
        total_extraction_time = sum(stats.extraction_time_seconds for stats in escritura_stats_list)
        
        # Aggregate localization method counts
        localization_method_counts: Dict[str, int] = {
            "text-regex": 0,
            "refinement-llm": 0,
        }
        for result in final_results:
            for inmueble_result in result.resultados:
                if inmueble_result.localization_method:
                    if inmueble_result.localization_method in localization_method_counts:
                        localization_method_counts[inmueble_result.localization_method] += 1
        
        # Aggregate extraction phase stats
        aggregated_extraction_stats = ExtractionPhaseStats(
            phase1_text_localization_time_seconds=sum(
                stats.extraction_stats.phase1_text_localization_time_seconds
                for stats in escritura_stats_list
            ),
            phase2_llm_refinement_time_seconds=sum(
                stats.extraction_stats.phase2_llm_refinement_time_seconds
                for stats in escritura_stats_list
            ),
            phase3_preparation_time_seconds=sum(
                stats.extraction_stats.phase3_preparation_time_seconds
                for stats in escritura_stats_list
            ),
            phase3_extraction_time_seconds=sum(
                stats.extraction_stats.phase3_extraction_time_seconds
                for stats in escritura_stats_list
            ),
            found_by_text_regex=sum(
                stats.extraction_stats.found_by_text_regex for stats in escritura_stats_list
            ),
            found_by_refinement_llm=sum(
                stats.extraction_stats.found_by_refinement_llm for stats in escritura_stats_list
            ),
            total_found=total_found,
            total_not_found=total_not_found,
        )
        
        return BatchStats(
            job_id=job_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_escrituras=total_escrituras,
            total_inmuebles=total_inmuebles,
            total_found=total_found,
            total_not_found=total_not_found,
            total_processing_time_seconds=total_processing_time,
            total_ocr_time_seconds=total_ocr_time,
            total_extraction_time_seconds=total_extraction_time,
            localization_method_counts=localization_method_counts,
            escritura_stats=escritura_stats_list,
            aggregated_extraction_stats=aggregated_extraction_stats,
        )

    def _create_error_result(self, request: EscrituraRequest, error_message: str) -> EscrituraResult:
        """
        Create an error result for a failed escritura.

        Args:
            request: Original request
            error_message: Error message

        Returns:
            EscrituraResult with error status
        """
        from app.models.full_process import EscrituraMetadata, InmuebleResult

        # Create error results for all inmuebles
        error_results = [
            InmuebleResult(
                inmueble=inm.inmueble,
                folio=inm.folio,
                status="error",
                paginas=None,
                text=f"[Error processing escritura: {error_message}]",
            )
            for inm in request.inmuebles
        ]

        return EscrituraResult(
            escritura_file_name=request.escritura_file_name,
            resultados=error_results,
            metadata=EscrituraMetadata(
                total_pages=0,
                processing_time_seconds=0.0,
                ocr_time_seconds=0.0,
                extraction_time_seconds=0.0,
            ),
        )

    async def close(self):
        """Close all service connections."""
        await self.full_ocr_service.close()
        await self.full_extractor_service.close()

