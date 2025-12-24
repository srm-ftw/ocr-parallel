"""OCR service that orchestrates the complete OCR process."""

import asyncio
import logging
from typing import Dict, List, Tuple

from app.clients.gcs_client import GCSClient
from app.clients.gemini_client import GeminiClient
from app.config import settings
from app.core.exceptions import GCSException, GeminiAPIException, PDFProcessingException
from app.models.ocr import OCRPageRequest, OCRPageResponse
from app.utils.deduplication import create_unique_key, deduplicate_requests, map_results_back
from app.utils.pdf_utils import extract_page_from_pdf
from app.utils.rate_limiter import ConcurrencyLimiter

logger = logging.getLogger(__name__)


class OCRService:
    """Service for processing OCR requests with deduplication and parallelization."""

    def __init__(self):
        """Initialize OCR service with clients."""
        self.gcs_client = GCSClient()
        self.gemini_client = GeminiClient()
        self.concurrency_limiter = ConcurrencyLimiter(settings.max_concurrent_requests)

    async def _process_single_page(
        self, request: OCRPageRequest, unique_key: Tuple[str, str, str]
    ) -> Tuple[Tuple[str, str, str], OCRPageResponse]:
        """
        Process a single page: download, extract, and OCR.

        Args:
            request: The OCR page request
            unique_key: Unique key for this page

        Returns:
            Tuple of (unique_key, OCRPageResponse)
        """
        async with self.concurrency_limiter:
            try:
                logger.debug(f"Processing page {request.page} from {request.file}")

                # Step 1: Download PDF from GCS
                pdf_bytes = await self.gcs_client.download_pdf(request.bucket, request.file)

                # Step 2: Extract specific page
                page_number = int(request.page)
                page_pdf_bytes = extract_page_from_pdf(pdf_bytes, page_number)

                # Step 3: Call Gemini API for OCR
                texto_ocr = await self.gemini_client.extract_text_from_pdf(page_pdf_bytes)

                logger.debug(f"Successfully processed page {request.page} from {request.file}")

                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=texto_ocr),
                )

            except GCSException as e:
                logger.error(f"GCS error processing page {request.page}: {e}")
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error GCS: {str(e)}]"),
                )
            except PDFProcessingException as e:
                logger.error(f"PDF processing error for page {request.page}: {e}")
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error PDF: {str(e)}]"),
                )
            except GeminiAPIException as e:
                logger.error(f"Gemini API error for page {request.page}: {e}")
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error OCR: {str(e)}]"),
                )
            except Exception as e:
                logger.error(f"Unexpected error processing page {request.page}: {e}")
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error: {str(e)}]"),
                )

    async def process_pages_async(self, requests: List[OCRPageRequest]) -> List[OCRPageResponse]:
        """
        Process multiple pages in parallel with deduplication.

        Args:
            requests: List of OCR page requests

        Returns:
            List of OCR responses in the same order as input
        """
        if not requests:
            return []

        logger.info(f"Processing {len(requests)} page requests")

        # Step 1: Deduplication - identify unique pages
        unique_requests, index_mapping = deduplicate_requests(requests)

        if not unique_requests:
            logger.warning("No unique requests after deduplication")
            return [OCRPageResponse(page="", texto_ocr="")] * len(requests)

        # Step 2: Create tasks for unique pages only
        tasks = []
        for unique_request in unique_requests:
            unique_key = create_unique_key(unique_request)
            task = self._process_single_page(unique_request, unique_key)
            tasks.append(task)

        # Step 3: Process all unique pages in parallel
        logger.info(f"Processing {len(tasks)} unique pages in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: Build results dictionary
        unique_results: Dict[Tuple[str, str, str], OCRPageResponse] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue

            key, response = result
            unique_results[key] = response

        # Step 5: Map results back to all original positions
        final_results = map_results_back(unique_results, index_mapping, len(requests))

        logger.info(f"Completed processing {len(requests)} requests ({len(unique_requests)} unique)")

        return final_results

    async def close(self):
        """Close all client connections."""
        await self.gemini_client.close()

