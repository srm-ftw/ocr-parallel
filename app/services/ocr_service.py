"""OCR service that orchestrates the complete OCR process."""

import asyncio
import concurrent.futures
import logging
import time
from typing import Dict, List, Optional, Tuple

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
        
        # PDF cache to avoid downloading same file multiple times
        self._pdf_cache: Dict[Tuple[str, str], bytes] = {}  # (bucket, file) -> PDF bytes
        self._cache_locks: Dict[Tuple[str, str], asyncio.Lock] = {}  # Locks per file
        
        # Thread pool for PDF extraction (CPU-bound operation)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

    async def _get_pdf_with_cache(self, bucket: str, file: str) -> bytes:
        """
        Get PDF from cache or download if not cached.
        
        Uses locks to prevent multiple concurrent downloads of the same file.
        
        Args:
            bucket: GCS bucket name
            file: File name in bucket
            
        Returns:
            bytes: PDF file content
        """
        cache_key = (bucket, file)
        
        # Check cache first (fast path)
        if cache_key in self._pdf_cache:
            logger.debug(f"PDF cache hit for {file}")
            return self._pdf_cache[cache_key]
        
        # Get or create lock for this file
        if cache_key not in self._cache_locks:
            self._cache_locks[cache_key] = asyncio.Lock()
        
        # Acquire lock to prevent concurrent downloads
        async with self._cache_locks[cache_key]:
            # Double-check after acquiring lock (another task might have downloaded it)
            if cache_key in self._pdf_cache:
                logger.debug(f"PDF cache hit for {file} (after lock)")
                return self._pdf_cache[cache_key]
            
            # Download PDF (only one task per file will reach here)
            logger.debug(f"Downloading PDF {file} (not in cache)")
            pdf_bytes = await self.gcs_client.download_pdf(bucket, file)
            self._pdf_cache[cache_key] = pdf_bytes
            logger.debug(f"Cached PDF {file} ({len(pdf_bytes)} bytes)")
            return pdf_bytes

    async def _extract_page_async(self, pdf_bytes: bytes, page_number: int) -> bytes:
        """
        Extract page from PDF in thread pool to avoid blocking event loop.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_number: Page number to extract (1-indexed)
            
        Returns:
            bytes: Single-page PDF as bytes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            extract_page_from_pdf,
            pdf_bytes,
            page_number
        )

    async def _process_single_page_with_pdf(
        self,
        request: OCRPageRequest,
        unique_key: Tuple[str, str, str],
        pdf_bytes: bytes,
        gemini_client: GeminiClient,
        page_index: int,
        total_pages: int,
        model_name: str,
    ) -> Tuple[Tuple[str, str, str], OCRPageResponse, dict]:
        """
        Process a single page using a pre-loaded PDF (for grouped processing).
        
        Args:
            request: The OCR page request
            unique_key: Unique key for this page
            pdf_bytes: Pre-loaded PDF bytes (already in memory)
            
        Returns:
            Tuple of (unique_key, OCRPageResponse, metrics_dict)
        """
        start_time = time.time()
        metrics = {
            "success": False,
            "error_type": None,
            "processing_time": 0.0,
            "gcs_time": 0.0,  # Will be 0 since PDF is pre-loaded
            "pdf_extraction_time": 0.0,
            "gemini_time": 0.0,
            "response_size": 0,
            "pdf_size": len(pdf_bytes),
        }

        async with self.concurrency_limiter:
            try:
                logger.debug(f"Processing page {request.page} from {request.file}")

                # Step 1: PDF is already loaded (no GCS download needed)
                metrics["gcs_time"] = 0.0

                # Step 2: Extract specific page (async in thread pool)
                pdf_start = time.time()
                page_number = int(request.page)
                page_pdf_bytes = await self._extract_page_async(pdf_bytes, page_number)
                metrics["pdf_extraction_time"] = time.time() - pdf_start

                # Step 3: Call Gemini API for OCR
                gemini_start = time.time()
                texto_ocr = await self.gemini_client.extract_text_from_pdf(page_pdf_bytes)
                metrics["gemini_time"] = time.time() - gemini_start
                metrics["response_size"] = len(texto_ocr)

                metrics["success"] = True
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed

                logger.info(
                    f"[{page_index + 1}/{total_pages}][success][model:{model_name}] "
                    f"Page {request.page} from {request.file} processed in {elapsed:.2f}s "
                    f"(PDF: {metrics['pdf_extraction_time']:.2f}s, "
                    f"Gemini: {metrics['gemini_time']:.2f}s, response: {len(texto_ocr)} chars)"
                )

                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=texto_ocr),
                    metrics,
                )

            except PDFProcessingException as e:
                metrics["error_type"] = "PDF"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"[{page_index + 1}/{total_pages}][error:PDF][model:{model_name}] "
                    f"Page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}"
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error PDF: {str(e)}]"),
                    metrics,
                )
            except GeminiAPIException as e:
                metrics["error_type"] = "Gemini"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"[{page_index + 1}/{total_pages}][error:Gemini][model:{model_name}] "
                    f"Page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}"
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error OCR: {str(e)}]"),
                    metrics,
                )
            except Exception as e:
                metrics["error_type"] = "Unknown"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"[{page_index + 1}/{total_pages}][error:Unknown][model:{model_name}] "
                    f"Page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error: {str(e)}]"),
                    metrics,
                )

    async def _process_single_page(
        self, request: OCRPageRequest, unique_key: Tuple[str, str, str], gemini_client: GeminiClient
    ) -> Tuple[Tuple[str, str, str], OCRPageResponse, dict]:
        """
        Process a single page: download, extract, and OCR.

        Args:
            request: The OCR page request
            unique_key: Unique key for this page

        Returns:
            Tuple of (unique_key, OCRPageResponse, metrics_dict)
        """
        start_time = time.time()
        metrics = {
            "success": False,
            "error_type": None,
            "processing_time": 0.0,
            "gcs_time": 0.0,
            "pdf_extraction_time": 0.0,
            "gemini_time": 0.0,
            "response_size": 0,
            "pdf_size": 0,
        }

        async with self.concurrency_limiter:
            try:
                logger.debug(f"Processing page {request.page} from {request.file}")

                # Step 1: Download PDF from GCS (with cache)
                gcs_start = time.time()
                pdf_bytes = await self._get_pdf_with_cache(request.bucket, request.file)
                metrics["gcs_time"] = time.time() - gcs_start
                metrics["pdf_size"] = len(pdf_bytes)

                # Step 2: Extract specific page (async in thread pool)
                pdf_start = time.time()
                page_number = int(request.page)
                page_pdf_bytes = await self._extract_page_async(pdf_bytes, page_number)
                metrics["pdf_extraction_time"] = time.time() - pdf_start

                # Step 3: Call Gemini API for OCR
                gemini_start = time.time()
                texto_ocr = await self.gemini_client.extract_text_from_pdf(page_pdf_bytes)
                metrics["gemini_time"] = time.time() - gemini_start
                metrics["response_size"] = len(texto_ocr)

                metrics["success"] = True
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed

                logger.info(
                    f"Page {request.page} from {request.file} processed successfully in {elapsed:.2f}s "
                    f"(GCS: {metrics['gcs_time']:.2f}s, PDF: {metrics['pdf_extraction_time']:.2f}s, "
                    f"Gemini: {metrics['gemini_time']:.2f}s, response: {len(texto_ocr)} chars)"
                )

                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=texto_ocr),
                    metrics,
                )

            except GCSException as e:
                metrics["error_type"] = "GCS"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"GCS error processing page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}"
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error GCS: {str(e)}]"),
                    metrics,
                )
            except PDFProcessingException as e:
                metrics["error_type"] = "PDF"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"PDF processing error for page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}"
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error PDF: {str(e)}]"),
                    metrics,
                )
            except GeminiAPIException as e:
                metrics["error_type"] = "Gemini"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"Gemini API error for page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}"
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error OCR: {str(e)}]"),
                    metrics,
                )
            except Exception as e:
                metrics["error_type"] = "Unknown"
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"Unexpected error processing page {request.page} from {request.file} "
                    f"after {elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                return (
                    unique_key,
                    OCRPageResponse(page=request.page, texto_ocr=f"[Error: {str(e)}]"),
                    metrics,
                )

    async def process_pages_async(
        self, requests: List[OCRPageRequest], model: Optional[str] = None
    ) -> List[OCRPageResponse]:
        """
        Process multiple pages in parallel with deduplication.

        Args:
            requests: List of OCR page requests
            model: Optional Gemini model name. If not provided, uses default client.

        Returns:
            List of OCR responses in the same order as input
        """
        if not requests:
            return []

        # Use provided model or default client
        model_to_use = model or DEFAULT_GEMINI_MODEL
        use_temporary_client = model is not None
        gemini_client = (
            GeminiClient(model_name=model_to_use) if use_temporary_client else self.gemini_client
        )

        try:
            batch_start_time = time.time()
            logger.info(
                f"Processing {len(requests)} page requests (model: {model_to_use})"
            )

            # Step 1: Deduplication - identify unique pages
            dedup_start = time.time()
            unique_requests, index_mapping = deduplicate_requests(requests)
            dedup_time = time.time() - dedup_start
            deduplication_savings = len(requests) - len(unique_requests) if unique_requests else 0

            if not unique_requests:
                logger.warning("No unique requests after deduplication")
                return [OCRPageResponse(page="", texto_ocr="")] * len(requests)

            logger.info(
                f"Deduplication: {len(unique_requests)} unique pages from {len(requests)} requests "
                f"({deduplication_savings} duplicates, {dedup_time:.3f}s)"
            )

            # Step 2: Group requests by PDF file to optimize memory usage
            # This allows us to load one PDF at a time, process all its pages, then free memory
            requests_by_pdf: Dict[Tuple[str, str], List[OCRPageRequest]] = {}
            for req in unique_requests:
                pdf_key = (req.bucket, req.file)
                if pdf_key not in requests_by_pdf:
                    requests_by_pdf[pdf_key] = []
                requests_by_pdf[pdf_key].append(req)

            logger.info(
                f"Grouped into {len(requests_by_pdf)} unique PDFs "
                f"(avg {len(unique_requests) / len(requests_by_pdf):.1f} pages per PDF)"
            )

            # Step 3: Process PDFs one at a time (or in small batches) to manage memory
            # Process all pages of each PDF in parallel, then free the PDF from memory
            all_results: List[Tuple[Tuple[str, str, str], OCRPageResponse, dict]] = []
            total_unique_pages = len(unique_requests)
            global_page_index = 0
            
            for pdf_key, pdf_requests in requests_by_pdf.items():
                bucket, file = pdf_key
                logger.debug(f"Processing PDF {file} with {len(pdf_requests)} pages")
                
                try:
                    # Download PDF once for all pages
                    pdf_bytes = await self._get_pdf_with_cache(bucket, file)
                    
                    # Process all pages of this PDF in parallel
                    pdf_tasks = []
                    for req in pdf_requests:
                        unique_key = create_unique_key(req)
                        task = self._process_single_page_with_pdf(
                            req, unique_key, pdf_bytes, gemini_client, global_page_index, total_unique_pages, model_to_use
                        )
                        pdf_tasks.append(task)
                        global_page_index += 1
                    
                    # Wait for all pages of this PDF to complete
                    pdf_results = await asyncio.gather(*pdf_tasks, return_exceptions=True)
                    all_results.extend(pdf_results)
                    
                except GCSException as e:
                    # If PDF download fails, create error responses for all pages
                    logger.error(f"Failed to download PDF {file}: {e}")
                    for req in pdf_requests:
                        unique_key = create_unique_key(req)
                        error_response = (
                            unique_key,
                            OCRPageResponse(
                                page=req.page,
                                texto_ocr=f"[Error GCS: Failed to download PDF: {str(e)}]"
                            ),
                            {
                                "success": False,
                                "error_type": "GCS",
                                "processing_time": 0.0,
                                "gcs_time": 0.0,
                                "pdf_extraction_time": 0.0,
                                "gemini_time": 0.0,
                                "response_size": 0,
                                "pdf_size": 0,
                            },
                        )
                        all_results.append(error_response)
                finally:
                    # Always free PDF from cache after processing (even on error)
                    if pdf_key in self._pdf_cache:
                        pdf_size = len(self._pdf_cache[pdf_key])
                        del self._pdf_cache[pdf_key]
                        logger.debug(f"Freed PDF {file} from memory ({pdf_size} bytes)")
            
            results = all_results

            # Step 4: Build results dictionary and collect metrics
            unique_results: Dict[Tuple[str, str, str], OCRPageResponse] = {}
            all_metrics = []
            success_count = 0
            error_count = 0
            error_types = {"GCS": 0, "PDF": 0, "Gemini": 0, "Unknown": 0}
            total_response_size = 0
            total_pdf_size = 0
            error_list = []

            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    logger.error(f"Task failed with exception: {result}", exc_info=True)
                    continue

                # Type check: result should be a tuple of (key, response, metrics)
                try:
                    key, response, metrics = result  # type: ignore
                except (ValueError, TypeError) as e:
                    error_count += 1
                    logger.error(f"Invalid result format: {result}, error: {e}")
                    continue

                unique_results[key] = response
                all_metrics.append(metrics)
                total_response_size += metrics.get("response_size", 0)
                total_pdf_size += metrics.get("pdf_size", 0)

                if metrics.get("success"):
                    success_count += 1
                else:
                    error_count += 1
                    error_type = metrics.get("error_type", "Unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    # Store error details for summary
                    bucket, file, page = key
                    error_list.append({
                        "bucket": bucket,
                        "file": file,
                        "page": page,
                        "error_type": error_type,
                    })

            # Step 5: Map results back to all original positions
            final_results = map_results_back(unique_results, index_mapping, len(requests))

            # Calculate aggregate metrics
            batch_elapsed = time.time() - batch_start_time
            processing_times = [m.get("processing_time", 0) for m in all_metrics if m]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            min_processing_time = min(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0

            gcs_times = [m.get("gcs_time", 0) for m in all_metrics if m.get("gcs_time")]
            avg_gcs_time = sum(gcs_times) / len(gcs_times) if gcs_times else 0

            pdf_times = [m.get("pdf_extraction_time", 0) for m in all_metrics if m.get("pdf_extraction_time")]
            avg_pdf_time = sum(pdf_times) / len(pdf_times) if pdf_times else 0

            gemini_times = [m.get("gemini_time", 0) for m in all_metrics if m.get("gemini_time")]
            avg_gemini_time = sum(gemini_times) / len(gemini_times) if gemini_times else 0

            # Log comprehensive metrics
            error_summary = ", ".join([f"{k}: {v}" for k, v in error_types.items() if v > 0])
            logger.info(
                f"Completed processing {len(requests)} requests ({len(unique_requests)} unique) "
                f"in {batch_elapsed:.2f}s | "
                f"Model: {model_to_use} | "
                f"Success: {success_count} | Errors: {error_count} ({error_summary}) | "
                f"Avg time/page: {avg_processing_time:.2f}s (min: {min_processing_time:.2f}s, "
                f"max: {max_processing_time:.2f}s) | "
                f"Avg GCS: {avg_gcs_time:.2f}s | Avg PDF: {avg_pdf_time:.2f}s | "
                f"Avg Gemini: {avg_gemini_time:.2f}s | "
                f"Total response: {total_response_size:,} chars | "
                f"Total PDF size: {total_pdf_size:,} bytes | "
                f"Deduplication savings: {deduplication_savings} pages"
            )

            # Log detailed list of errors
            if error_list:
                error_details = []
                for item in error_list[:20]:  # Limit to first 20 for readability
                    error_details.append(f"{item['file']}:page{item['page']} ({item['error_type']})")
                error_summary_text = ", ".join(error_details)
                if len(error_list) > 20:
                    error_summary_text += f" ... and {len(error_list) - 20} more"
                logger.error(f"ERRORS ({len(error_list)}): {error_summary_text}")

            return final_results
        finally:
            # Close temporary client if we created one
            if use_temporary_client:
                await gemini_client.close()

    async def close(self):
        """Close all client connections and cleanup resources."""
        await self.gemini_client.close()
        self._executor.shutdown(wait=True)
        # Clear cache to free memory
        self._pdf_cache.clear()
        self._cache_locks.clear()

