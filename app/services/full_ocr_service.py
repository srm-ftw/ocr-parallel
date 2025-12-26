"""Full OCR service for processing entire escrituras."""

import asyncio
import concurrent.futures
import logging
import time
from typing import Dict, List, Optional, Tuple

from app.clients.gcs_client import GCSClient
from app.clients.gemini_client import GeminiClient
from app.config import DEFAULT_GEMINI_MODEL, settings
from app.core.exceptions import GCSException, GeminiAPIException, PDFProcessingException
from app.models.full_process import ChunkInfo, EscrituraOCRResult, PaginaOCR
from app.utils.chunking import create_chunks_with_overlap
from app.utils.pdf_utils import extract_page_from_pdf, get_pdf_page_count
from app.utils.rate_limiter import ConcurrencyLimiter
from app.utils.text_cleaning import clean_ocr_text

logger = logging.getLogger(__name__)


class FullOCRService:
    """Service for processing complete escrituras with OCR."""

    def __init__(self):
        """Initialize Full OCR service with clients."""
        self.gcs_client = GCSClient()
        self.gemini_client = GeminiClient()
        self.concurrency_limiter = ConcurrencyLimiter(settings.max_concurrent_requests)

        # PDF cache to avoid downloading same file multiple times
        self._pdf_cache: Dict[Tuple[str, str], bytes] = {}
        self._cache_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

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
            # Double-check after acquiring lock
            if cache_key in self._pdf_cache:
                logger.debug(f"PDF cache hit for {file} (after lock)")
                return self._pdf_cache[cache_key]

            # Download PDF
            logger.debug(f"Downloading PDF {file} (not in cache)")
            pdf_bytes = await self.gcs_client.download_pdf(bucket, file)
            self._pdf_cache[cache_key] = pdf_bytes
            logger.debug(f"Cached PDF {file} ({len(pdf_bytes)} bytes)")
            return pdf_bytes

    async def _load_pdf_bytes(
        self, bucket: Optional[str] = None, file_name: Optional[str] = None, local_path: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Load PDF bytes from either GCS or local file.

        Args:
            bucket: GCS bucket name (if loading from GCS)
            file_name: PDF file name in bucket (if loading from GCS)
            local_path: Local file path (if loading from local file)

        Returns:
            Tuple of (pdf_bytes, display_name)

        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If local file doesn't exist
        """
        if local_path:
            # Load from local file
            from pathlib import Path

            file_path_obj = Path(local_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"PDF file not found: {local_path}")

            with open(local_path, "rb") as f:
                pdf_bytes = f.read()

            return pdf_bytes, file_path_obj.name

        elif bucket and file_name:
            # Load from GCS
            pdf_bytes = await self._get_pdf_with_cache(bucket, file_name)
            return pdf_bytes, file_name

        else:
            raise ValueError("Must provide either (bucket, file_name) or local_path")

    async def process_escritura(
        self,
        bucket: Optional[str] = None,
        file_name: Optional[str] = None,
        local_path: Optional[str] = None,
        model: Optional[str] = None,
    ) -> EscrituraOCRResult:
        """
        Process a complete escritura: OCR all pages, create chunks.

        Can load PDF from either GCS or local file:
        - From GCS: provide bucket and file_name
        - From local: provide local_path

        Args:
            bucket: GCS bucket name (if loading from GCS)
            file_name: PDF file name in bucket (if loading from GCS)
            local_path: Local file path (if loading from local file)
            model: Optional Gemini model name

        Returns:
            EscrituraOCRResult with all pages OCR'd and chunks created

        Raises:
            ValueError: If parameters are invalid
        """
        start_time = time.time()
        model_to_use = model or DEFAULT_GEMINI_MODEL
        use_temporary_client = model is not None
        gemini_client = (
            GeminiClient(model_name=model_to_use) if use_temporary_client else self.gemini_client
        )

        try:
            # Load PDF bytes (from GCS or local)
            load_start = time.time()
            pdf_bytes, display_name = await self._load_pdf_bytes(bucket, file_name, local_path)
            load_time = time.time() - load_start

            source = f"local file {display_name}" if local_path else f"bucket {bucket}/{display_name}"
            logger.info(
                f"Starting full OCR for {source} (model: {model_to_use})"
            )
            logger.info(f"Loaded {display_name} ({len(pdf_bytes)} bytes) in {load_time:.2f}s")

            # Get total page count
            total_pages = get_pdf_page_count(pdf_bytes)
            logger.info(f"PDF {display_name} has {total_pages} pages")

            # Process all pages in parallel
            ocr_start = time.time()
            logger.info(f"Processing {total_pages} pages in parallel...")

            tasks = []
            for page_num in range(1, total_pages + 1):
                task = self._process_single_page(
                    pdf_bytes,
                    page_num,
                    gemini_client,
                    page_num - 1,
                    total_pages,
                    model_to_use,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build paginas list (sorted by page number)
            paginas: List[PaginaOCR] = []
            error_count = 0
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    logger.error(f"Page processing failed: {result}", exc_info=True)
                    continue

                # Type check: result should be a tuple of (page_number, PaginaOCR)
                if not isinstance(result, tuple) or len(result) != 2:
                    error_count += 1
                    logger.error(f"Invalid result format: {result}")
                    continue

                page_num, pagina = result
                paginas.append(pagina)

            ocr_time = time.time() - ocr_start

            # Sort by page number (should already be sorted, but just in case)
            paginas.sort(key=lambda p: p.page)

            logger.info(
                f"OCR complete: {len(paginas)} pages processed in {ocr_time:.2f}s "
                f"({error_count} errors, model: {model_to_use})"
            )

            # Create chunks with overlap
            chunk_start = time.time()
            chunks = create_chunks_with_overlap(
                paginas,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            chunk_time = time.time() - chunk_start

            total_time = time.time() - start_time
            logger.info(
                f"Full OCR complete for {display_name}: {total_time:.2f}s total "
                f"(load: {load_time:.2f}s, OCR: {ocr_time:.2f}s, "
                f"chunking: {chunk_time:.2f}s, {len(chunks)} chunks created)"
            )

            return EscrituraOCRResult(
                escritura_bucket=bucket or "local",
                escritura_file_name=display_name,
                total_pages=total_pages,
                paginas=paginas,
                chunks=chunks,
            )

        finally:
            # Close temporary client if we created one
            if use_temporary_client:
                await gemini_client.close()

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
            self._executor, extract_page_from_pdf, pdf_bytes, page_number
        )

    async def _process_single_page(
        self,
        pdf_bytes: bytes,
        page_number: int,
        gemini_client: GeminiClient,
        page_index: int,
        total_pages: int,
        model_name: str,
    ) -> Tuple[int, PaginaOCR]:
        """
        Process a single page: extract and OCR.

        Args:
            pdf_bytes: Pre-loaded PDF bytes
            page_number: Page number to process (1-indexed)
            gemini_client: Gemini client to use
            page_index: Index for logging (0-based)
            total_pages: Total number of pages
            model_name: Model name for logging

        Returns:
            Tuple of (page_number, PaginaOCR)
        """
        try:
            logger.debug(f"Processing page {page_number}/{total_pages}")

            # Extract page
            page_pdf_bytes = await self._extract_page_async(pdf_bytes, page_number)

            # Call Gemini API for OCR
            async with self.concurrency_limiter:
                texto_ocr_raw = await gemini_client.extract_text_from_pdf(page_pdf_bytes)

            # Clean OCR text (remove headers, footers, artifacts)
            texto_ocr = clean_ocr_text(texto_ocr_raw)

            logger.debug(
                f"[{page_index + 1}/{total_pages}][success][model:{model_name}] "
                f"Page {page_number} processed ({len(texto_ocr_raw)} -> {len(texto_ocr)} chars after cleaning)"
            )

            return (page_number, PaginaOCR(page=page_number, texto=texto_ocr))

        except PDFProcessingException as e:
            logger.error(
                f"[{page_index + 1}/{total_pages}][error:PDF][model:{model_name}] "
                f"Page {page_number}: {e}"
            )
            # Return page with error message
            return (
                page_number,
                PaginaOCR(page=page_number, texto=f"[Error PDF: {str(e)}]"),
            )
        except GeminiAPIException as e:
            logger.error(
                f"[{page_index + 1}/{total_pages}][error:Gemini][model:{model_name}] "
                f"Page {page_number}: {e}"
            )
            return (
                page_number,
                PaginaOCR(page=page_number, texto=f"[Error OCR: {str(e)}]"),
            )
        except Exception as e:
            logger.error(
                f"[{page_index + 1}/{total_pages}][error:Unknown][model:{model_name}] "
                f"Page {page_number}: {e}",
                exc_info=True,
            )
            return (
                page_number,
                PaginaOCR(page=page_number, texto=f"[Error: {str(e)}]"),
            )


    async def close(self):
        """Close all client connections and cleanup resources."""
        await self.gemini_client.close()
        self._executor.shutdown(wait=True)
        # Clear cache to free memory
        self._pdf_cache.clear()
        self._cache_locks.clear()

