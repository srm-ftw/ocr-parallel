"""Deduplication utilities for OCR requests."""

import logging
from typing import Dict, List, Tuple

from app.models.ocr import OCRPageRequest, OCRPageResponse

logger = logging.getLogger(__name__)


def create_unique_key(request: OCRPageRequest) -> Tuple[str, str, str]:
    """
    Create a unique key for a page request.

    Args:
        request: The OCR page request

    Returns:
        Tuple of (bucket, file, page) as the unique key
    """
    return (request.bucket, request.file, request.page)


def deduplicate_requests(
    requests: List[OCRPageRequest],
) -> Tuple[List[OCRPageRequest], Dict[int, Tuple[str, str, str]]]:
    """
    Identify duplicate page requests and create a mapping.

    Args:
        requests: List of OCR page requests

    Returns:
        Tuple of:
        - unique_requests: List of unique requests (first occurrence of each)
        - index_mapping: Dict mapping original index -> unique key
    """
    seen_keys: Dict[Tuple[str, str, str], int] = {}
    unique_requests: List[OCRPageRequest] = []
    index_mapping: Dict[int, Tuple[str, str, str]] = {}

    for original_index, request in enumerate(requests):
        key = create_unique_key(request)

        if key not in seen_keys:
            # First time seeing this key - add to unique requests
            unique_index = len(unique_requests)
            seen_keys[key] = unique_index
            unique_requests.append(request)

        # Map original index to unique key
        index_mapping[original_index] = key

    duplicates_count = len(requests) - len(unique_requests)
    if duplicates_count > 0:
        logger.info(
            f"Deduplication: Processing {len(unique_requests)} unique pages from "
            f"{len(requests)} total requests ({duplicates_count} duplicates detected)"
        )

    return unique_requests, index_mapping


def map_results_back(
    unique_results: Dict[Tuple[str, str, str], OCRPageResponse],
    index_mapping: Dict[int, Tuple[str, str, str]],
    total_count: int,
) -> List[OCRPageResponse]:
    """
    Map unique results back to all original positions.

    Args:
        unique_results: Dict mapping unique key -> OCR result
        index_mapping: Dict mapping original index -> unique key
        total_count: Total number of original requests

    Returns:
        List of OCR responses in the same order as original input
    """
    results = []
    for original_index in range(total_count):
        key = index_mapping.get(original_index)
        if key and key in unique_results:
            results.append(unique_results[key])
        else:
            # This shouldn't happen if deduplication worked correctly
            logger.warning(f"No result found for original index {original_index}")
            # Create an error response
            results.append(
                OCRPageResponse(
                    page="", texto_ocr=""
                )  # Empty page will be filled by service with error info
            )

    return results

