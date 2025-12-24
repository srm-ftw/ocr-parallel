"""PDF utilities for extracting specific pages from PDFs."""

import logging
from io import BytesIO

from pypdf import PdfReader, PdfWriter

from app.core.exceptions import PDFProcessingException

logger = logging.getLogger(__name__)


def extract_page_from_pdf(pdf_bytes: bytes, page_number: int) -> bytes:
    """
    Extract a specific page from a PDF and return it as a new PDF.

    Args:
        pdf_bytes: The PDF file as bytes
        page_number: The page number to extract (1-indexed)

    Returns:
        bytes: A new PDF containing only the specified page

    Raises:
        PDFProcessingException: If the extraction fails
    """
    try:
        logger.debug(f"Extracting page {page_number} from PDF ({len(pdf_bytes)} bytes)")

        # Validate page number
        if page_number < 1:
            raise PDFProcessingException(f"Page number must be >= 1, got {page_number}")

        # Read the PDF
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)

        # Validate page exists
        if page_number > total_pages:
            raise PDFProcessingException(
                f"Page {page_number} does not exist. PDF has {total_pages} pages."
            )

        # Convert to 0-indexed
        page_index = page_number - 1

        # Create a new PDF with only the requested page
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_index])

        # Write to bytes
        output_buffer = BytesIO()
        pdf_writer.write(output_buffer)
        output_bytes = output_buffer.getvalue()
        output_buffer.close()

        logger.debug(f"Successfully extracted page {page_number} ({len(output_bytes)} bytes)")
        return output_bytes

    except PDFProcessingException:
        raise
    except Exception as e:
        logger.error(f"Error extracting page {page_number}: {e}")
        raise PDFProcessingException(f"Failed to extract page {page_number}: {e}") from e

