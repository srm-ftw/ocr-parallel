"""Custom exceptions for the application."""


class OCRException(Exception):
    """Base exception for OCR-related errors."""

    pass


class GCSException(OCRException):
    """Exception raised for GCS-related errors."""

    pass


class GeminiAPIException(OCRException):
    """Exception raised for Gemini API errors."""

    pass


class PDFProcessingException(OCRException):
    """Exception raised for PDF processing errors."""

    pass


class RateLimitException(GeminiAPIException):
    """Exception raised when rate limit is exceeded."""

    pass


class ValidationException(OCRException):
    """Exception raised for validation errors."""

    pass

