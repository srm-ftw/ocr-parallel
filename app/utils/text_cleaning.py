"""Text cleaning utilities for OCR output from escrituras públicas."""

import re


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text by normalizing whitespace and line breaks.
    
    Simple cleaning that:
    - Removes literal \n strings
    - Converts newlines to spaces
    - Normalizes multiple spaces to single space
    - Trims whitespace
    
    Args:
        text: Raw OCR text from escritura
    
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return text

    # Remove literal \n strings (escaped newlines)
    cleaned = text.replace("\\n", " ")
    
    # Convert actual newlines to spaces
    cleaned = cleaned.replace("\n", " ")
    
    # Convert multiple spaces/tabs to single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    # Trim whitespace
    cleaned = cleaned.strip()
    
    return cleaned

