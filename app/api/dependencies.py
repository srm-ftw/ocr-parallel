"""API dependencies for authentication and authorization."""

import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key")) -> str:
    """
    Verify the API key from the x-api-key header.
    
    Args:
        x_api_key: The API key from the x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # If no API key is configured, skip authentication
    if not settings.api_key:
        logger.warning("API key not configured in environment. Skipping authentication.")
        return "no-auth"
    
    # Check if header is present
    if not x_api_key:
        logger.warning("Missing x-api-key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Verify API key matches
    if x_api_key != settings.api_key:
        logger.warning(f"Invalid API key provided (first 4 chars: {x_api_key[:4] if len(x_api_key) >= 4 else 'N/A'})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    logger.debug("API key verified successfully")
    return x_api_key

