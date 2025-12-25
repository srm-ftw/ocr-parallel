"""Gemini API client with rate limiting and retry logic."""

import asyncio
import base64
import json
import logging
import random
from typing import Optional

import aiohttp

from app.config import settings, get_model_limits
from app.core.exceptions import GeminiAPIException, RateLimitException
from app.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Prompt para escrituras públicas según especificación
OCR_PROMPT = """
#ROL
Eres un asistente experto en transcribir imágenes de escrituras públicas colombianas a texto claro, continuo y legible, preservando la estructura jurídica sin introducir errores ni inventos.

#OBJETIVO DEL AGENTE
Extraer únicamente el texto jurídico principal ubicado dentro del documento, dejando por fuera cualquier elemento visual o marginal que no haga parte del contenido legal de la escritura.

#INSTRUCCIONES PRINCIPALES
1. Ignora por completo elementos no jurídicos
Excluye:
- Sellos ("RECIBIDO", "COPIA", timbres, logos).
- Texto en márgenes, bordes, encabezados externos, pies de página.
- Sombras, pliegues, manchas, perforaciones, líneas verticales u horizontales decorativas.
- Códigos de barras, numeración del escaneo, anotaciones manuales o superpuestas.

2. Manejo de texto parcial
- Si una palabra está parcialmente tapada, recupera lo visible y marca lo perdido como [ilegible].
- Si una línea está cortada, complétala únicamente con lo visible.

3. Preserva la estructura real del documento
- Respeta los saltos de línea auténticos.
- Elimina saltos generados por el OCR que no existan en el documento.
- No combines partes del texto que no están unidas visualmente.

5. No corrijas ni interpretes

#QUE NO DEBES HACER
1. No resumas ni reinterpretes el contenido.
2. No mezcles texto jurídico con elementos marginales.
3. No agregues explicaciones o comentarios.
4. No inventes texto que no sea legible.
5. No adaptes el contenido para hacerlo "más bonito".

#FORMATO DE SALIDA
1. Texto plano continuo.
2. Sin comentarios, intros, conclusiones ni notas.
3. Sin títulos inventados.
4. Solo el contenido jurídico transcrito del documento.
"""

# Base URL for Gemini API
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def get_gemini_api_url(model_name: str) -> str:
    """
    Get the API URL for a specific Gemini model.
    
    Args:
        model_name: Name of the Gemini model (e.g., 'gemini-2.0-flash')
        
    Returns:
        Full API URL for the model
    """
    return f"{GEMINI_API_BASE_URL}/{model_name}:generateContent"


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Gemini client with rate limiting.
        
        Args:
            model_name: Gemini model to use (defaults to settings.gemini_model)
        """
        self.api_keys = settings.gemini_api_keys_list
        self.current_key_index = 0
        self.model_name = model_name or settings.gemini_model
        
        # Get rate limits for the selected model
        model_limits = get_model_limits(self.model_name)
        rpm_limit = model_limits["rpm"]
        
        # Rate limiter per API key
        # We use the full RPM limit per key (with multiple keys, each gets full limit)
        self.rate_limiter = RateLimiter(
            max_rate=rpm_limit, 
            time_period=60
        )
        
        # Retry configuration with specific wait times: 10s, 30s, 60s
        self.max_retries = 3  # 3 retries + initial attempt = 4 total attempts
        self.retry_wait_times = [10, 30, 60]  # Wait times in seconds for each retry attempt
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_url = get_gemini_api_url(self.model_name)

        if not self.api_keys:
            raise GeminiAPIException("No Gemini API keys configured")

        logger.info(
            f"Gemini client initialized with model '{self.model_name}', "
            f"{len(self.api_keys)} API key(s), "
            f"rate limit: {rpm_limit} RPM per key, "
            f"{model_limits['tpm']:,} TPM, "
            f"optimal concurrency: {model_limits['optimal_concurrency']} per key"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _get_next_api_key(self) -> str:
        """Get next API key (round-robin for multiple projects)."""
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    async def _call_gemini_with_retry(
        self, pdf_base64: str, api_key: str, attempt: int = 0
    ) -> str:
        """
        Call Gemini API with retry logic and exponential backoff.

        Args:
            pdf_base64: PDF encoded in base64
            api_key: Gemini API key
            attempt: Current retry attempt number

        Returns:
            str: Extracted text from the PDF

        Raises:
            GeminiAPIException: If all retries fail
            RateLimitException: If rate limit is exceeded
        """
        url = f"{self.api_url}?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": OCR_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": pdf_base64,
                            }
                        },
                    ]
                }
            ]
        }

        session = await self._get_session()

        try:
            async with self.rate_limiter:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 429:
                        # Rate limit exceeded
                        if attempt < self.max_retries:
                            wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                            logger.warning(
                                f"Rate limit exceeded, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})"
                            )
                            await asyncio.sleep(wait_time)
                            return await self._call_gemini_with_retry(pdf_base64, api_key, attempt + 1)
                        else:
                            raise RateLimitException(
                                f"Rate limit exceeded after {self.max_retries + 1} total attempts"
                            )

                    if response.status != 200:
                        # Retry on non-200 status codes (except 429 which is handled above)
                        if attempt < self.max_retries:
                            wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                            error_text = await response.text()
                            logger.warning(
                                f"Gemini API error {response.status}: {error_text}, "
                                f"retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})"
                            )
                            await asyncio.sleep(wait_time)
                            return await self._call_gemini_with_retry(pdf_base64, api_key, attempt + 1)
                        else:
                            error_text = await response.text()
                            logger.error(f"Gemini API error {response.status}: {error_text}")
                            raise GeminiAPIException(
                                f"Gemini API returned status {response.status} after {self.max_retries + 1} total attempts: {error_text}"
                            )

                    result = await response.json()

                    # Extract text from response
                    if "candidates" not in result or not result["candidates"]:
                        raise GeminiAPIException("No candidates in Gemini response")

                    candidate = result["candidates"][0]
                    if "content" not in candidate or "parts" not in candidate["content"]:
                        raise GeminiAPIException("Invalid response structure from Gemini")

                    parts = candidate["content"]["parts"]
                    if not parts or "text" not in parts[0]:
                        raise GeminiAPIException("No text in Gemini response")

                    text = parts[0]["text"]
                    logger.debug(f"Successfully extracted text ({len(text)} characters)")
                    return text

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling Gemini API: {e}")
            if attempt < self.max_retries:
                wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                logger.info(f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})")
                await asyncio.sleep(wait_time)
                return await self._call_gemini_with_retry(pdf_base64, api_key, attempt + 1)
            raise GeminiAPIException(f"HTTP error after {self.max_retries + 1} total attempts: {e}") from e

    async def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF using Gemini API.

        Args:
            pdf_bytes: PDF file as bytes

        Returns:
            str: Extracted text

        Raises:
            GeminiAPIException: If extraction fails
        """
        try:
            # Encode PDF to base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

            # Get API key (round-robin if multiple keys)
            api_key = self._get_next_api_key()

            # Call API with retry logic
            text = await self._call_gemini_with_retry(pdf_base64, api_key)

            return text

        except RateLimitException as e:
            # If rate limit exhausted and not using Flash, try once with Flash as fallback
            if self.model_name != "gemini-2.0-flash":
                logger.warning(
                    f"Rate limit exhausted for {self.model_name}, attempting fallback to gemini-2.0-flash"
                )
                flash_client = None
                try:
                    # Create temporary Flash client for fallback attempt
                    flash_client = GeminiClient(model_name="gemini-2.0-flash")
                    # Use same API key for consistency
                    flash_api_key = api_key
                    text = await flash_client._call_gemini_with_retry(pdf_base64, flash_api_key, attempt=0)
                    await flash_client.close()
                    logger.info("Successfully used Flash fallback after rate limit exhaustion")
                    return text
                except Exception as fallback_error:
                    logger.error(f"Flash fallback also failed: {fallback_error}")
                    if flash_client:
                        await flash_client.close()
                    # Re-raise original rate limit exception
                    raise e
            else:
                # Already using Flash, no fallback available
                raise
        except GeminiAPIException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting text: {e}")
            raise GeminiAPIException(f"Failed to extract text: {e}") from e

    async def _call_gemini_text_with_retry(
        self, prompt: str, api_key: str, timeout: int, attempt: int = 0
    ) -> str:
        """
        Call Gemini API with text-only prompt (no PDF) with retry logic and exponential backoff.

        Args:
            prompt: Text prompt to send to Gemini
            api_key: Gemini API key
            timeout: Request timeout in seconds
            attempt: Current retry attempt number

        Returns:
            str: Generated text from Gemini

        Raises:
            GeminiAPIException: If all retries fail
            RateLimitException: If rate limit is exceeded
        """
        url = f"{self.api_url}?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ]
        }

        session = await self._get_session()

        try:
            async with self.rate_limiter:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 429:
                        # Rate limit exceeded
                        if attempt < self.max_retries:
                            wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                            logger.warning(
                                f"Rate limit exceeded, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})"
                            )
                            await asyncio.sleep(wait_time)
                            return await self._call_gemini_text_with_retry(
                                prompt, api_key, timeout, attempt + 1
                            )
                        else:
                            raise RateLimitException(
                                f"Rate limit exceeded after {self.max_retries + 1} total attempts"
                            )

                    if response.status != 200:
                        # Retry on non-200 status codes (except 429 which is handled above)
                        if attempt < self.max_retries:
                            wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                            error_text = await response.text()
                            logger.warning(
                                f"Gemini API error {response.status}: {error_text}, "
                                f"retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})"
                            )
                            await asyncio.sleep(wait_time)
                            return await self._call_gemini_text_with_retry(
                                prompt, api_key, timeout, attempt + 1
                            )
                        else:
                            error_text = await response.text()
                            logger.error(f"Gemini API error {response.status}: {error_text}")
                            raise GeminiAPIException(
                                f"Gemini API returned status {response.status} after {self.max_retries + 1} total attempts: {error_text}"
                            )

                    result = await response.json()

                    # Extract text from response
                    if "candidates" not in result or not result["candidates"]:
                        raise GeminiAPIException("No candidates in Gemini response")

                    candidate = result["candidates"][0]
                    if "content" not in candidate or "parts" not in candidate["content"]:
                        raise GeminiAPIException("Invalid response structure from Gemini")

                    parts = candidate["content"]["parts"]
                    if not parts or "text" not in parts[0]:
                        raise GeminiAPIException("No text in Gemini response")

                    text = parts[0]["text"]
                    logger.debug(f"Successfully generated text ({len(text)} characters)")
                    return text

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling Gemini API: {e}")
            if attempt < self.max_retries:
                wait_time = self.retry_wait_times[attempt] if attempt < len(self.retry_wait_times) else 60
                logger.info(f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries + 1})")
                await asyncio.sleep(wait_time)
                return await self._call_gemini_text_with_retry(prompt, api_key, timeout, attempt + 1)
            raise GeminiAPIException(f"HTTP error after {self.max_retries + 1} total attempts: {e}") from e

    async def generate_text_from_prompt(self, prompt: str, timeout: Optional[int] = None) -> str:
        """
        Generate text from a text-only prompt using Gemini API.

        Args:
            prompt: Text prompt to send to Gemini
            timeout: Request timeout in seconds (defaults to extractor_timeout_per_item from settings)

        Returns:
            str: Generated text

        Raises:
            GeminiAPIException: If generation fails
        """
        try:
            # Use configured timeout or default
            if timeout is None:
                timeout = settings.extractor_timeout_per_item

            # Get API key (round-robin if multiple keys)
            api_key = self._get_next_api_key()

            # Call API with retry logic
            text = await self._call_gemini_text_with_retry(prompt, api_key, timeout)

            return text

        except RateLimitException as e:
            # If rate limit exhausted and not using Flash, try once with Flash as fallback
            if self.model_name != "gemini-2.0-flash":
                logger.warning(
                    f"Rate limit exhausted for {self.model_name}, attempting fallback to gemini-2.0-flash"
                )
                flash_client = None
                try:
                    # Create temporary Flash client for fallback attempt
                    flash_client = GeminiClient(model_name="gemini-2.0-flash")
                    # Use same API key for consistency
                    flash_api_key = api_key
                    # Ensure timeout is set
                    fallback_timeout = timeout or settings.extractor_timeout_per_item
                    # Use same timeout, attempt 0 for fresh start
                    text = await flash_client._call_gemini_text_with_retry(prompt, flash_api_key, fallback_timeout, attempt=0)
                    await flash_client.close()
                    logger.info("Successfully used Flash fallback after rate limit exhaustion")
                    return text
                except Exception as fallback_error:
                    logger.error(f"Flash fallback also failed: {fallback_error}")
                    if flash_client:
                        await flash_client.close()
                    # Re-raise original rate limit exception
                    raise e
            else:
                # Already using Flash, no fallback available
                raise
        except GeminiAPIException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating text: {e}")
            raise GeminiAPIException(f"Failed to generate text: {e}") from e

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

