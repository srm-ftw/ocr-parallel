"""Gemini API client with rate limiting and retry logic."""

import asyncio
import base64
import json
import logging
import random
from typing import Optional

import aiohttp

from app.config import settings
from app.core.exceptions import GeminiAPIException, RateLimitException
from app.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Prompt para escrituras públicas según especificación
OCR_PROMPT = """#ROL
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
4. Solo el contenido jurídico transcrito del documento."""

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self):
        """Initialize Gemini client with rate limiting."""
        self.api_keys = settings.gemini_api_keys_list
        self.current_key_index = 0
        self.rate_limiter = RateLimiter(
            max_rate=settings.gemini_rate_limit_per_minute, time_period=60
        )
        self.max_retries = settings.gemini_max_retries
        self.backoff_base = settings.gemini_retry_backoff_base
        self.session: Optional[aiohttp.ClientSession] = None

        if not self.api_keys:
            raise GeminiAPIException("No Gemini API keys configured")

        logger.info(
            f"Gemini client initialized with {len(self.api_keys)} API key(s), "
            f"rate limit: {settings.gemini_rate_limit_per_minute} req/min"
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
        url = f"{GEMINI_API_URL}?key={api_key}"

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
                            wait_time = (self.backoff_base ** attempt) + random.uniform(0, 1)
                            logger.warning(
                                f"Rate limit exceeded, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            return await self._call_gemini_with_retry(pdf_base64, api_key, attempt + 1)
                        else:
                            raise RateLimitException(
                                f"Rate limit exceeded after {self.max_retries} retries"
                            )

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API error {response.status}: {error_text}")
                        raise GeminiAPIException(
                            f"Gemini API returned status {response.status}: {error_text}"
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
                wait_time = (self.backoff_base ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._call_gemini_with_retry(pdf_base64, api_key, attempt + 1)
            raise GeminiAPIException(f"HTTP error after {self.max_retries} retries: {e}") from e

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

        except (RateLimitException, GeminiAPIException):
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting text: {e}")
            raise GeminiAPIException(f"Failed to extract text: {e}") from e

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

