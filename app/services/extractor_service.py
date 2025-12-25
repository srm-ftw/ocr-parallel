"""Extractor service that orchestrates the inmueble extraction process."""

import asyncio
import logging
import time
from typing import List, Optional, Tuple

from app.clients.gemini_client import GeminiClient
from app.config import DEFAULT_GEMINI_MODEL, settings
from app.core.exceptions import GeminiAPIException
from app.models.extractor import InmuebleRequest, InmuebleResponse
from app.utils.rate_limiter import ConcurrencyLimiter

logger = logging.getLogger(__name__)

# Prompt template for inmueble extraction
EXTRACTOR_PROMPT_TEMPLATE = """Eres un asistente experto en extracción de información de escrituras públicas en Colombia.
Actúas como un analista registral y notarial, con criterio jurídico estricto.

# CONTEXTO
Descripción del inmueble: {{$json.nombre_inmueble}}
Folio de matrícula: {{$json.folio}}

# TAREA
Localiza en el texto la sección completa que corresponde al inmueble descrito en el contexto
y copia TEXTUALMENTE toda su descripción.

Un inmueble no debe entenderse como un texto literal único, sino como un conjunto de
identificadores (por ejemplo: tipo de inmueble, número, torre, manzana, lote, bloque,
piso u otros identificadores registrales), los cuales pueden aparecer en el texto
de forma fragmentada, en frases separadas o en distinto orden.

Se considera que un inmueble ha sido encontrado cuando:
•  Los identificadores del inmueble aparecen explícitamente en el texto,
•  Dichos identificadores describen de forma coherente un único bien inmueble,
•  Aunque no todos los identificadores aparezcan juntos en una misma frase.

Una vez identificado el inmueble, copia TEXTUALMENTE toda su descripción completa.

# REGLAS CRÍTICAS
1.  Copia el texto EXACTAMENTE como aparece en la escritura.
2.  NO parafrasees, NO resumas, NO corrijas errores.
3.  Incluye TODO el contenido de la descripción del inmueble (linderos, medidas, áreas,
   coeficientes, ubicación, notas aclaratorias, etc.).
4.  Mantén el formato original del texto: mayúsculas, puntuación y saltos de línea.
5.  Si hay errores de OCR, cópialos tal cual.
6.  NO mezcles información de otros inmuebles ni de bienes distintos.
7.  NO completes, NO infieras ni inventes información que no esté explícitamente en el texto.
8.  Si no tienes una alta certeza de que el inmueble se encuentra en el texto, escribe
   exactamente: Inmueble no encontrado

# PISTAS DE BÚSQUEDA (NO LIMITANTES)
La descripción del inmueble suele incluir expresiones como:
•  "APARTAMENTO [NÚMERO]", "CASA", "LOTE", "PARQUEADERO", "DEPÓSITO", "MANZANA"
•  "TORRE", "BLOQUE", "PISO", "NIVEL"
•  "Folio de Matrícula"
•  "LINDEROS:", "Por el NORTE con:", "CENIT:", "NADIR:"
•  "ÁREA PRIVADA:", "ÁREA CONSTRUIDA:", "COEFICIENTE:"

La descripción normalmente termina cuando inicia la descripción de otro inmueble
o una nueva sección independiente de la escritura.

# TEXTO A ANALIZAR
{{$json.texto_ocr_completo}}

# FORMATO DE RESPUESTA
Responde SOLO con el texto copiado de la escritura.
NO incluyas explicaciones, encabezados, comentarios ni formato adicional.
"""


def build_extractor_prompt(nombre_inmueble: str, folio: str, texto_ocr_completo: str) -> str:
    """
    Build the extractor prompt by replacing template variables.

    Args:
        nombre_inmueble: Name of the inmueble
        folio: Folio number
        texto_ocr_completo: Complete OCR text

    Returns:
        str: Prompt with variables replaced
    """
    prompt = EXTRACTOR_PROMPT_TEMPLATE.replace("{{$json.nombre_inmueble}}", nombre_inmueble)
    prompt = prompt.replace("{{$json.folio}}", folio)
    prompt = prompt.replace("{{$json.texto_ocr_completo}}", texto_ocr_completo)
    return prompt


class ExtractorService:
    """Service for processing inmueble extraction requests with parallelization and order preservation."""

    def __init__(self):
        """Initialize Extractor service with clients."""
        self.gemini_client = GeminiClient()
        self.concurrency_limiter = ConcurrencyLimiter(settings.max_concurrent_requests)

    async def _process_single_inmueble(
        self, index: int, request: InmuebleRequest, gemini_client: GeminiClient, total: int, model_name: str
    ) -> Tuple[int, InmuebleResponse, dict]:
        """
        Process a single inmueble: build prompt and call Gemini.

        Args:
            index: Original index of the inmueble in the input array
            request: The inmueble extraction request
            gemini_client: Gemini client to use
            total: Total number of inmuebles in the batch

        Returns:
            Tuple of (index, InmuebleResponse, metrics_dict)
        """
        start_time = time.time()
        metrics = {
            "success": False,
            "not_found": False,
            "error": False,
            "processing_time": 0.0,
            "response_size": 0,
            "input_text_size": len(request.texto_ocr_completo),
        }

        async with self.concurrency_limiter:
            try:
                logger.debug(
                    f"Processing inmueble {index}: {request.nombre_inmueble} (folio: {request.folio})"
                )

                # Build prompt with variables replaced
                prompt = build_extractor_prompt(
                    request.nombre_inmueble, request.folio, request.texto_ocr_completo
                )
                prompt_size = len(prompt)

                # Call Gemini API for text extraction
                gemini_start = time.time()
                extracted_text = await gemini_client.generate_text_from_prompt(prompt)
                gemini_time = time.time() - gemini_start

                # Clean up response - remove markdown formatting if present, strip whitespace
                extracted_text = extracted_text.strip()
                metrics["response_size"] = len(extracted_text)

                # Check if response indicates not found
                if not extracted_text or extracted_text.lower() in [
                    "inmueble no encontrado",
                    "no encontrado",
                ]:
                    metrics["not_found"] = True
                    elapsed = time.time() - start_time
                    metrics["processing_time"] = elapsed
                    logger.warning(
                        f"[{index + 1}/{total}][not_found][model:{model_name}] "
                        f"Inmueble {request.nombre_inmueble} (folio: {request.folio}) "
                        f"processed in {elapsed:.2f}s (gemini: {gemini_time:.2f}s)"
                    )
                    return (index, InmuebleResponse(text="Inmueble no encontrado"), metrics)

                metrics["success"] = True
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                metrics["gemini_time"] = gemini_time
                metrics["prompt_size"] = prompt_size

                logger.info(
                    f"[{index + 1}/{total}][success][model:{model_name}] "
                    f"Inmueble {request.nombre_inmueble} (folio: {request.folio}) "
                    f"processed in {elapsed:.2f}s (gemini: {gemini_time:.2f}s, "
                    f"response: {len(extracted_text)} chars)"
                )

                return (index, InmuebleResponse(text=extracted_text), metrics)

            except GeminiAPIException as e:
                metrics["error"] = True
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"[{index + 1}/{total}][error][model:{model_name}] "
                    f"Inmueble {request.nombre_inmueble} (folio: {request.folio}) "
                    f"Gemini API error after {elapsed:.2f}s: {e}"
                )
                return (index, InmuebleResponse(text="Inmueble no encontrado"), metrics)
            except Exception as e:
                metrics["error"] = True
                elapsed = time.time() - start_time
                metrics["processing_time"] = elapsed
                logger.error(
                    f"[{index + 1}/{total}][error][model:{model_name}] "
                    f"Inmueble {request.nombre_inmueble} (folio: {request.folio}) "
                    f"Unexpected error after {elapsed:.2f}s: {e}",
                    exc_info=True,
                )
                return (index, InmuebleResponse(text="Inmueble no encontrado"), metrics)

    async def process_inmuebles_async(
        self, requests: List[InmuebleRequest], model: Optional[str] = None
    ) -> List[InmuebleResponse]:
        """
        Process multiple inmuebles in parallel with order preservation.

        Args:
            requests: List of inmueble extraction requests
            model: Optional Gemini model name. If not provided, uses default client.

        Returns:
            List of extraction responses in the same order as input
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
            total_input_size = sum(len(r.texto_ocr_completo) for r in requests)

            logger.info(
                f"Processing {len(requests)} inmueble extraction requests "
                f"(total input size: {total_input_size:,} chars, model: {model_to_use})"
            )

            # Create tasks for all inmuebles with their original indices
            total = len(requests)
            tasks = []
            for index, request in enumerate(requests):
                task = self._process_single_inmueble(index, request, gemini_client, total, model_to_use)
                tasks.append(task)

            # Process all inmuebles in parallel
            logger.info(f"Processing {len(tasks)} inmuebles in parallel")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build results array preserving order and collect metrics
            final_results: List[InmuebleResponse] = [None] * len(requests)  # type: ignore
            all_metrics = []
            success_count = 0
            not_found_count = 0
            error_count = 0
            total_response_size = 0
            not_found_list = []
            error_list = []

            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    logger.error(f"Task failed with exception: {result}", exc_info=True)
                    # This shouldn't happen since _process_single_inmueble catches all exceptions,
                    # but handle it as a safety measure
                    # We can't determine the index from the exception, so we'll fill None positions
                    continue

                # Type check: result should be a tuple of (index, response, metrics)
                try:
                    index, response, metrics = result  # type: ignore
                except (ValueError, TypeError) as e:
                    error_count += 1
                    logger.error(f"Invalid result format: {result}, error: {e}")
                    continue

                all_metrics.append(metrics)
                total_response_size += metrics.get("response_size", 0)

                if metrics.get("success"):
                    success_count += 1
                elif metrics.get("not_found"):
                    not_found_count += 1
                    not_found_list.append({
                        "index": index + 1,
                        "nombre": requests[index].nombre_inmueble,
                        "folio": requests[index].folio,
                    })
                elif metrics.get("error"):
                    error_count += 1
                    error_list.append({
                        "index": index + 1,
                        "nombre": requests[index].nombre_inmueble,
                        "folio": requests[index].folio,
                    })

                if index < len(final_results):
                    final_results[index] = response
                else:
                    logger.error(f"Invalid index {index} returned, max expected: {len(final_results) - 1}")

            # Fill any remaining None values (shouldn't happen, but safety check)
            for i in range(len(final_results)):
                if final_results[i] is None:
                    logger.warning(f"Missing result for index {i}, filling with default")
                    final_results[i] = InmuebleResponse(text="Inmueble no encontrado")
                    error_count += 1

            # Calculate aggregate metrics
            batch_elapsed = time.time() - batch_start_time
            processing_times = [m.get("processing_time", 0) for m in all_metrics if m]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            min_processing_time = min(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0

            gemini_times = [m.get("gemini_time", 0) for m in all_metrics if m.get("gemini_time")]
            avg_gemini_time = sum(gemini_times) / len(gemini_times) if gemini_times else 0

            # Log comprehensive metrics
            logger.info(
                f"Completed processing {len(requests)} inmueble requests in {batch_elapsed:.2f}s | "
                f"Model: {model_to_use} | "
                f"Success: {success_count} | Not Found: {not_found_count} | Errors: {error_count} | "
                f"Avg time/item: {avg_processing_time:.2f}s (min: {min_processing_time:.2f}s, "
                f"max: {max_processing_time:.2f}s) | Avg Gemini time: {avg_gemini_time:.2f}s | "
                f"Total response size: {total_response_size:,} chars"
            )

            # Log detailed lists of not found and errors
            if not_found_list:
                logger.warning(
                    f"NOT FOUND ({len(not_found_list)}): " +
                    ", ".join([f"#{item['index']} {item['nombre']} (folio: {item['folio']})" for item in not_found_list])
                )
            
            if error_list:
                logger.error(
                    f"ERRORS ({len(error_list)}): " +
                    ", ".join([f"#{item['index']} {item['nombre']} (folio: {item['folio']})" for item in error_list])
                )

            return final_results
        finally:
            # Close temporary client if we created one
            if use_temporary_client:
                await gemini_client.close()

    async def close(self):
        """Close all client connections."""
        await self.gemini_client.close()

