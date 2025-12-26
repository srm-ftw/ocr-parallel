"""Full extractor service with hybrid text localization + LLM extraction."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from app.clients.gemini_client import GeminiClient
from app.config import DEFAULT_GEMINI_MODEL, settings
from app.core.exceptions import GeminiAPIException
from app.models.extractor import InmuebleRequest, InmuebleResponse
from app.models.full_process import EscrituraOCRResult, InmuebleInput, InmuebleResult
from app.services.extractor_service import ExtractorService
from app.utils.chunking import concatenate_pages, get_pages_from_chunks
from app.utils.text_localization import localize_inmuebles_in_chunks, refine_pages_with_boundaries

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Statistics for extraction process."""

    phase1_text_localization_time_seconds: float
    phase2_llm_refinement_time_seconds: float
    phase3_preparation_time_seconds: float
    phase3_extraction_time_seconds: float
    found_by_text_regex: int
    found_by_refinement_llm: int
    total_found: int
    total_not_found: int


# Prompt for LLM-based localization refinement
LOCALIZADOR_PROMPT_TEMPLATE = """Eres un asistente experto en análisis de escrituras públicas en Colombia.

# TAREA
Analiza el siguiente fragmento de una escritura pública y determina cuáles de los inmuebles 
listados se mencionan en el texto.

# LISTA DE INMUEBLES A BUSCAR
{lista_inmuebles}

# TEXTO A ANALIZAR  
{texto_chunk}

# REGLAS
- Un inmueble se considera "encontrado" si sus identificadores principales aparecen en el texto
- Los identificadores pueden aparecer con variaciones (ej: "T3" = "TORRE 3", "MZ" = "MANZANA")
- Si hay ambigüedad, indica todos los posibles matches
- Un inmueble puede aparecer con variaciones en el nombre (ej: "APARTAMENTO 102 -T3 MZ 12" vs "APARTAMENTO 102 TORRE 3 MANZANA 12")

# FORMATO DE RESPUESTA (JSON)
Responde SOLO con un JSON array. Cada elemento debe tener:
{{
  "inmueble": "nombre exacto del inmueble de la lista",
  "encontrado": true/false,
  "confianza": "alta" | "media" | "baja"
}}

Ejemplo:
[
  {{"inmueble": "APARTAMENTO 102 -T3", "encontrado": true, "confianza": "alta"}},
  {{"inmueble": "APARTAMENTO 103 -T3", "encontrado": false, "confianza": "n/a"}}
]
"""


class FullExtractorService:
    """Service for extracting inmuebles from full escritura OCR results."""

    def __init__(self):
        """Initialize Full Extractor service."""
        self.extractor_service = ExtractorService()
        self.gemini_client = GeminiClient()

    def _build_localizador_prompt(self, inmuebles: List[InmuebleInput], chunk_text: str) -> str:
        """
        Build prompt for LLM-based localization.

        Args:
            inmuebles: List of inmuebles to search for
            chunk_text: Text chunk to analyze

        Returns:
            Formatted prompt string
        """
        # Format inmuebles list
        inmuebles_list = []
        for inmueble in inmuebles:
            inmuebles_list.append(f"- {inmueble.inmueble} (folio: {inmueble.folio})")

        lista_inmuebles = "\n".join(inmuebles_list)

        prompt = LOCALIZADOR_PROMPT_TEMPLATE.format(
            lista_inmuebles=lista_inmuebles, texto_chunk=chunk_text
        )
        return prompt

    async def _refine_with_llm(
        self,
        chunks: List,
        inmuebles: List[InmuebleInput],
        candidatos: Dict[int, Set[int]],
        unmatched_indices: List[int],
    ) -> Dict[int, Set[int]]:
        """
        Use LLM to refine localization for inmuebles that weren't found textually.

        Args:
            chunks: List of ChunkInfo objects
            inmuebles: List of InmuebleInput objects
            candidatos: Current candidate chunks per inmueble
            unmatched_indices: List of inmueble indices that need LLM refinement

        Returns:
            Updated candidatos dictionary
        """
        if not unmatched_indices:
            return candidatos

        logger.info(
            f"Refining localization with LLM for {len(unmatched_indices)} inmuebles "
            f"across {len(chunks)} chunks"
        )

        # Process chunks in parallel
        tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            # Get unmatched inmuebles for this chunk
            unmatched_in_chunk = [
                inmuebles[i] for i in unmatched_indices if i < len(inmuebles)
            ]

            if not unmatched_in_chunk:
                continue

            task = self._process_chunk_with_llm(chunk, unmatched_in_chunk, unmatched_indices)
            tasks.append((chunk_idx, task))

        # Wait for all chunks to be processed
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Update candidatos based on LLM results
        for (chunk_idx, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"LLM localization failed for chunk {chunk_idx}: {result}")
                continue

            # result is a list of inmueble indices found in this chunk
            if isinstance(result, list):
                for inmueble_idx in result:
                    if inmueble_idx < len(candidatos):
                        candidatos[inmueble_idx].add(chunk_idx)

        # Log results
        newly_found = sum(1 for i in unmatched_indices if candidatos[i])
        chunks_processed = len(tasks)
        logger.info(
            f"  🔍 LLM procesó {chunks_processed} chunks | "
            f"Encontró {newly_found}/{len(unmatched_indices)} inmuebles adicionales"
        )

        return candidatos

    async def _process_chunk_with_llm(
        self, chunk, inmuebles: List[InmuebleInput], unmatched_indices: List[int]
    ) -> List[int]:
        """
        Process a single chunk with LLM to find unmatched inmuebles.

        Args:
            chunk: ChunkInfo object
            inmuebles: List of unmatched inmuebles to search for
            unmatched_indices: Original indices of these inmuebles

        Returns:
            List of inmueble indices (from unmatched_indices) found in this chunk
        """
        try:
            prompt = self._build_localizador_prompt(inmuebles, chunk.texto)

            # Call Gemini API
            response_text = await self.gemini_client.generate_text_from_prompt(prompt)

            # Parse JSON response
            # Try to extract JSON from response (might have markdown formatting)
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            results = json.loads(json_text)

            # Map results back to indices
            found_indices = []
            for result in results:
                if result.get("encontrado", False):
                    inmueble_name = result.get("inmueble", "")
                    # Find index in unmatched_inmuebles list
                    for idx, inmueble in enumerate(inmuebles):
                        if inmueble.inmueble == inmueble_name:
                            # Map back to original index
                            if idx < len(unmatched_indices):
                                found_indices.append(unmatched_indices[idx])
                            break

            return found_indices

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response for chunk {chunk.index}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.index} with LLM: {e}", exc_info=True)
            return []

    def _build_extraction_requests(
        self,
        ocr_result: EscrituraOCRResult,
        inmuebles: List[InmuebleInput],
        candidatos: Dict[int, Set[int]],
    ) -> List[InmuebleRequest]:
        """
        Build extraction requests with refined candidate pages for each inmueble.

        Args:
            ocr_result: Full OCR result with pages and chunks
            inmuebles: List of inmuebles to extract
            candidatos: Candidate chunks per inmueble

        Returns:
            List of InmuebleRequest objects
        """
        requests = []

        for inmueble_idx, inmueble in enumerate(inmuebles):
            chunk_indices = list(candidatos.get(inmueble_idx, set()))

            if not chunk_indices:
                # No candidates found, use all pages (fallback)
                logger.warning(
                    f"Inmueble {inmueble_idx + 1} '{inmueble.inmueble}': "
                    f"no candidate chunks, using all pages"
                )
                page_numbers = [p.page for p in ocr_result.paginas]
                boundaries = None
            else:
                # Get candidate pages from chunks
                candidate_pages = get_pages_from_chunks(chunk_indices, ocr_result.chunks)
                
                # Refine pages to only include where inmueble actually appears
                refined_pages, start_pos, end_pos = refine_pages_with_boundaries(
                    inmueble, candidate_pages, ocr_result.paginas, threshold=0.6
                )
                
                page_numbers = refined_pages
                
                # Build boundaries dict if detected
                boundaries = None
                if start_pos is not None and end_pos is not None:
                    boundaries = {
                        'start_char': start_pos,
                        'end_char': end_pos
                    }
                
                # Log refinement results
                reduction = ((len(candidate_pages) - len(page_numbers)) / len(candidate_pages) * 100) if candidate_pages else 0
                logger.info(
                    f"Inmueble {inmueble_idx + 1} '{inmueble.inmueble}': "
                    f"found in {len(chunk_indices)} chunk(s), "
                    f"refined {len(candidate_pages)} → {len(page_numbers)} pages "
                    f"({reduction:.1f}% reduction)"
                    + (f" [boundaries: {start_pos}-{end_pos}]" if boundaries else "")
                )

            # Concatenate text from refined pages (with boundaries if detected)
            texto_ocr_completo = concatenate_pages(
                ocr_result.paginas, page_numbers, boundaries=boundaries
            )

            # Format pages as comma-separated string
            paginas_str = ",".join(str(p) for p in sorted(page_numbers))

            request = InmuebleRequest(
                nombre_inmueble=inmueble.inmueble,
                folio=inmueble.folio,
                texto_ocr_completo=texto_ocr_completo,
                paginas_original=paginas_str,
            )

            requests.append(request)

        return requests

    def _build_final_results(
        self,
        inmuebles: List[InmuebleInput],
        candidatos: Dict[int, Set[int]],
        extraction_results: List[InmuebleResponse],
        ocr_result: EscrituraOCRResult,
        found_by_text: Set[int],
    ) -> List[InmuebleResult]:
        """
        Combine extraction results with page information.

        Args:
            inmuebles: Original inmueble inputs
            candidatos: Candidate chunks per inmueble
            extraction_results: Results from ExtractorService
            ocr_result: Full OCR result
            found_by_text: Set of inmueble indices found by text localization

        Returns:
            List of InmuebleResult objects
        """
        results = []

        for inmueble_idx, (inmueble, extraction_response) in enumerate(
            zip(inmuebles, extraction_results)
        ):
            # Determine status
            text = extraction_response.text.strip()
            if not text or text.lower() in ["inmueble no encontrado", "no encontrado"]:
                status = "not_found"
                paginas = None
                localization_method = None
            else:
                status = "found"
                # Determine localization method
                if inmueble_idx in found_by_text:
                    localization_method = "text-regex"
                else:
                    # Found by LLM (or not found at all, but status is found so must be LLM)
                    localization_method = "refinement-llm"
                
                # Get refined pages (same logic as in _build_extraction_requests)
                chunk_indices = list(candidatos.get(inmueble_idx, set()))
                if chunk_indices:
                    candidate_pages = get_pages_from_chunks(chunk_indices, ocr_result.chunks)
                    # Refine pages to only include where inmueble actually appears
                    refined_pages, _, _ = refine_pages_with_boundaries(
                        inmueble, candidate_pages, ocr_result.paginas, threshold=0.6
                    )
                    page_numbers = refined_pages
                    paginas = ",".join(str(p) for p in sorted(page_numbers)) if page_numbers else None
                else:
                    paginas = None

            result = InmuebleResult(
                inmueble=inmueble.inmueble,
                folio=inmueble.folio,
                status=status,
                paginas=paginas,
                text=text,
                localization_method=localization_method,
            )

            results.append(result)

        return results

    async def extract_inmuebles(
        self,
        ocr_result: EscrituraOCRResult,
        inmuebles: List[InmuebleInput],
        model: Optional[str] = None,
    ) -> Tuple[List[InmuebleResult], ExtractionStats]:
        """
        Extract inmuebles from escritura using hybrid approach.

        Phase 1: Text localization (fast, no LLM)
        Phase 2: LLM refinement for unmatched inmuebles
        Phase 3: Extraction using ExtractorService

        Args:
            ocr_result: Full OCR result with pages and chunks
            inmuebles: List of inmuebles to extract
            model: Optional Gemini model name

        Returns:
            Tuple of (List of InmuebleResult objects, ExtractionStats)
        """
        start_time = time.time()
        model_to_use = model or DEFAULT_GEMINI_MODEL

        logger.info(
            f"Starting extraction for {len(inmuebles)} inmuebles from "
            f"{ocr_result.escritura_file_name} (model: {model_to_use})"
        )

        # Phase 1: Text localization
        phase1_start = time.time()
        candidatos = localize_inmuebles_in_chunks(ocr_result.chunks, inmuebles)
        phase1_time = time.time() - phase1_start

        unmatched_indices = [i for i, c in candidatos.items() if not c]
        found_by_text_set = {i for i, c in candidatos.items() if c}  # Track which ones were found by text
        found_by_text = len(found_by_text_set)
        text_success_rate = (found_by_text / len(inmuebles) * 100) if inmuebles else 0
        
        # Calculate average chunks per inmueble
        avg_chunks = sum(len(c) for c in candidatos.values()) / len(inmuebles) if inmuebles else 0
        
        logger.info(
            f"Phase 1 (text localization) complete in {phase1_time:.2f}s: "
            f"{found_by_text}/{len(inmuebles)} inmuebles found ({text_success_rate:.1f}% success rate)"
        )
        logger.info(
            f"  📊 Localizador efectividad: {text_success_rate:.1f}% | "
            f"Promedio {avg_chunks:.1f} chunks por inmueble | "
            f"Tiempo: {phase1_time:.2f}s ({phase1_time/len(inmuebles)*1000:.1f}ms por inmueble)"
        )
        
        if unmatched_indices:
            unmatched_names = [inmuebles[i].inmueble for i in unmatched_indices]
            logger.info(
                f"  ⚠️  Inmuebles no encontrados por localizador textual ({len(unmatched_indices)}): "
                f"{', '.join(unmatched_names[:3])}"
                f"{'...' if len(unmatched_names) > 3 else ''}"
            )

        # Phase 2: LLM refinement for unmatched inmuebles
        phase2_time = 0.0
        llm_found_count = 0
        if unmatched_indices:
            logger.info(
                f"Phase 2 (LLM refinement) starting: "
                f"🔍 Procesando {len(unmatched_indices)} inmuebles con LLM "
                f"en {len(ocr_result.chunks)} chunks"
            )
            phase2_start = time.time()
            candidatos_before_llm = {k: set(v) for k, v in candidatos.items()}
            candidatos = await self._refine_with_llm(
                ocr_result.chunks, inmuebles, candidatos, unmatched_indices
            )
            phase2_time = time.time() - phase2_start
            
            # Calculate how many were found by LLM
            llm_found_count = sum(
                1 for i in unmatched_indices 
                if len(candidatos[i]) > len(candidatos_before_llm.get(i, set()))
            )
            llm_success_rate = (llm_found_count / len(unmatched_indices) * 100) if unmatched_indices else 0
            
            logger.info(
                f"Phase 2 (LLM refinement) complete in {phase2_time:.2f}s: "
                f"✅ LLM encontró {llm_found_count}/{len(unmatched_indices)} inmuebles adicionales "
                f"({llm_success_rate:.1f}% éxito)"
            )
            logger.info(
                f"  🤖 LLM efectividad: {llm_success_rate:.1f}% | "
                f"Tiempo: {phase2_time:.2f}s ({phase2_time/len(unmatched_indices)*1000:.1f}ms por inmueble)"
            )
        else:
            logger.info("Phase 2 (LLM refinement): ⏭️  SKIPPED - todos los inmuebles encontrados por localizador textual")

        # Phase 3: Build extraction requests and call ExtractorService
        phase3_start = time.time()
        extraction_requests = self._build_extraction_requests(ocr_result, inmuebles, candidatos)
        phase3_time = time.time() - phase3_start

        logger.info(
            f"Phase 3 (extraction) starting: {len(extraction_requests)} requests "
            f"(preparation: {phase3_time:.2f}s)"
        )

        extraction_start = time.time()
        extraction_results = await self.extractor_service.process_inmuebles_async(
            extraction_requests, model=model_to_use
        )
        extraction_time = time.time() - extraction_start

        # Build final results
        final_results = self._build_final_results(
            inmuebles, candidatos, extraction_results, ocr_result, found_by_text_set
        )

        total_time = time.time() - start_time
        found_count = sum(1 for r in final_results if r.status == "found")
        final_success_rate = (found_count / len(inmuebles) * 100) if inmuebles else 0
        
        # Calculate page refinement statistics from final results
        total_candidate_pages = 0
        total_refined_pages = 0
        for inmueble_idx, inmueble in enumerate(inmuebles):
            chunk_indices = list(candidatos.get(inmueble_idx, set()))
            if chunk_indices:
                candidate_pages = get_pages_from_chunks(chunk_indices, ocr_result.chunks)
                total_candidate_pages += len(candidate_pages)
                # Get refined pages from result
                result = final_results[inmueble_idx]
                if result.paginas:
                    refined_count = len(result.paginas.split(","))
                    total_refined_pages += refined_count
        
        avg_candidate_pages = total_candidate_pages / found_count if found_count > 0 else 0
        avg_refined_pages = total_refined_pages / found_count if found_count > 0 else 0
        page_reduction = ((total_candidate_pages - total_refined_pages) / total_candidate_pages * 100) if total_candidate_pages > 0 else 0
        
        # Count by localization method
        found_by_text_count = sum(1 for r in final_results if r.localization_method == "text-regex")
        found_by_llm_count = sum(1 for r in final_results if r.localization_method == "refinement-llm")
        
        logger.info("=" * 80)
        logger.info(
            f"✅ Extraction complete in {total_time:.2f}s: "
            f"{found_count}/{len(inmuebles)} inmuebles found ({final_success_rate:.1f}% éxito final)"
        )
        logger.info(
            f"  🔍 Método de localización: "
            f"📝 Texto={found_by_text_count} | 🤖 LLM={found_by_llm_count}"
        )
        logger.info(
            f"  ⏱️  Tiempos: Localización={phase1_time:.2f}s | "
            f"LLM={'✅ ' + str(phase2_time) + 's' if phase2_time > 0 else '⏭️  SKIPPED'} | "
            f"Extracción={extraction_time:.2f}s"
        )
        logger.info(
            f"  📄 Refinamiento de páginas: "
            f"{avg_candidate_pages:.1f} → {avg_refined_pages:.1f} páginas promedio "
            f"({page_reduction:.1f}% reducción)"
        )
        logger.info("=" * 80)
        
        # Log individual results with localization method
        for result in final_results:
            if result.status == "found":
                method_emoji = "📝" if result.localization_method == "text-regex" else "🤖"
                method_name = "texto" if result.localization_method == "text-regex" else "LLM"
                logger.info(
                    f"  {method_emoji} '{result.inmueble}' encontrado por {method_name} "
                    f"(páginas: {result.paginas})"
                )

        # Build extraction stats
        not_found_count = len(inmuebles) - found_count
        stats = ExtractionStats(
            phase1_text_localization_time_seconds=phase1_time,
            phase2_llm_refinement_time_seconds=phase2_time,
            phase3_preparation_time_seconds=phase3_time,
            phase3_extraction_time_seconds=extraction_time,
            found_by_text_regex=found_by_text_count,
            found_by_refinement_llm=found_by_llm_count,
            total_found=found_count,
            total_not_found=not_found_count,
        )

        return final_results, stats

    async def close(self):
        """Close all client connections."""
        await self.gemini_client.close()
        await self.extractor_service.close()

