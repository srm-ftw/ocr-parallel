"""Text localization utilities for finding inmuebles in escritura chunks."""

import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from app.models.full_process import InmuebleInput, PaginaOCR

logger = logging.getLogger(__name__)

# Dictionary of synonyms and variations for common inmueble terms
# Based on real-world patterns from escrituras públicas colombianas
SYNONYMS: Dict[str, List[str]] = {
    # Tipos de inmueble - Apartamentos
    "APTO": ["APARTAMENTO", "APT", "APTO.", "APTO", "APARTAMENTO", "APTO"],
    "APARTAMENTO": ["APTO", "APT", "APTO.", "APARTAMENTO"],
    "APT": ["APARTAMENTO", "APTO", "APTO."],
    
    # Parqueaderos
    "PQRO": ["PARQUEADERO", "PARQUEO", "GARAJE", "ESTACIONAMIENTO"],
    "PARQUEADERO": ["PQRO", "PARQUEO", "GARAJE", "ESTACIONAMIENTO"],
    "PARQUEO": ["PARQUEADERO", "PQRO", "GARAJE"],
    "GARAJE": ["PARQUEADERO", "PARQUEO", "PQRO"],
    
    # Depósitos y bodegas
    "DEP": ["DEPOSITO", "DEPÓSITO", "BODEGA"],
    "DEPOSITO": ["DEP", "DEPÓSITO", "BODEGA"],
    "DEPÓSITO": ["DEP", "DEPOSITO", "BODEGA"],
    "BODEGA": ["DEPOSITO", "DEPÓSITO", "DEP"],
    
    # Viviendas
    "CASA": ["VIVIENDA", "CASA", "VIVIENDA UNIFAMILIAR"],
    "VIVIENDA": ["CASA", "VIVIENDA"],
    
    # Terrenos
    "LOTE": ["LOTE", "TERRENO", "PREDIO", "LOTE DE TERRENO"],
    "TERRENO": ["LOTE", "PREDIO"],
    "PREDIO": ["LOTE", "TERRENO"],
    
    # Locales y oficinas
    "LOCAL": ["LOCAL", "LOCAL COMERCIAL", "ESTABLECIMIENTO"],
    "OFICINA": ["OFICINA", "OFICINA COMERCIAL"],
    
    # Torres - números del 1 al 20
    "T1": ["TORRE 1", "TORRE1", "T-1", "TORRE UNO", "TORRE I", "T1", "T 1"],
    "T2": ["TORRE 2", "TORRE2", "T-2", "TORRE DOS", "TORRE II", "T2", "T 2"],
    "T3": ["TORRE 3", "TORRE3", "T-3", "TORRE TRES", "TORRE III", "T3", "T 3"],
    "T4": ["TORRE 4", "TORRE4", "T-4", "TORRE CUATRO", "TORRE IV", "T4", "T 4"],
    "T5": ["TORRE 5", "TORRE5", "T-5", "TORRE CINCO", "TORRE V", "T5", "T 5"],
    "T6": ["TORRE 6", "TORRE6", "T-6", "TORRE SEIS", "TORRE VI", "T6", "T 6"],
    "T7": ["TORRE 7", "TORRE7", "T-7", "TORRE SIETE", "TORRE VII", "T7", "T 7"],
    "T8": ["TORRE 8", "TORRE8", "T-8", "TORRE OCHO", "TORRE VIII", "T8", "T 8"],
    "T9": ["TORRE 9", "TORRE9", "T-9", "TORRE NUEVE", "TORRE IX", "T9", "T 9"],
    "T10": ["TORRE 10", "TORRE10", "T-10", "TORRE DIEZ", "TORRE X", "T10", "T 10"],
    "TORRE": ["T", "TORRE"],
    
    # Manzanas
    "MZ": ["MANZANA", "MZA", "MZ.", "MZ", "MANZ"],
    "MANZANA": ["MZ", "MZA", "MZ.", "MANZ"],
    "MZA": ["MANZANA", "MZ", "MZ."],
    
    # Bloques
    "BQ": ["BLOQUE", "BLOCK", "BLQ", "BQ.", "BQ"],
    "BLOQUE": ["BQ", "BLOCK", "BLQ"],
    "BLQ": ["BLOQUE", "BQ", "BLOCK"],
    
    # Pisos - números escritos y numéricos
    "P1": ["PISO 1", "PISO1", "NIVEL 1", "NIVEL1", "PRIMER PISO", "PISO UNO", "PISO PRIMERO"],
    "P2": ["PISO 2", "PISO2", "NIVEL 2", "NIVEL2", "SEGUNDO PISO", "PISO DOS"],
    "P3": ["PISO 3", "PISO3", "NIVEL 3", "NIVEL3", "TERCER PISO", "PISO TRES"],
    "P4": ["PISO 4", "PISO4", "NIVEL 4", "NIVEL4", "CUARTO PISO", "PISO CUATRO"],
    "P5": ["PISO 5", "PISO5", "NIVEL 5", "NIVEL5", "QUINTO PISO", "PISO CINCO"],
    "P6": ["PISO 6", "PISO6", "NIVEL 6", "NIVEL6", "SEXTO PISO", "PISO SEIS"],
    "P7": ["PISO 7", "PISO7", "NIVEL 7", "NIVEL7", "SEPTIMO PISO", "PISO SIETE"],
    "P8": ["PISO 8", "PISO8", "NIVEL 8", "NIVEL8", "OCTAVO PISO", "PISO OCHO"],
    "P9": ["PISO 9", "PISO9", "NIVEL 9", "NIVEL9", "NOVENO PISO", "PISO NUEVE"],
    "P10": ["PISO 10", "PISO10", "NIVEL 10", "NIVEL10", "DECIMO PISO", "PISO DIEZ"],
    "PISO": ["NIVEL", "PISO"],
    "NIVEL": ["PISO", "NIVEL"],
    
    # Etapas
    "ETAPA": ["ETAPA", "ETAP", "ETP"],
    "ETAPA I": ["ETAPA 1", "ETAPA UNO", "ETAPA PRIMERA", "ETAPA I"],
    "ETAPA II": ["ETAPA 2", "ETAPA DOS", "ETAPA SEGUNDA", "ETAPA II"],
    "ETAPA III": ["ETAPA 3", "ETAPA TRES", "ETAPA TERCERA", "ETAPA III"],
    
    # Números escritos comunes
    "UNO": ["1", "UNO", "PRIMERO", "PRIMER"],
    "DOS": ["2", "DOS", "SEGUNDO"],
    "TRES": ["3", "TRES", "TERCERO", "TERCER"],
    "CUATRO": ["4", "CUATRO"],
    "CINCO": ["5", "CINCO", "QUINTO"],
    "SEIS": ["6", "SEIS", "SEXTO"],
    "SIETE": ["7", "SIETE", "SEPTIMO"],
    "OCHO": ["8", "OCHO", "OCTAVO"],
    "NUEVE": ["9", "NUEVE", "NOVENO"],
    "DIEZ": ["10", "DIEZ", "DECIMO"],
    
    # Números grandes escritos (comunes en escrituras)
    "CIEN": ["100", "CIEN", "CIENTO"],
    "DOSCIENTOS": ["200", "DOSCIENTOS", "DOSCIENTAS"],
    "TRESCIENTOS": ["300", "TRESCIENTOS", "TRESCIENTAS"],
    "CUATROCIENTOS": ["400", "CUATROCIENTOS", "CUATROCIENTAS"],
    "QUINIENTOS": ["500", "QUINIENTOS", "QUINIENTAS"],
    "SEISCIENTOS": ["600", "SEISCIENTOS", "SEISCIENTAS"],
    "OCHOCIENTOS": ["800", "OCHOCIENTOS", "OCHOCIENTAS"],
    
    # Cesión
    "CESION": ["CESIÓN", "CESION"],
    "CESIÓN": ["CESION", "CESIÓN"],
}


def tokenize_inmueble_name(nombre: str) -> Set[str]:
    """
    Extract searchable tokens from an inmueble name.
    
    Args:
        nombre: Inmueble name (e.g., "APARTAMENTO 102 -T3 MZ 12")
    
    Returns:
        Set of normalized tokens for searching
    """
    # Normalize: uppercase, remove extra spaces, handle dashes
    normalized = re.sub(r"\s+", " ", nombre.upper().strip())
    normalized = re.sub(r"\s*-\s*", " ", normalized)  # Remove dashes
    
    # Split by spaces and common separators
    tokens = re.split(r"[\s\-_,]+", normalized)
    
    # Filter out empty strings and very short tokens
    tokens = {t for t in tokens if len(t) >= 2}
    
    # Add number tokens separately (e.g., "102" from "APARTAMENTO 102")
    numbers = re.findall(r"\d+", nombre)
    tokens.update(numbers)
    
    return tokens


def generate_search_variants(tokens: Set[str]) -> Set[str]:
    """
    Generate all possible text variants for matching using synonyms.
    
    Args:
        tokens: Set of base tokens
    
    Returns:
        Expanded set of tokens including synonyms
    """
    variants = set(tokens)
    
    for token in tokens:
        # Add synonyms if available
        if token in SYNONYMS:
            variants.update(SYNONYMS[token])
        
        # Also check if token is a synonym key (reverse lookup)
        for key, synonyms in SYNONYMS.items():
            if token in synonyms:
                variants.add(key)
                variants.update(synonyms)
    
    return variants


def match_tokens_in_text(tokens: Set[str], text: str, threshold: float = 0.6) -> bool:
    """
    Check if tokens appear in text with a minimum threshold.
    
    Args:
        tokens: Set of tokens to search for
        text: Text to search in (normalized to uppercase)
        threshold: Minimum fraction of tokens that must match (default: 0.6)
    
    Returns:
        True if threshold is met, False otherwise
    """
    if not tokens:
        return False
    
    text_upper = text.upper()
    matches = 0
    
    for token in tokens:
        # Check if token appears in text (word boundary aware for numbers)
        if token.isdigit():
            # For numbers, use word boundaries
            pattern = r"\b" + re.escape(token) + r"\b"
            if re.search(pattern, text_upper):
                matches += 1
        else:
            # For words, check if token appears (allowing partial matches for flexibility)
            if token in text_upper:
                matches += 1
    
    match_ratio = matches / len(tokens) if tokens else 0
    return match_ratio >= threshold


def localize_inmuebles_in_chunks(
    chunks: List, inmuebles: List, threshold: float = 0.6
) -> Dict[int, Set[int]]:
    """
    Find candidate chunks for each inmueble using text matching.
    
    Args:
        chunks: List of ChunkInfo objects with texto field
        inmuebles: List of InmuebleInput objects
        threshold: Minimum token match ratio (default: 0.6)
    
    Returns:
        Dictionary mapping inmueble index to set of candidate chunk indices
    """
    candidatos: Dict[int, Set[int]] = {}
    
    logger.info(f"Localizing {len(inmuebles)} inmuebles in {len(chunks)} chunks (threshold: {threshold})")
    
    for inmueble_idx, inmueble in enumerate(inmuebles):
        candidatos[inmueble_idx] = set()
        
        # Tokenize inmueble name
        base_tokens = tokenize_inmueble_name(inmueble.inmueble)
        
        # Generate variants
        search_tokens = generate_search_variants(base_tokens)
        
        logger.debug(
            f"Inmueble {inmueble_idx + 1}: '{inmueble.inmueble}' -> "
            f"{len(base_tokens)} base tokens, {len(search_tokens)} search tokens"
        )
        
        # Search in each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if match_tokens_in_text(search_tokens, chunk.texto, threshold):
                candidatos[inmueble_idx].add(chunk_idx)
                logger.debug(
                    f"  Match found in chunk {chunk_idx} "
                    f"(pages {chunk.start_page}-{chunk.end_page})"
                )
        
        match_count = len(candidatos[inmueble_idx])
        if match_count > 0:
            logger.info(
                f"Inmueble {inmueble_idx + 1} '{inmueble.inmueble}': "
                f"found in {match_count} chunk(s)"
            )
        else:
            logger.warning(
                f"Inmueble {inmueble_idx + 1} '{inmueble.inmueble}': "
                f"no matches found (will try LLM refinement)"
            )
    
    # Summary
    total_matches = sum(len(c) for c in candidatos.values())
    unmatched = sum(1 for c in candidatos.values() if not c)
    matched = len(inmuebles) - unmatched
    success_rate = (matched / len(inmuebles) * 100) if inmuebles else 0
    avg_chunks_per_matched = total_matches / matched if matched > 0 else 0
    
    logger.info(
        f"Text localization complete: {matched}/{len(inmuebles)} inmuebles found "
        f"({success_rate:.1f}% success rate), {total_matches} total chunk matches "
        f"(avg {avg_chunks_per_matched:.1f} chunks per inmueble encontrado)"
    )
    
    return candidatos


def refine_pages_for_inmueble(
    inmueble: "InmuebleInput",
    candidate_pages: List[int],
    paginas: List["PaginaOCR"],
    threshold: float = 0.6,
) -> Tuple[List[int], Optional[Dict[str, int]]]:
    """
    Refine page list to only include pages where the inmueble actually appears.
    
    Versión mejorada que detecta límites de la descripción del inmueble.
    
    Args:
        inmueble: InmuebleInput object with inmueble name
        candidate_pages: List of candidate page numbers (from chunks)
        paginas: List of PaginaOCR objects (all pages)
        threshold: Minimum token match ratio (default: 0.6)
    
    Returns:
        (page_numbers, boundaries_dict)
        boundaries_dict contiene: {'start_char': int, 'end_char': int} o None
    """
    if not candidate_pages:
        return [], None
    
    # Usar la nueva función con detección de límites
    pages, start_pos, end_pos = refine_pages_with_boundaries(
        inmueble, candidate_pages, paginas, threshold
    )
    
    boundaries = None
    if start_pos is not None and end_pos is not None:
        boundaries = {
            'start_char': start_pos,
            'end_char': end_pos
        }
        logger.debug(
            f"Refined pages for '{inmueble.inmueble}': "
            f"{len(candidate_pages)} candidate -> {len(pages)} actual pages "
            f"(boundaries detected: {start_pos}-{end_pos})"
        )
    else:
        logger.debug(
            f"Refined pages for '{inmueble.inmueble}': "
            f"{len(candidate_pages)} candidate -> {len(pages)} actual pages "
            f"(no boundaries detected, using context expansion)"
        )
    
    return pages, boundaries


# ============================================================================
# FUNCIONES DE DETECCIÓN DE LÍMITES DE INMUEBLE
# ============================================================================

def detect_inmueble_start(
    text: str, inmueble_name: str, folio: str, search_tokens: Optional[Set[str]] = None
) -> Optional[int]:
    """
    Detecta dónde empieza la descripción del inmueble en el texto.
    
    Busca:
    1. Nombre del inmueble (con variaciones)
    2. "DESCRIPCIÓN Y LINDEROS" o "LINDEROS"
    3. Folio de matrícula
    
    Args:
        text: Texto completo donde buscar
        inmueble_name: Nombre del inmueble
        folio: Folio de matrícula
        search_tokens: Tokens de búsqueda pre-generados (opcional)
    
    Returns:
        Posición en el texto donde empieza, o None
    """
    if not search_tokens:
        base_tokens = tokenize_inmueble_name(inmueble_name)
        search_tokens = generate_search_variants(base_tokens)
    
    text_upper = text.upper()
    
    # Patrones de inicio (en orden de prioridad)
    start_patterns = [
        r"DESCRIPCIÓN\s+Y\s+LINDEROS",
        r"LINDEROS\s+DEL\s+INMUEBLE",
        r"LINDEROS:",
        rf"FOLIO\s+DE\s+MATRÍCULA.*?{re.escape(folio)}",
        rf"MATRÍCULA.*?{re.escape(folio)}",
    ]
    
    # Buscar nombre del inmueble primero (ventana deslizante)
    window_size = 300
    best_match_pos = None
    best_match_score = 0
    
    for i in range(0, len(text) - window_size, 50):  # Avanzar de 50 en 50
        window = text_upper[i:i + window_size]
        matches = sum(1 for token in search_tokens if token in window)
        match_score = matches / len(search_tokens) if search_tokens else 0
        
        if match_score > best_match_score and match_score >= 0.5:  # Al menos 50% de tokens
            # Buscar patrón de inicio cerca
            for pattern in start_patterns:
                match = re.search(pattern, window, re.IGNORECASE)
                if match:
                    best_match_pos = i + match.start()
                    best_match_score = match_score
                    break
    
    # Si no encontramos patrón específico, buscar solo el nombre con alta confianza
    if best_match_pos is None:
        for i in range(0, len(text) - 200, 50):
            window = text_upper[i:i + 200]
            matches = sum(1 for token in search_tokens if token in window)
            match_score = matches / len(search_tokens) if search_tokens else 0
            
            if match_score >= 0.7:  # 70% de tokens = alta confianza
                # Buscar folio cerca
                folio_pattern = rf"{re.escape(folio)}"
                if re.search(folio_pattern, window):
                    best_match_pos = i
                    break
    
    return best_match_pos


def detect_inmueble_end(text: str, start_pos: int, folio: str) -> Optional[int]:
    """
    Detecta dónde termina la descripción del inmueble.
    
    Busca marcadores de fin:
    1. "RÉGIMEN DE PROPIEDAD HORIZONTAL"
    2. "TRADICIÓN ANTERIOR"
    3. "GRAVÁMENES, LIMITACIONES"
    4. "AFECTACIÓN A VIVIENDA FAMILIAR"
    5. "RÉGIMEN DE VIVIENDA DE INTERÉS SOCIAL"
    6. "PARÁGRAFO:" (último elemento)
    7. Inicio de otro inmueble (nuevo folio diferente)
    8. Patrones de inicio de otro inmueble (LOTE, APARTAMENTO, etc.)
    
    Args:
        text: Texto completo
        start_pos: Posición donde empieza la descripción
        folio: Folio del inmueble actual
    
    Returns:
        Posición donde termina, o None
    """
    if start_pos >= len(text):
        return None
    
    text_upper = text.upper()
    search_text = text_upper[start_pos:]
    
    # Límite máximo de búsqueda (3000 caracteres = ~3-4 páginas típicas)
    MAX_SEARCH_LENGTH = 3000
    if len(search_text) > MAX_SEARCH_LENGTH:
        search_text = search_text[:MAX_SEARCH_LENGTH]
    
    # Patrones de fin (en orden de prioridad)
    end_patterns = [
        (r"RÉGIMEN\s+DE\s+PROPIEDAD\s+HORIZONTAL", "after"),
        (r"TRADICIÓN\s+ANTERIOR:", "after"),
        (r"GRAVÁMENES.*?LIMITACIONES.*?DOMINIO", "after"),
        (r"AFECTACIÓN\s+A\s+VIVIENDA\s+FAMILIAR", "after"),
        (r"RÉGIMEN\s+DE\s+VIVIENDA\s+DE\s+INTERÉS\s+SOCIAL", "after"),
        (r"PARÁGRAFO:", "after"),  # Último elemento
        # Patrones más específicos para detectar otro inmueble
        (rf"LOTE\s+\d+.*?CONDOMINIO.*?LA\s+MOLINA", "before"),  # Otro lote del mismo condominio
        (rf"FOLIO\s+DE\s+MATRÍCULA.*?(\d+-\d+)(?!.*{re.escape(folio)})", "before"),  # Otro folio
        (r"LOTE\s+\d+.*?CONDOMINIO|APARTAMENTO\s+\d+.*?TORRE|CASA\s+.*?LOTE", "before"),  # Otro inmueble
    ]
    
    # Buscar el primer patrón de fin
    earliest_end = len(search_text)
    
    for pattern, position in end_patterns:
        matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
        if matches:
            # Tomar el primer match (más cercano al inicio)
            first_match = matches[0]
            if position == "after":
                # Buscar fin de sección (siguiente sección o fin de párrafo)
                end_pos = find_section_end(search_text, first_match.end())
                earliest_end = min(earliest_end, end_pos)
            else:  # before
                # El inmueble termina antes de este patrón
                earliest_end = min(earliest_end, first_match.start())
    
    # Si encontramos un fin, retornarlo
    if earliest_end < len(search_text):
        return start_pos + earliest_end
    
    # Si no encontramos patrón de fin pero el texto es muy largo, limitar
    # (probablemente la detección falló)
    if len(search_text) > 2000:
        logger.debug(
            f"No end pattern found for folio {folio}, limiting to 2000 chars from start"
        )
        return start_pos + 2000
    
    return None


def find_section_end(text: str, start_pos: int) -> int:
    """
    Encuentra el fin de una sección buscando el siguiente marcador.
    
    Args:
        text: Texto donde buscar
        start_pos: Posición donde empezar a buscar
    
    Returns:
        Posición del fin de la sección
    """
    if start_pos >= len(text):
        return len(text)
    
    search_text = text[start_pos:]
    
    # Buscar siguiente sección (título en mayúsculas seguido de salto de línea)
    next_section = re.search(r"\n\n+[A-ZÁÉÍÓÚÑ\s]{10,}\n", search_text)
    if next_section:
        return start_pos + next_section.start()
    
    # Si no hay siguiente sección, buscar fin de párrafo largo
    paragraph_end = re.search(r"\.\s*\n\n+", search_text)
    if paragraph_end:
        return start_pos + paragraph_end.end()
    
    # Fallback: 500 caracteres o hasta el fin del texto
    return min(start_pos + 500, len(text))


def expand_page_range(
    matching_pages: List[int],
    candidate_pages: List[int],
    context_before: int = 1,
    context_after: int = 2,
) -> List[int]:
    """
    Expande el rango de páginas con contexto.
    
    Args:
        matching_pages: Páginas donde se encontró el inmueble
        candidate_pages: Todas las páginas candidatas
        context_before: Páginas antes del match a incluir
        context_after: Páginas después del match a incluir
    
    Returns:
        Lista expandida de páginas
    """
    if not matching_pages:
        return candidate_pages
    
    min_page = min(matching_pages)
    max_page = max(matching_pages)
    
    # Expandir con contexto
    all_pages = sorted(set(candidate_pages))
    
    try:
        start_idx = max(0, all_pages.index(min_page) - context_before)
        end_idx = min(len(all_pages), all_pages.index(max_page) + context_after + 1)
        return all_pages[start_idx:end_idx]
    except ValueError:
        # Si alguna página no está en la lista, usar rango numérico
        return [p for p in all_pages if min_page - context_before <= p <= max_page + context_after]


def refine_pages_with_boundaries(
    inmueble: "InmuebleInput",
    candidate_pages: List[int],
    paginas: List["PaginaOCR"],
    threshold: float = 0.6,
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """
    Refina páginas y detecta límites de la descripción del inmueble.
    
    Args:
        inmueble: InmuebleInput object
        candidate_pages: Lista de páginas candidatas
        paginas: Lista de todas las páginas OCR
        threshold: Threshold para matching de tokens
    
    Returns:
        (page_numbers, start_char_pos, end_char_pos)
        start_char_pos y end_char_pos son posiciones relativas en el texto concatenado
    """
    if not candidate_pages:
        return [], None, None
    
    # Tokenizar y generar variantes
    base_tokens = tokenize_inmueble_name(inmueble.inmueble)
    search_tokens = generate_search_variants(base_tokens)
    
    # Crear mapa de páginas
    pagina_map = {p.page: p for p in paginas}
    
    # 1. Encontrar páginas con match
    matching_pages = []
    for page_num in sorted(candidate_pages):
        if page_num not in pagina_map:
            continue
        pagina = pagina_map[page_num]
        if match_tokens_in_text(search_tokens, pagina.texto, threshold):
            matching_pages.append(page_num)
    
    if not matching_pages:
        # Fallback: si no hay matches, usar solo las primeras páginas candidatas con límite
        MAX_FALLBACK_PAGES = 5
        limited_candidates = candidate_pages[:MAX_FALLBACK_PAGES]
        logger.debug(
            f"No matching pages found for '{inmueble.inmueble}', "
            f"using first {len(limited_candidates)} candidate pages as fallback"
        )
        return sorted(limited_candidates), None, None
    
    # 2. Concatenar texto de páginas candidatas para detectar límites
    candidate_text_pages = sorted(set(candidate_pages))
    full_text_parts = []
    for p in candidate_text_pages:
        if p in pagina_map:
            full_text_parts.append(pagina_map[p].texto)
    
    if not full_text_parts:
        return sorted(matching_pages), None, None
    
    full_text = "\n\n".join(full_text_parts)
    
    # 3. Detectar inicio y fin de descripción
    start_pos = detect_inmueble_start(full_text, inmueble.inmueble, inmueble.folio, search_tokens)
    end_pos = None
    if start_pos is not None:
        end_pos = detect_inmueble_end(full_text, start_pos, inmueble.folio)
    
    # 4. Determinar qué páginas incluir basado en las posiciones
    if start_pos is not None and end_pos is not None:
        # Calcular en qué páginas están start_pos y end_pos
        char_count = 0
        start_page = None
        end_page = None
        
        for page_num in candidate_text_pages:
            if page_num not in pagina_map:
                continue
            pagina = pagina_map[page_num]
            page_start = char_count
            # +2 por el separador "\n\n"
            page_end = char_count + len(pagina.texto) + 2
            
            if start_page is None and start_pos >= page_start and start_pos < page_end:
                start_page = page_num
            if end_pos >= page_start and end_pos < page_end:
                end_page = page_num
                break
            
            char_count = page_end
        
        if start_page and end_page:
            # Incluir todas las páginas entre start_page y end_page
            final_pages = [p for p in candidate_text_pages if start_page <= p <= end_page]
            
            # Límite máximo incluso con límites detectados (por si la detección falla)
            MAX_PAGES_WITH_BOUNDARIES = 15
            if len(final_pages) > MAX_PAGES_WITH_BOUNDARIES:
                logger.warning(
                    f"Detected boundaries for '{inmueble.inmueble}' but range is too large "
                    f"({len(final_pages)} pages), limiting to {MAX_PAGES_WITH_BOUNDARIES} pages"
                )
                final_pages = final_pages[:MAX_PAGES_WITH_BOUNDARIES]
            
            return sorted(final_pages), start_pos, end_pos
    
    # Fallback: usar páginas con match + contexto limitado
    # Si hay muchas páginas con match, agruparlas y tomar el rango más concentrado
    if len(matching_pages) > 10:
        # Si hay muchos matches, encontrar el grupo más concentrado
        # (probablemente el inmueble está en un rango específico)
        sorted_matches = sorted(matching_pages)
        
        # Encontrar el rango más denso (donde hay más matches consecutivos)
        best_start = sorted_matches[0]
        best_end = sorted_matches[0]
        best_count = 1
        current_start = sorted_matches[0]
        current_count = 1
        
        for i in range(1, len(sorted_matches)):
            if sorted_matches[i] - sorted_matches[i-1] <= 3:  # Páginas consecutivas o cercanas
                current_count += 1
            else:
                if current_count > best_count:
                    best_start = current_start
                    best_end = sorted_matches[i-1]
                    best_count = current_count
                current_start = sorted_matches[i]
                current_count = 1
        
        # Verificar el último grupo
        if current_count > best_count:
            best_start = current_start
            best_end = sorted_matches[-1]
            best_count = current_count
        
        # Usar el rango más concentrado
        matching_pages = [p for p in sorted_matches if best_start <= p <= best_end]
        logger.debug(
            f"Many matching pages found for '{inmueble.inmueble}', "
            f"using most concentrated range: {best_start}-{best_end} ({len(matching_pages)} pages)"
        )
    
    expanded_pages = expand_page_range(matching_pages, candidate_pages, context_before=2, context_after=5)
    
    # Límite máximo de páginas cuando no hay límites detectados
    MAX_PAGES_WITHOUT_BOUNDARIES = 20  # Aumentado para dar más margen
    if len(expanded_pages) > MAX_PAGES_WITHOUT_BOUNDARIES:
        # Si el rango expandido es muy grande, limitar alrededor de las páginas con match
        if matching_pages:
            min_match = min(matching_pages)
            max_match = max(matching_pages)
            # Limitar a un rango razonable alrededor de las páginas con match
            # Incluir más contexto después del match (donde suele estar la descripción)
            limited_pages = [
                p for p in candidate_pages 
                if min_match - 2 <= p <= max_match + 12  # Más páginas después del match
            ]
            # Limitar a máximo MAX_PAGES_WITHOUT_BOUNDARIES
            if len(limited_pages) > MAX_PAGES_WITHOUT_BOUNDARIES:
                limited_pages = limited_pages[:MAX_PAGES_WITHOUT_BOUNDARIES]
            
            logger.debug(
                f"Limiting page range for '{inmueble.inmueble}': "
                f"{len(expanded_pages)} -> {len(limited_pages)} pages "
                f"(range: {min_match}-{max_match}, final: {limited_pages[0] if limited_pages else 'N/A'}-{limited_pages[-1] if limited_pages else 'N/A'})"
            )
            return sorted(limited_pages), start_pos, end_pos
    
    return sorted(expanded_pages), start_pos, end_pos


# ============================================================================
# FUNCIONES DE LIMPIEZA DE TEXTO
# ============================================================================

def clean_text_between_pages(
    paginas: List["PaginaOCR"],
    page_numbers: List[int],
    start_pos: Optional[int] = None,
    end_pos: Optional[int] = None,
) -> str:
    """
    Concatena páginas limpiando texto irrelevante entre ellas.
    
    Limpia:
    1. Headers/footers repetitivos
    2. Números de página
    3. Texto de "Papel notarial"
    4. Separadores excesivos
    
    Args:
        paginas: Lista de todas las páginas OCR
        page_numbers: Números de página a incluir
        start_pos: Posición de inicio en texto concatenado (opcional)
        end_pos: Posición de fin en texto concatenado (opcional)
    
    Returns:
        Texto limpio concatenado
    """
    pagina_map = {p.page: p for p in paginas}
    
    # Patrones a limpiar (headers/footers comunes)
    header_footer_patterns = [
        r"Papel notarial para uso exclusivo.*?No tiene costo",
        r"República de Colombia",
        r"Aa\d{9}",  # Números de serie
        r"---+\s*---+",  # Separadores
        r"Página\s+\d+",
        r"Pág\.\s+\d+",
    ]
    
    texts = []
    full_text = ""
    char_offset = 0
    
    for page_num in sorted(page_numbers):
        if page_num not in pagina_map:
            continue
        
        pagina = pagina_map[page_num]
        page_text = pagina.texto
        
        # Limpiar headers/footers comunes
        for pattern in header_footer_patterns:
            page_text = re.sub(pattern, "", page_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Si tenemos límites, extraer solo la parte relevante
        if start_pos is not None and end_pos is not None:
            page_start = char_offset
            page_end = char_offset + len(page_text)
            
            if page_start <= start_pos < page_end:
                # Esta página contiene el inicio
                relative_start = start_pos - page_start
                page_text = page_text[relative_start:]
            elif page_start > start_pos:
                # Esta página está después del inicio
                if page_end > end_pos:
                    # Esta página contiene el fin
                    relative_end = end_pos - page_start
                    page_text = page_text[:relative_end]
                # Si está completamente dentro del rango, usar todo
            else:
                # Esta página está antes del inicio, no incluir
                char_offset += len(page_text) + 2
                continue
        
        texts.append(page_text)
        char_offset += len(page_text) + 2  # +2 por separador
    
    # Unir con separador inteligente
    if start_pos is not None and end_pos is not None:
        # Si hay límites detectados, no agregar separadores extra
        return "\n".join(texts)
    else:
        # Agregar separador claro entre páginas
        return "\n\n".join(texts)


def remove_redundant_separators(text: str) -> str:
    """
    Elimina separadores redundantes y espacios excesivos.
    
    Args:
        text: Texto a limpiar
    
    Returns:
        Texto limpio
    """
    # Reemplazar múltiples saltos de línea con máximo 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Eliminar espacios al inicio/fin de líneas
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Eliminar líneas vacías múltiples
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text.strip()

