"""Chunking utilities for dividing escritura pages into overlapping chunks."""

import logging
from typing import Dict, List, Optional

from app.models.full_process import ChunkInfo, PaginaOCR

logger = logging.getLogger(__name__)


def create_chunks_with_overlap(
    paginas: List[PaginaOCR], chunk_size: int = 40, overlap: int = 5
) -> List[ChunkInfo]:
    """
    Divide pages into chunks with overlap to avoid cutting inmuebles at boundaries.
    
    Args:
        paginas: List of PaginaOCR objects (ordered by page number)
        chunk_size: Number of pages per chunk (default: 40)
        overlap: Number of pages to overlap between chunks (default: 5)
    
    Returns:
        List of ChunkInfo objects with overlapping page ranges
    """
    if not paginas:
        logger.warning("No pages provided for chunking")
        return []
    
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    
    chunks = []
    step = chunk_size - overlap  # How many pages to advance between chunks
    
    logger.info(
        f"Creating chunks: {len(paginas)} pages, "
        f"chunk_size={chunk_size}, overlap={overlap}, step={step}"
    )
    
    for start_idx in range(0, len(paginas), step):
        end_idx = min(start_idx + chunk_size, len(paginas))
        
        # Get pages for this chunk
        chunk_paginas = paginas[start_idx:end_idx]
        
        # Concatenate text from all pages in chunk
        chunk_text = "\n\n".join(p.texto for p in chunk_paginas)
        
        # Get page numbers (1-indexed)
        start_page = chunk_paginas[0].page
        end_page = chunk_paginas[-1].page
        
        chunk = ChunkInfo(
            index=len(chunks),
            start_page=start_page,
            end_page=end_page,
            texto=chunk_text,
        )
        
        chunks.append(chunk)
        
        logger.debug(
            f"Created chunk {chunk.index}: pages {start_page}-{end_page} "
            f"({len(chunk_paginas)} pages, {len(chunk_text)} chars)"
        )
        
        # Stop if we've covered all pages
        if end_idx >= len(paginas):
            break
    
    logger.info(f"Created {len(chunks)} chunks with overlap")
    return chunks


def get_pages_from_chunks(
    chunk_indices: List[int], chunks: List[ChunkInfo]
) -> List[int]:
    """
    Convert chunk indices to a sorted list of unique page numbers.
    
    Args:
        chunk_indices: List of chunk indices to extract pages from
        chunks: List of ChunkInfo objects
    
    Returns:
        Sorted list of unique page numbers (1-indexed)
    """
    pages = set()
    
    for chunk_idx in chunk_indices:
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            logger.warning(f"Invalid chunk index: {chunk_idx} (max: {len(chunks) - 1})")
            continue
        
        chunk = chunks[chunk_idx]
        # Add all pages in the chunk range
        for page in range(chunk.start_page, chunk.end_page + 1):
            pages.add(page)
    
    return sorted(pages)


def concatenate_pages(
    paginas: List[PaginaOCR],
    page_numbers: List[int],
    boundaries: Optional[Dict[str, int]] = None,
) -> str:
    """
    Concatenate text from specific pages with optional cleaning.
    
    Args:
        paginas: List of all PaginaOCR objects
        page_numbers: List of page numbers to concatenate (1-indexed)
        boundaries: Optional dict with 'start_char' and 'end_char' positions
    
    Returns:
        Concatenated text from the specified pages (cleaned if boundaries provided)
    """
    # Import here to avoid circular imports
    from app.utils.text_localization import clean_text_between_pages, remove_redundant_separators
    
    start_pos = boundaries.get('start_char') if boundaries else None
    end_pos = boundaries.get('end_char') if boundaries else None
    
    # Use cleaning function if boundaries are provided
    if start_pos is not None and end_pos is not None:
        text = clean_text_between_pages(paginas, page_numbers, start_pos, end_pos)
        return remove_redundant_separators(text)
    
    # Fallback to simple concatenation
    pagina_map = {p.page: p for p in paginas}
    
    # Get texts in page order
    texts = []
    for page_num in sorted(page_numbers):
        if page_num in pagina_map:
            texts.append(pagina_map[page_num].texto)
        else:
            logger.warning(f"Page {page_num} not found in paginas list")
    
    return "\n\n".join(texts)

