from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid

from app.services.document_parser import ExtractedDocument, PageContent


class ChunkingError(Exception):
    """Custom exception raised when text chunking validation fails."""
    pass


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Splits an ExtractedDocument into smaller text chunks page by page.

    :param document: ExtractedDocument object containing pages and metadata.
    :param chunk_size: Maximum number of characters per chunk (default: 500).
    :param chunk_overlap: Number of overlapping characters between consecutive chunks (default: 100).
    :return: List of DocumentChunk objects with metadata.
    :raises ChunkingError: If chunk_size or chunk_overlap configuration is invalid.
    """
    # 1. Validate chunk parameters
    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be greater than 0, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ChunkingError(f"chunk_overlap cannot be negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    # 2. Check for empty document or pages
    if not document.pages:
        return []

    chunks: List[DocumentChunk] = []
    global_chunk_counter = 0

    # 3. Process page by page
    for page in document.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        text_length = len(page_text)
        step = chunk_size - chunk_overlap

        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_str = page_text[start:end]

            # Avoid empty or whitespace-only chunks
            if chunk_str.strip():
                global_chunk_counter += 1
                chunk_id = f"{document.filename}_p{page.page_number}_c{global_chunk_counter}_{uuid.uuid4().hex[:6]}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    source_filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_counter,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)

            # Move sliding window
            if end >= text_length:
                break
            start += step

    return chunks
