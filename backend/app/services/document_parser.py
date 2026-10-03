import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import pymupdf  # PyMuPDF library for PDF processing


class DocumentExtractionError(Exception):
    """Custom exception raised when document text extraction fails."""
    pass


@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    total_pages: int
    pages: List[PageContent]
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "pages": [
                {"page_number": p.page_number, "text": p.text}
                for p in self.pages
            ],
            "full_text": self.full_text,
        }


def extract_text_from_file(file_path: Union[str, Path]) -> ExtractedDocument:
    """
    Extracts text content from a .txt or .pdf file.
    
    Returns an ExtractedDocument object preserving page-level text and metadata.
    Raises DocumentExtractionError for unsupported formats or invalid/empty files.
    """
    path = Path(file_path)

    # 1. Verify file exists
    if not path.exists() or not path.is_file():
        raise DocumentExtractionError(f"File not found: {path}")

    # 2. Check for 0-byte empty file
    if path.stat().st_size == 0:
        raise DocumentExtractionError(f"Document '{path.name}' is empty (0 bytes).")

    ext = path.suffix.lower()
    filename = path.name

    if ext == ".txt":
        return _extract_txt(path, filename)
    elif ext == ".pdf":
        return _extract_pdf(path, filename)
    else:
        raise DocumentExtractionError(
            f"Unsupported file format '{ext}'. Only .txt and .pdf files are supported."
        )


def _extract_txt(path: Path, filename: str) -> ExtractedDocument:
    """Helper to extract text from a plain text file."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as e:
            raise DocumentExtractionError(f"Failed to read TXT file '{filename}': {str(e)}")

    stripped_text = content.strip()
    if not stripped_text:
        raise DocumentExtractionError(f"TXT file '{filename}' contains no readable text.")

    pages = [PageContent(page_number=1, text=stripped_text)]

    return ExtractedDocument(
        filename=filename,
        file_type="txt",
        total_pages=1,
        pages=pages,
        full_text=stripped_text,
    )


def _extract_pdf(path: Path, filename: str) -> ExtractedDocument:
    """Helper to extract text page-by-page from a PDF file using PyMuPDF."""
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise DocumentExtractionError(f"Failed to open PDF file '{filename}': {str(e)}")

    if doc.page_count == 0:
        doc.close()
        raise DocumentExtractionError(f"PDF file '{filename}' has 0 pages.")

    pages: List[PageContent] = []
    full_text_parts: List[str] = []

    for page_num in range(1, doc.page_count + 1):
        page = doc.load_page(page_num - 1)
        page_text = page.get_text().strip()

        pages.append(PageContent(page_number=page_num, text=page_text))
        if page_text:
            full_text_parts.append(page_text)

    doc.close()

    full_text = "\n\n".join(full_text_parts).strip()
    if not full_text:
        raise DocumentExtractionError(f"PDF file '{filename}' contains no readable text.")

    return ExtractedDocument(
        filename=filename,
        file_type="pdf",
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
    )
