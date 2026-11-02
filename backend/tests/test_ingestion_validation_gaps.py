"""
test_ingestion_validation_gaps.py
==================================
Focused tests for ingestion-related validation gaps not covered by the
existing test suite.  No production code is changed.

Gaps addressed
--------------
1. document_parser: TXT file that is non-empty but contains only
   whitespace characters (' ', '\\t', '\\n').  This is a *different* code
   path from the 0-byte check – it reaches ``_extract_txt`` where
   ``stripped_text`` is empty, raising DocumentExtractionError with the
   message "contains no readable text".

2. document_parser: ExtractedDocument.to_dict() and PageContent fields
   are exercised by the parser but ``to_dict()`` itself has no direct
   unit test.  One test verifies the serialised shape is correct.

3. ingest endpoint: the "no_op" branch (lines 159-168 in endpoints.py).
   When a document is valid but produces zero chunks after chunking, the
   endpoint returns HTTP 200 with ``status="no_op"`` and
   ``total_chunks=0``.  This branch is never reached by existing tests.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.services.document_parser import (
    DocumentExtractionError,
    ExtractedDocument,
    PageContent,
    extract_text_from_file,
)
from app.services.embedding_service import EmbeddedChunk

# ---------------------------------------------------------------------------
# FastAPI test client (mirrors the setup in test_ingest_endpoint.py)
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


@pytest.fixture()
def mock_services():
    """Patch embedding and vector-store services so no real I/O occurs."""
    mock_emb = MagicMock()
    mock_vec = MagicMock()
    mock_vec.collection_name = "test_collection"
    mock_vec.persist_directory = "/tmp/chroma_test"

    def _fake_embed(chunks):
        return [
            EmbeddedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                source_filename=c.source_filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                embedding=[0.1] * 768,
                model_name="text-embedding-004",
            )
            for c in chunks
        ]

    mock_emb.embed_chunks.side_effect = _fake_embed
    mock_vec.add_embedded_chunks.return_value = {
        "added_count": 1,
        "collection_name": "test_collection",
        "status": "success",
        "model_name": "text-embedding-004",
        "vector_dimension": 768,
    }

    with (
        patch("app.api.endpoints.get_embedding_service", return_value=mock_emb),
        patch("app.api.endpoints.get_vector_store_service", return_value=mock_vec),
    ):
        yield mock_emb, mock_vec


# ===========================================================================
# Gap 1 – document_parser: whitespace-only TXT
# ===========================================================================

class TestDocumentParserWhitespaceTxt(unittest.TestCase):
    """TXT files that are non-empty (have bytes) but contain only whitespace."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_whitespace_only_txt_raises_extraction_error(self):
        """A .txt file with only spaces/tabs/newlines should raise DocumentExtractionError."""
        ws_path = self.tmp_path / "whitespace.txt"
        ws_path.write_text("   \t\n\n   ", encoding="utf-8")

        # File exists and is NOT 0 bytes, so the 0-byte guard is bypassed.
        # The error comes from _extract_txt's `stripped_text` check.
        with self.assertRaises(DocumentExtractionError) as ctx:
            extract_text_from_file(ws_path)

        self.assertIn("no readable text", str(ctx.exception))

    def test_whitespace_only_txt_error_message_includes_filename(self):
        """The error message should name the offending file."""
        ws_path = self.tmp_path / "blank_doc.txt"
        ws_path.write_text("\n\n\n", encoding="utf-8")

        with self.assertRaises(DocumentExtractionError) as ctx:
            extract_text_from_file(ws_path)

        self.assertIn("blank_doc.txt", str(ctx.exception))


# ===========================================================================
# Gap 2 – document_parser: ExtractedDocument.to_dict() serialisation
# ===========================================================================

class TestExtractedDocumentToDict(unittest.TestCase):
    """Verify the to_dict() helper on ExtractedDocument produces the expected shape."""

    def _make_doc(self) -> ExtractedDocument:
        pages = [
            PageContent(page_number=1, text="First page content."),
            PageContent(page_number=2, text="Second page content."),
        ]
        return ExtractedDocument(
            filename="report.pdf",
            file_type="pdf",
            total_pages=2,
            pages=pages,
            full_text="First page content.\n\nSecond page content.",
        )

    def test_to_dict_top_level_keys(self):
        doc = self._make_doc()
        d = doc.to_dict()
        self.assertIn("filename", d)
        self.assertIn("file_type", d)
        self.assertIn("total_pages", d)
        self.assertIn("pages", d)
        self.assertIn("full_text", d)

    def test_to_dict_values(self):
        doc = self._make_doc()
        d = doc.to_dict()
        self.assertEqual(d["filename"], "report.pdf")
        self.assertEqual(d["file_type"], "pdf")
        self.assertEqual(d["total_pages"], 2)
        self.assertIn("First page content.", d["full_text"])

    def test_to_dict_pages_structure(self):
        """Each page entry must have 'page_number' and 'text' keys."""
        doc = self._make_doc()
        pages = doc.to_dict()["pages"]
        self.assertEqual(len(pages), 2)
        for page_dict in pages:
            self.assertIn("page_number", page_dict)
            self.assertIn("text", page_dict)

    def test_to_dict_page_values(self):
        doc = self._make_doc()
        pages = doc.to_dict()["pages"]
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["text"], "First page content.")
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertEqual(pages[1]["text"], "Second page content.")


# ===========================================================================
# Gap 3 – ingest endpoint: zero-chunk "no_op" response
# ===========================================================================

def test_ingest_no_op_when_zero_chunks(mock_services):
    """
    When chunk_document() returns an empty list, the endpoint must respond
    with HTTP 200 and status='no_op' rather than proceeding to embedding.

    We simulate this by patching chunk_document to return [].
    """
    mock_emb, mock_vec = mock_services
    content = b"Valid text that will appear chunked as empty for this test."
    files = {"file": ("sample.txt", io.BytesIO(content), "text/plain")}

    with patch("app.api.endpoints.chunk_document", return_value=[]) as mock_chunk:
        response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_op"
    assert data["total_chunks"] == 0
    assert data["added_count"] == 0
    # Embedding and vector store must NOT have been called
    mock_emb.embed_chunks.assert_not_called()
    mock_vec.add_embedded_chunks.assert_not_called()


def test_ingest_no_op_response_fields_complete(mock_services):
    """
    The no_op response must include all required DocumentIngestionResponse fields.
    """
    content = b"Another valid document."
    files = {"file": ("another.txt", io.BytesIO(content), "text/plain")}

    with patch("app.api.endpoints.chunk_document", return_value=[]):
        response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    # All required schema fields must be present
    for field in ("filename", "total_pages", "total_chunks", "added_count",
                  "collection_name", "vector_dimension", "status"):
        assert field in data, f"Missing field: {field}"

    assert data["filename"] == "another.txt"
    assert data["status"] == "no_op"
    assert data["vector_dimension"] == 0
