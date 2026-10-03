import unittest

from app.services.document_parser import ExtractedDocument, PageContent
from app.services.text_chunker import (
    chunk_document,
    ChunkingError,
    DocumentChunk,
)


class TestTextChunker(unittest.TestCase):

    def setUp(self):
        # Sample ExtractedDocument fixture
        self.sample_doc = ExtractedDocument(
            filename="report.pdf",
            file_type="pdf",
            total_pages=2,
            pages=[
                PageContent(
                    page_number=1,
                    text="Page 1: Artificial Intelligence and Machine Learning form the core of modern software development."
                ),
                PageContent(
                    page_number=2,
                    text="Page 2: Vector databases enable semantic retrieval by storing high-dimensional embeddings."
                ),
            ],
            full_text="Page 1: Artificial Intelligence... Page 2: Vector databases..."
        )

    def test_chunking_normal_text(self):
        """Text shorter than chunk_size should produce a single chunk."""
        chunks = chunk_document(self.sample_doc, chunk_size=500, chunk_overlap=50)
        self.assertEqual(len(chunks), 2)  # 1 chunk per page since text < 500
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[1].page_number, 2)
        self.assertEqual(chunks[0].source_filename, "report.pdf")
        self.assertIn("Artificial Intelligence", chunks[0].text)

    def test_chunking_long_text(self):
        """Long text on a single page should be split into multiple chunks."""
        long_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        doc = ExtractedDocument(
            filename="long.txt",
            file_type="txt",
            total_pages=1,
            pages=[PageContent(page_number=1, text=long_text)],
            full_text=long_text,
        )

        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        # step = 80 -> starts: 0, 80, 160, 240
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 100)
            self.assertEqual(chunk.source_filename, "long.txt")

    def test_chunking_overlap(self):
        """Verifies that consecutive chunks have the expected overlap."""
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 chars
        doc = ExtractedDocument(
            filename="overlap.txt",
            file_type="txt",
            total_pages=1,
            pages=[PageContent(page_number=1, text=text)],
            full_text=text,
        )

        # chunk_size = 20, chunk_overlap = 5 -> step = 15
        chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)
        self.assertGreaterEqual(len(chunks), 2)

        # Chunk 0: text[0:20] -> "0123456789ABCDEFGHIJ"
        # Chunk 1: text[15:35] -> "FGHIJKLMNOPQRSTUVWXY"
        # Overlap: "FGHIJ" (5 chars)
        overlap_str = text[15:20]
        self.assertTrue(chunks[0].text.endswith(overlap_str))
        self.assertTrue(chunks[1].text.startswith(overlap_str))

    def test_chunking_multiple_pages(self):
        """Verifies chunk metadata across multiple pages."""
        chunks = chunk_document(self.sample_doc, chunk_size=30, chunk_overlap=5)
        page_numbers = [c.page_number for c in chunks]
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)
        for chunk in chunks:
            self.assertEqual(chunk.source_filename, "report.pdf")
            self.assertIsNotNone(chunk.chunk_id)

    def test_chunking_empty_content(self):
        """Empty document or empty pages should return empty chunk list."""
        empty_doc = ExtractedDocument(
            filename="empty.txt",
            file_type="txt",
            total_pages=0,
            pages=[],
            full_text="",
        )
        chunks = chunk_document(empty_doc)
        self.assertEqual(chunks, [])

    def test_chunking_invalid_settings(self):
        """Invalid chunk_size or chunk_overlap should raise ChunkingError."""
        # Invalid chunk_size <= 0
        with self.assertRaises(ChunkingError):
            chunk_document(self.sample_doc, chunk_size=0, chunk_overlap=10)

        # Negative chunk_overlap
        with self.assertRaises(ChunkingError):
            chunk_document(self.sample_doc, chunk_size=100, chunk_overlap=-5)

        # chunk_overlap >= chunk_size
        with self.assertRaises(ChunkingError):
            chunk_document(self.sample_doc, chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ChunkingError):
            chunk_document(self.sample_doc, chunk_size=100, chunk_overlap=150)


if __name__ == "__main__":
    unittest.main()
