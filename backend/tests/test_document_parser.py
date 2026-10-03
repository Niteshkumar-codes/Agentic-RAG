import os
import tempfile
import unittest
from pathlib import Path
import pymupdf

from app.services.document_parser import (
    extract_text_from_file,
    DocumentExtractionError,
    ExtractedDocument,
)


class TestDocumentParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # 1. Valid TXT file
        self.txt_path = self.tmp_path / "sample.txt"
        self.txt_path.write_text("Hello, this is a sample text file for Agentic-RAG.", encoding="utf-8")

        # 2. Valid Multi-page PDF file created with PyMuPDF
        self.pdf_path = self.tmp_path / "sample.pdf"
        doc = pymupdf.open()

        page1 = doc.new_page()
        page1.insert_text((50, 50), "Page 1: Introduction to Agentic RAG Architecture.")

        page2 = doc.new_page()
        page2.insert_text((50, 50), "Page 2: Vector Database and Embedding Retrieval.")

        doc.save(str(self.pdf_path))
        doc.close()

        # 3. Empty TXT file (0 bytes)
        self.empty_txt_path = self.tmp_path / "empty.txt"
        self.empty_txt_path.write_text("", encoding="utf-8")

        # 4. Unsupported file extension (.docx)
        self.unsupported_path = self.tmp_path / "sample.docx"
        self.unsupported_path.write_text("Dummy docx content", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_txt_success(self):
        result = extract_text_from_file(self.txt_path)
        self.assertIsInstance(result, ExtractedDocument)
        self.assertEqual(result.filename, "sample.txt")
        self.assertEqual(result.file_type, "txt")
        self.assertEqual(result.total_pages, 1)
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertIn("Hello, this is a sample text file", result.pages[0].text)
        self.assertIn("Hello, this is a sample text file", result.full_text)

    def test_extract_pdf_success(self):
        result = extract_text_from_file(self.pdf_path)
        self.assertIsInstance(result, ExtractedDocument)
        self.assertEqual(result.filename, "sample.pdf")
        self.assertEqual(result.file_type, "pdf")
        self.assertEqual(result.total_pages, 2)
        self.assertEqual(len(result.pages), 2)

        # Page 1 assertion
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertIn("Page 1: Introduction to Agentic RAG Architecture.", result.pages[0].text)

        # Page 2 assertion
        self.assertEqual(result.pages[1].page_number, 2)
        self.assertIn("Page 2: Vector Database and Embedding Retrieval.", result.pages[1].text)

        # Full text assertion
        self.assertIn("Page 1:", result.full_text)
        self.assertIn("Page 2:", result.full_text)

    def test_extract_empty_txt_raises_error(self):
        with self.assertRaises(DocumentExtractionError) as ctx:
            extract_text_from_file(self.empty_txt_path)
        self.assertIn("is empty", str(ctx.exception))

    def test_extract_unsupported_file_raises_error(self):
        with self.assertRaises(DocumentExtractionError) as ctx:
            extract_text_from_file(self.unsupported_path)
        self.assertIn("Unsupported file format '.docx'", str(ctx.exception))

    def test_extract_non_existent_file_raises_error(self):
        non_existent = self.tmp_path / "does_not_exist.pdf"
        with self.assertRaises(DocumentExtractionError) as ctx:
            extract_text_from_file(non_existent)
        self.assertIn("File not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
