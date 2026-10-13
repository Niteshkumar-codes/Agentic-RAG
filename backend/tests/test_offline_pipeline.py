import gc
import hashlib
import math
from pathlib import Path
import shutil
import tempfile
from typing import List
import unittest

import pymupdf

from app.services.document_parser import extract_text_from_file, ExtractedDocument
from app.services.text_chunker import chunk_document, DocumentChunk
from app.services.embedding_service import EmbeddingService, EmbeddedChunk
from app.services.vector_store import VectorStoreService
from app.services.retrieval_service import RetrievalService, SearchResult


def generate_deterministic_embedding(text: str, dimension: int = 16) -> List[float]:
    """
    Generates a deterministic unit-normalized float vector for given text.
    Uses token hashing so overlapping words produce similar vectors without Gemini.
    """
    tokens = text.lower().split()
    vector = [0.0] * dimension
    if not tokens:
        vector[0] = 1.0
        return vector

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        bucket = h % dimension
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector

    return [round(v / norm, 6) for v in vector]


class DeterministicOfflineEmbeddingService(EmbeddingService):
    """
    Offline embedding service for testing. Inherits from EmbeddingService and
    overrides embed_text with a deterministic hash-based generator.
    Never instantiates Google GenAI client or makes network calls.
    """

    def __init__(self, dimension: int = 16, model_name: str = "synthetic-offline-v1"):
        super().__init__(api_key="synthetic_offline_key", model_name=model_name)
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            from app.services.embedding_service import EmbeddingError
            raise EmbeddingError("Cannot generate embedding for empty or whitespace-only text.")
        return generate_deterministic_embedding(text, dimension=self.dimension)


class TestOfflineRAGPipeline(unittest.TestCase):
    """
    End-to-end offline integration test verifying the complete RAG pipeline:
    DocumentParser -> TextChunker -> Deterministic Embedding -> VectorStoreService -> RetrievalService.
    Operates in an isolated temporary directory without touching project ChromaDB or Gemini.
    """

    def setUp(self):
        # 1. Create isolated temporary directory for test artifacts and ChromaDB
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.persist_path = str(self.tmp_path / "chroma_test_db")

        # 2. Create a realistic multi-page PDF document locally with PyMuPDF
        self.pdf_filename = "offline_architecture_guide.pdf"
        self.pdf_path = self.tmp_path / self.pdf_filename

        doc = pymupdf.open()

        page1 = doc.new_page()
        page1.insert_text(
            (50, 50),
            "Page 1: Deep learning models and neural network architectures for modern machine learning systems."
        )

        page2 = doc.new_page()
        page2.insert_text(
            (50, 50),
            "Page 2: Traditional culinary baking techniques using sourdough yeast and artisan bread recipes."
        )

        doc.save(str(self.pdf_path))
        doc.close()

        # 3. Instantiate offline services using dependency injection
        self.embedding_service = DeterministicOfflineEmbeddingService(dimension=16)
        self.vector_store_service = VectorStoreService(
            persist_directory=self.persist_path,
            collection_name="test_offline_rag_collection",
        )
        self.retrieval_service = RetrievalService(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
        )

    def tearDown(self):
        # Clean up service references and garbage collect open SQLite handles
        del self.retrieval_service
        del self.vector_store_service
        del self.embedding_service
        gc.collect()

        try:
            self.temp_dir.cleanup()
        except Exception:
            shutil.rmtree(self.persist_path, ignore_errors=True)

    def test_full_offline_pipeline_ingest_and_retrieval(self):
        """
        Executes and validates the full pipeline:
        1. DocumentParser extracts text and page metadata from PDF
        2. TextChunker creates bounded chunks with metadata
        3. EmbeddingService generates deterministic vectors
        4. VectorStoreService stores records in isolated ChromaDB
        5. RetrievalService retrieves relevant chunks matching queries
        """
        # --- Step 1: Real Document Parsing ---
        extracted_doc: ExtractedDocument = extract_text_from_file(self.pdf_path)

        self.assertEqual(extracted_doc.filename, self.pdf_filename)
        self.assertEqual(extracted_doc.file_type, "pdf")
        self.assertEqual(extracted_doc.total_pages, 2)
        self.assertEqual(len(extracted_doc.pages), 2)
        self.assertIn("Deep learning models", extracted_doc.pages[0].text)
        self.assertIn("culinary baking techniques", extracted_doc.pages[1].text)

        # --- Step 2: Real Text Chunking ---
        chunks: List[DocumentChunk] = chunk_document(
            extracted_doc, chunk_size=300, chunk_overlap=50
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_filename, self.pdf_filename)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[1].source_filename, self.pdf_filename)
        self.assertEqual(chunks[1].page_number, 2)

        # --- Step 3: Deterministic Embedding Generation ---
        embedded_chunks: List[EmbeddedChunk] = self.embedding_service.embed_chunks(chunks)

        self.assertEqual(len(embedded_chunks), len(chunks))
        for emb_chunk in embedded_chunks:
            self.assertEqual(len(emb_chunk.embedding), 16)
            self.assertEqual(emb_chunk.model_name, "synthetic-offline-v1")
            self.assertEqual(emb_chunk.source_filename, self.pdf_filename)

        # --- Step 4: Real ChromaDB Vector Storage ---
        store_result = self.vector_store_service.add_embedded_chunks(embedded_chunks)

        self.assertEqual(store_result["status"], "success")
        self.assertEqual(store_result["added_count"], len(embedded_chunks))
        self.assertEqual(store_result["vector_dimension"], 16)

        # Verify collection count in temporary database
        collection_stats = self.vector_store_service.get_collection_info()
        self.assertEqual(collection_stats["total_records"], len(embedded_chunks))

        # --- Step 5: Semantic Retrieval & Ranking ---
        # Query matching Page 1 content (neural networks)
        results_page1: List[SearchResult] = self.retrieval_service.search(
            query="neural network architectures", top_k=2
        )

        self.assertGreater(len(results_page1), 0)
        top_match_p1 = results_page1[0]
        self.assertIsInstance(top_match_p1, SearchResult)
        self.assertEqual(top_match_p1.source_filename, self.pdf_filename)
        self.assertEqual(top_match_p1.page_number, 1)
        self.assertIn("neural network architectures", top_match_p1.text)
        self.assertIsInstance(top_match_p1.distance, float)

        # Query matching Page 2 content (sourdough bread baking)
        results_page2: List[SearchResult] = self.retrieval_service.search(
            query="sourdough bread baking recipes", top_k=2
        )

        self.assertGreater(len(results_page2), 0)
        top_match_p2 = results_page2[0]
        self.assertIsInstance(top_match_p2, SearchResult)
        self.assertEqual(top_match_p2.source_filename, self.pdf_filename)
        self.assertEqual(top_match_p2.page_number, 2)
        self.assertIn("sourdough yeast", top_match_p2.text)
        self.assertIsInstance(top_match_p2.distance, float)


if __name__ == "__main__":
    unittest.main()
