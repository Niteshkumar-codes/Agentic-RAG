import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from app.services.embedding_service import EmbeddingService, EmbeddedChunk
from app.services.vector_store import VectorStoreService
from app.services.retrieval_service import (
    RetrievalService,
    RetrievalError,
    SearchResult,
)


class TestRetrievalService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.persist_path = self.temp_dir.name

        self.vector_store = VectorStoreService(
            persist_directory=self.persist_path,
            collection_name="test_retrieval_collection",
        )

        # Mock embedding service
        self.mock_embedding_service = MagicMock(spec=EmbeddingService)
        self.mock_embedding_service.model_name = "text-embedding-004"
        # Mock embed_text to return dummy vector [0.1, 0.2, 0.3, 0.4]
        self.mock_embedding_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]

        self.retrieval_service = RetrievalService(
            embedding_service=self.mock_embedding_service,
            vector_store_service=self.vector_store,
        )

        # Seed vector store with sample chunks
        self.chunk_1 = EmbeddedChunk(
            chunk_id="doc1.pdf_p1_c1_aaa111",
            text="Machine learning algorithms analyze large datasets.",
            source_filename="doc1.pdf",
            page_number=1,
            chunk_index=1,
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="text-embedding-004",
        )

        self.chunk_2 = EmbeddedChunk(
            chunk_id="doc1.pdf_p2_c2_bbb222",
            text="Vector databases store embeddings for similarity search.",
            source_filename="doc1.pdf",
            page_number=2,
            chunk_index=2,
            embedding=[0.9, 0.8, 0.7, 0.6],
            model_name="text-embedding-004",
        )

    def tearDown(self):
        del self.vector_store
        del self.retrieval_service
        try:
            self.temp_dir.cleanup()
        except Exception:
            shutil.rmtree(self.persist_path, ignore_errors=True)

    def test_semantic_search_success(self):
        """Verify successful search returning results sorted by distance."""
        self.vector_store.add_embedded_chunks([self.chunk_1, self.chunk_2])

        results = self.retrieval_service.search("machine learning", top_k=2)

        self.assertEqual(len(results), 2)
        top_match = results[0]
        self.assertIsInstance(top_match, SearchResult)
        self.assertEqual(top_match.chunk_id, "doc1.pdf_p1_c1_aaa111")
        self.assertEqual(top_match.source_filename, "doc1.pdf")
        self.assertEqual(top_match.page_number, 1)
        self.assertEqual(top_match.model_name, "text-embedding-004")
        self.assertIsNotNone(top_match.distance)

    def test_empty_collection_returns_empty_list(self):
        """Search on an empty collection should return an empty list gracefully."""
        results = self.retrieval_service.search("any query", top_k=3)
        self.assertEqual(results, [])

    def test_blank_query_raises_error(self):
        """Blank or whitespace query should raise RetrievalError."""
        with self.assertRaises(RetrievalError) as ctx:
            self.retrieval_service.search("   ", top_k=4)
        self.assertIn("Query cannot be empty", str(ctx.exception))

    def test_invalid_top_k_raises_error(self):
        """Invalid top_k (<= 0) should raise RetrievalError."""
        with self.assertRaises(RetrievalError) as ctx:
            self.retrieval_service.search("valid query", top_k=0)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        with self.assertRaises(RetrievalError) as ctx:
            self.retrieval_service.search("valid query", top_k=-5)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

    def test_model_mismatch_raises_error(self):
        """If collection record has a different model_name, raise RetrievalError."""
        mismatched_chunk = EmbeddedChunk(
            chunk_id="mismatched_1",
            text="Content created with old model",
            source_filename="old.pdf",
            page_number=1,
            chunk_index=1,
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="old-embedding-model-v0",
        )
        self.vector_store.add_embedded_chunks([mismatched_chunk])

        with self.assertRaises(RetrievalError) as ctx:
            self.retrieval_service.search("test query", top_k=1)
        self.assertIn("Model mismatch error", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
