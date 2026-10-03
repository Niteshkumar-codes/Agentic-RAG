import os
import unittest
from unittest.mock import MagicMock, patch

from app.services.text_chunker import DocumentChunk
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingError,
    EmbeddedChunk,
)


class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        self.sample_chunk = DocumentChunk(
            chunk_id="report.pdf_p1_c1_123456",
            text="Artificial intelligence powers modern agentic systems.",
            source_filename="report.pdf",
            page_number=1,
            chunk_index=1,
            start_char=0,
            end_char=54,
        )

    def test_missing_api_key_raises_error(self):
        """Service should raise EmbeddingError if API key is missing."""
        service = EmbeddingService(api_key="")
        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_text("Sample text")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    @patch("app.services.embedding_service.genai.Client")
    def test_embed_single_text_success_mocked(self, mock_genai_client_cls):
        """Verify successful single text embedding generation using mocked client."""
        # Setup mock client & response
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1, 0.25, -0.42, 0.99]

        mock_response = MagicMock()
        mock_response.embedding = mock_embedding_obj
        mock_client.models.embed_content.return_value = mock_response

        service = EmbeddingService(api_key="fake_test_api_key", model_name="text-embedding-004")
        vector = service.embed_text("Test query string")

        self.assertEqual(vector, [0.1, 0.25, -0.42, 0.99])
        mock_client.models.embed_content.assert_called_once()

    @patch("app.services.embedding_service.genai.Client")
    def test_embed_chunks_preserves_metadata(self, mock_genai_client_cls):
        """Verify chunk embeddings preserve metadata (chunk_id, filename, page, index)."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.5, 0.6, 0.7]
        mock_response = MagicMock()
        mock_response.embedding = mock_embedding_obj
        mock_client.models.embed_content.return_value = mock_response

        service = EmbeddingService(api_key="fake_test_api_key", model_name="text-embedding-004")
        embedded_chunks = service.embed_chunks([self.sample_chunk])

        self.assertEqual(len(embedded_chunks), 1)
        res = embedded_chunks[0]
        self.assertIsInstance(res, EmbeddedChunk)
        self.assertEqual(res.chunk_id, "report.pdf_p1_c1_123456")
        self.assertEqual(res.source_filename, "report.pdf")
        self.assertEqual(res.page_number, 1)
        self.assertEqual(res.chunk_index, 1)
        self.assertEqual(res.embedding, [0.5, 0.6, 0.7])
        self.assertEqual(res.model_name, "text-embedding-004")

    def test_embed_empty_text_raises_error(self):
        """Empty or whitespace-only text should raise EmbeddingError without calling API."""
        service = EmbeddingService(api_key="fake_key")
        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_text("   ")
        self.assertIn("empty or whitespace-only", str(ctx.exception))

    def test_embed_empty_chunks_list(self):
        """Empty chunks list should return an empty list instantly."""
        service = EmbeddingService(api_key="fake_key")
        result = service.embed_chunks([])
        self.assertEqual(result, [])

    @patch("app.services.embedding_service.genai.Client")
    def test_api_error_handling(self, mock_genai_client_cls):
        """Network/API failures should be wrapped cleanly in EmbeddingError."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client
        mock_client.models.embed_content.side_effect = Exception("API Quota Exceeded")

        service = EmbeddingService(api_key="fake_key")
        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_text("Hello world")
        self.assertIn("Gemini API embedding request failed", str(ctx.exception))
        self.assertIn("API Quota Exceeded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
