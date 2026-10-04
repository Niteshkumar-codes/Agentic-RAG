import unittest
from unittest.mock import MagicMock, patch

from app.services.retrieval_service import SearchResult
from app.services.answer_generation_service import (
    AnswerGenerationService,
    AnswerGenerationError,
    AnswerResult,
    SourceReference,
)


class TestAnswerGenerationService(unittest.TestCase):

    def setUp(self):
        self.sample_result_1 = SearchResult(
            chunk_id="doc1.pdf_p1_c1_123456",
            text="Agentic RAG combines vector retrieval with LLM reasoning.",
            source_filename="doc1.pdf",
            page_number=1,
            chunk_index=1,
            model_name="text-embedding-004",
            distance=0.15,
        )

        self.sample_result_2 = SearchResult(
            chunk_id="doc2.pdf_p3_c5_654321",
            text="ChromaDB stores high-dimensional embedding vectors for fast search.",
            source_filename="doc2.pdf",
            page_number=3,
            chunk_index=5,
            model_name="text-embedding-004",
            distance=0.22,
        )

    def test_missing_api_key_raises_error(self):
        """Service should raise AnswerGenerationError if GEMINI_API_KEY is missing/empty."""
        service = AnswerGenerationService(api_key="")
        with self.assertRaises(AnswerGenerationError) as ctx:
            service.generate_answer("What is RAG?", [self.sample_result_1])
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_blank_question_raises_error(self):
        """Blank or whitespace-only question should raise AnswerGenerationError."""
        service = AnswerGenerationService(api_key="fake_key")
        with self.assertRaises(AnswerGenerationError) as ctx:
            service.generate_answer("   ", [self.sample_result_1])
        self.assertIn("Question cannot be empty", str(ctx.exception))

        with self.assertRaises(AnswerGenerationError) as ctx:
            service.generate_answer("", [self.sample_result_1])
        self.assertIn("Question cannot be empty", str(ctx.exception))

    @patch("app.services.answer_generation_service.genai.Client")
    def test_empty_retrieval_results_does_not_call_gemini(self, mock_genai_client_cls):
        """Empty search results should return insufficient context without creating client or calling Gemini API."""
        service = AnswerGenerationService(api_key="fake_key")
        result = service.generate_answer("What is quantum computing?", [])

        self.assertIsInstance(result, AnswerResult)
        self.assertFalse(result.sufficient_context)
        self.assertEqual(result.sources, [])
        self.assertIn("do not have enough information", result.answer.lower())
        mock_genai_client_cls.assert_not_called()

    @patch("app.services.answer_generation_service.genai.Client")
    def test_generate_answer_success_mocked(self, mock_genai_client_cls):
        """Verify successful answer generation with mocked Gemini response."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Based on [S1], Agentic RAG combines vector retrieval with LLM reasoning."
        mock_client.models.generate_content.return_value = mock_response

        service = AnswerGenerationService(api_key="fake_key", generation_model="gemini-2.5-flash")
        result = service.generate_answer("What is Agentic RAG?", [self.sample_result_1])

        self.assertIsInstance(result, AnswerResult)
        self.assertTrue(result.sufficient_context)
        self.assertEqual(result.model_name, "gemini-2.5-flash")
        self.assertEqual(
            result.answer,
            "Based on [S1], Agentic RAG combines vector retrieval with LLM reasoning."
        )
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].source_label, "[S1]")
        self.assertEqual(result.sources[0].source_filename, "doc1.pdf")
        self.assertEqual(result.sources[0].page_number, 1)
        self.assertEqual(result.sources[0].chunk_id, "doc1.pdf_p1_c1_123456")

        # Verify dictionary output schema
        res_dict = result.to_dict()
        self.assertEqual(res_dict["model_name"], "gemini-2.5-flash")
        self.assertEqual(len(res_dict["sources"]), 1)
        self.assertEqual(res_dict["sources"][0]["source_label"], "[S1]")

    @patch("app.services.answer_generation_service.genai.Client")
    def test_source_metadata_preservation(self, mock_genai_client_cls):
        """Verify filenames, page numbers, and chunk IDs are preserved across multiple sources."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "According to [S1] and [S2], both technologies work together."
        mock_client.models.generate_content.return_value = mock_response

        service = AnswerGenerationService(api_key="fake_key")
        result = service.generate_answer(
            "How do RAG and ChromaDB work?",
            [self.sample_result_1, self.sample_result_2],
        )

        self.assertEqual(len(result.sources), 2)
        s1, s2 = result.sources[0], result.sources[1]

        self.assertEqual(s1.source_label, "[S1]")
        self.assertEqual(s1.source_filename, "doc1.pdf")
        self.assertEqual(s1.page_number, 1)
        self.assertEqual(s1.chunk_id, "doc1.pdf_p1_c1_123456")

        self.assertEqual(s2.source_label, "[S2]")
        self.assertEqual(s2.source_filename, "doc2.pdf")
        self.assertEqual(s2.page_number, 3)
        self.assertEqual(s2.chunk_id, "doc2.pdf_p3_c5_654321")

    @patch("app.services.answer_generation_service.genai.Client")
    def test_prompt_includes_grounding_instructions_and_source_labels(self, mock_genai_client_cls):
        """Verify prompt contains untrusted context security warning, grounding, and source labels."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "[S1] gives the information."
        mock_client.models.generate_content.return_value = mock_response

        service = AnswerGenerationService(api_key="fake_key")
        service.generate_answer("Sample query?", [self.sample_result_1])

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        prompt_sent = call_kwargs.get("contents", "")

        self.assertIn("[S1]", prompt_sent)
        self.assertIn("doc1.pdf", prompt_sent)
        self.assertIn("Page 1", prompt_sent)
        self.assertIn("untrusted data", prompt_sent)
        self.assertIn("do not have enough information", prompt_sent)
        self.assertIn("Sample query?", prompt_sent)

    @patch("app.services.answer_generation_service.genai.Client")
    def test_gemini_api_error_handling(self, mock_genai_client_cls):
        """API failures during content generation should be wrapped in AnswerGenerationError."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Service Unavailable 503")

        service = AnswerGenerationService(api_key="fake_key")
        with self.assertRaises(AnswerGenerationError) as ctx:
            service.generate_answer("Valid query?", [self.sample_result_1])

        self.assertIn("Gemini API answer generation failed", str(ctx.exception))
        self.assertIn("Service Unavailable 503", str(ctx.exception))

    @patch("app.services.answer_generation_service.genai.Client")
    def test_empty_or_none_response_text_handled_gracefully(self, mock_genai_client_cls):
        """Missing or empty response.text from Gemini should produce an insufficient context answer."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        service = AnswerGenerationService(api_key="fake_key")
        result = service.generate_answer("Valid query?", [self.sample_result_1])

        self.assertFalse(result.sufficient_context)
        self.assertIn("do not have enough information", result.answer.lower())

    @patch("app.services.answer_generation_service.genai.Client")
    def test_invalid_source_label_validation_and_stripping(self, mock_genai_client_cls):
        """Valid source labels [S1] are preserved, while unsupplied labels like [S99] are stripped."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        mock_response = MagicMock()
        # Gemini returned a hallucinated [S99] label along with valid [S1]
        mock_response.text = "According to [S1] and [S99], Agentic RAG works well."
        mock_client.models.generate_content.return_value = mock_response

        service = AnswerGenerationService(api_key="fake_key")
        result = service.generate_answer("What is Agentic RAG?", [self.sample_result_1])

        self.assertIn("[S1]", result.answer)
        self.assertNotIn("[S99]", result.answer)
        self.assertEqual(result.answer, "According to [S1] and, Agentic RAG works well.")

    @patch("app.services.answer_generation_service.genai.Client")
    def test_various_insufficient_context_phrases(self, mock_genai_client_cls):
        """Verify sufficient_context is set to False for different wording variations indicating missing context."""
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        insufficient_answers = [
            "This detail is not mentioned in the provided document context.",
            "The context does not specify the exact deployment steps.",
            "I cannot answer based on the provided documents.",
            "There is insufficient information available.",
        ]

        service = AnswerGenerationService(api_key="fake_key")
        for ans in insufficient_answers:
            mock_response = MagicMock()
            mock_response.text = ans
            mock_client.models.generate_content.return_value = mock_response

            result = service.generate_answer("Query", [self.sample_result_1])
            self.assertFalse(result.sufficient_context, f"Failed for answer: {ans}")

    @patch("app.services.answer_generation_service.genai.Client")
    def test_secret_key_redacted_in_exception_message(self, mock_genai_client_cls):
        """If an API exception contains the secret API key, it should be redacted as [REDACTED]."""
        secret_key = "secret_api_key_12345"
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception(
            f"Invalid key {secret_key} passed to endpoint"
        )

        service = AnswerGenerationService(api_key=secret_key)
        with self.assertRaises(AnswerGenerationError) as ctx:
            service.generate_answer("Query?", [self.sample_result_1])

        err_str = str(ctx.exception)
        self.assertNotIn(secret_key, err_str)
        self.assertIn("[REDACTED]", err_str)


if __name__ == "__main__":
    unittest.main()
