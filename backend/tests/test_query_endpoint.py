from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.services.retrieval_service import RetrievalError, SearchResult
from app.services.answer_generation_service import AnswerGenerationError, AnswerResult, SourceReference

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_query_success_with_grounded_answer():
    mock_retrieval_service = MagicMock()
    mock_answer_gen_service = MagicMock()

    sample_search_results = [
        SearchResult(
            chunk_id="chunk_001",
            text="Agentic RAG utilizes iterative vector search.",
            source_filename="rag_paper.pdf",
            page_number=1,
            chunk_index=0,
            model_name="text-embedding-004",
            distance=0.10,
        ),
        SearchResult(
            chunk_id="chunk_002",
            text="Vector databases enable low-latency similarity search.",
            source_filename="vector_db.pdf",
            page_number=3,
            chunk_index=2,
            model_name="text-embedding-004",
            distance=0.20,
        ),
    ]

    sample_answer_result = AnswerResult(
        answer="Agentic RAG uses vector search [S1] and vector databases [S2].",
        sources=[
            SourceReference(
                source_label="[S1]",
                source_filename="rag_paper.pdf",
                page_number=1,
                chunk_id="chunk_001",
            ),
            SourceReference(
                source_label="[S2]",
                source_filename="vector_db.pdf",
                page_number=3,
                chunk_id="chunk_002",
            ),
        ],
        model_name="gemini-2.5-flash",
        sufficient_context=True,
    )

    mock_retrieval_service.search.return_value = sample_search_results
    mock_answer_gen_service.generate_answer.return_value = sample_answer_result

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service), \
         patch("app.api.endpoints.get_answer_generation_service", return_value=mock_answer_gen_service):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is Agentic RAG?", "top_k": 4},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is Agentic RAG?"
    assert data["answer"] == "Agentic RAG uses vector search [S1] and vector databases [S2]."
    assert data["sufficient_context"] is True
    assert data["model_name"] == "gemini-2.5-flash"

    # Verify source references mapping without losing fields
    assert len(data["sources"]) == 2
    src0 = data["sources"][0]
    assert src0["source_label"] == "[S1]"
    assert src0["source_filename"] == "rag_paper.pdf"
    assert src0["page_number"] == 1
    assert src0["chunk_id"] == "chunk_001"

    src1 = data["sources"][1]
    assert src1["source_label"] == "[S2]"
    assert src1["source_filename"] == "vector_db.pdf"
    assert src1["page_number"] == 3
    assert src1["chunk_id"] == "chunk_002"

    # Verify parameters passed to services
    mock_retrieval_service.search.assert_called_once_with(
        query="What is Agentic RAG?",
        top_k=4,
    )
    mock_answer_gen_service.generate_answer.assert_called_once_with(
        question="What is Agentic RAG?",
        search_results=sample_search_results,
    )


def test_query_empty_retrieval_results():
    mock_retrieval_service = MagicMock()
    mock_answer_gen_service = MagicMock()

    mock_retrieval_service.search.return_value = []
    mock_answer_gen_service.generate_answer.return_value = AnswerResult(
        answer="I do not have enough information in the provided document context to answer your question.",
        sources=[],
        model_name="gemini-2.5-flash",
        sufficient_context=False,
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service), \
         patch("app.api.endpoints.get_answer_generation_service", return_value=mock_answer_gen_service):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is quantum gravity?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is quantum gravity?"
    assert "do not have enough information" in data["answer"]
    assert data["sources"] == []
    assert data["sufficient_context"] is False


def test_query_retrieval_validation_or_model_mismatch_returns_400():
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = RetrievalError(
        "Model mismatch error: stored chunk uses old-model"
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/query",
            json={"question": "Test question"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid search parameters or embedding model mismatch."
    assert "old-model" not in response.text


def test_query_retrieval_internal_error_returns_500():
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = RetrievalError(
        "ChromaDB connection failure at /internal/db/path SECRET_KEY"
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/query",
            json={"question": "Test question"},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Context retrieval failed. Please try again later."
    assert "SECRET_KEY" not in response.text
    assert "/internal/db/path" not in response.text


def test_query_answer_generation_error_returns_500():
    mock_retrieval_service = MagicMock()
    mock_answer_gen_service = MagicMock()

    mock_retrieval_service.search.return_value = [
        SearchResult(
            chunk_id="chunk_01",
            text="Some text",
            source_filename="file.pdf",
            page_number=1,
            chunk_index=0,
            model_name="text-embedding-004",
            distance=0.1,
        )
    ]
    mock_answer_gen_service.generate_answer.side_effect = AnswerGenerationError(
        "Gemini API answer generation failed: API_KEY_EXPIRED"
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service), \
         patch("app.api.endpoints.get_answer_generation_service", return_value=mock_answer_gen_service):
        response = client.post(
            "/api/v1/query",
            json={"question": "Test question"},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Answer generation failed. Please try again later."
    assert "API_KEY_EXPIRED" not in response.text


def test_query_pydantic_validation_failures():
    # Empty question
    res1 = client.post("/api/v1/query", json={"question": "   ", "top_k": 4})
    assert res1.status_code == 422

    # Non-positive top_k
    res2 = client.post("/api/v1/query", json={"question": "Valid question?", "top_k": 0})
    assert res2.status_code == 422
