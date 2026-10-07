from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.services.retrieval_service import RetrievalError, SearchResult

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_search_success_with_results():
    mock_retrieval_service = MagicMock()
    sample_results = [
        SearchResult(
            chunk_id="chunk_101",
            text="Artificial intelligence and neural networks.",
            source_filename="ai_overview.pdf",
            page_number=1,
            chunk_index=0,
            model_name="text-embedding-004",
            distance=0.1234,
        ),
        SearchResult(
            chunk_id="chunk_102",
            text="Deep learning models require large datasets.",
            source_filename="ai_overview.pdf",
            page_number=2,
            chunk_index=1,
            model_name="text-embedding-004",
            distance=0.2345,
        ),
    ]
    mock_retrieval_service.search.return_value = sample_results

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/search",
            json={"query": "artificial intelligence", "top_k": 2},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "artificial intelligence"
    assert data["results_count"] == 2
    assert len(data["results"]) == 2

    # Verify field mapping for all SearchResult items
    item0 = data["results"][0]
    assert item0["chunk_id"] == "chunk_101"
    assert item0["text"] == "Artificial intelligence and neural networks."
    assert item0["source_filename"] == "ai_overview.pdf"
    assert item0["page_number"] == 1
    assert item0["chunk_index"] == 0
    assert item0["model_name"] == "text-embedding-004"
    assert pytest.approx(item0["distance"], 0.0001) == 0.1234

    item1 = data["results"][1]
    assert item1["chunk_id"] == "chunk_102"
    assert item1["text"] == "Deep learning models require large datasets."
    assert item1["source_filename"] == "ai_overview.pdf"
    assert item1["page_number"] == 2
    assert item1["chunk_index"] == 1
    assert item1["model_name"] == "text-embedding-004"
    assert pytest.approx(item1["distance"], 0.0001) == 0.2345


def test_search_empty_results():
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.return_value = []

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/search",
            json={"query": "unmatched search term", "top_k": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "unmatched search term"
    assert data["results_count"] == 0
    assert data["results"] == []


def test_search_model_mismatch_returns_400():
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = RetrievalError(
        "Model mismatch error: stored chunk 'c1' uses model 'old-model', but query retrieval uses model 'text-embedding-004'."
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid search parameters or embedding model mismatch."
    # Ensure internal exception details and model names are not exposed
    assert "old-model" not in response.text
    assert "stored chunk" not in response.text


def test_search_validation_error_returns_400():
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = RetrievalError(
        "Query cannot be empty or whitespace-only."
    )

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/search",
            json={"query": "valid_string_for_pydantic"},
        )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid search parameters or embedding model mismatch."


def test_search_internal_failure_returns_500():
    raw_sensitive_error = "ChromaDB vector search query failed: sqlite3.OperationalError /path/to/db GEMINI_API_KEY_ABC123"
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search.side_effect = RetrievalError(raw_sensitive_error)

    with patch("app.api.endpoints.get_retrieval_service", return_value=mock_retrieval_service):
        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
        )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Semantic vector search failed. Please try again later."
    # Ensure raw internal details, paths, API keys are completely withheld
    assert "GEMINI_API_KEY_ABC123" not in response.text
    assert "sqlite3" not in response.text
    assert "/path/to/db" not in response.text


def test_search_pydantic_validation_failures():
    # Empty query
    res1 = client.post("/api/v1/search", json={"query": "   ", "top_k": 4})
    assert res1.status_code == 422

    # Negative top_k
    res2 = client.post("/api/v1/search", json={"query": "valid query", "top_k": 0})
    assert res2.status_code == 422
