from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.services.vector_store import VectorStoreError

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_get_collection_stats_success():
    mock_vector_store = MagicMock()
    mock_vector_store.collection_name = "test_custom_collection"
    mock_vector_store.persist_directory = "custom_chroma_dir"
    mock_vector_store.get_collection_info.return_value = {
        "collection_name": "test_custom_collection",
        "persist_directory": "custom_chroma_dir",
        "total_records": 42,
    }

    with patch("app.api.endpoints.get_vector_store_service", return_value=mock_vector_store):
        response = client.get("/api/v1/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == "test_custom_collection"
    assert data["persist_directory"] == "custom_chroma_dir"
    assert data["total_records"] == 42
    mock_vector_store.get_collection_info.assert_called_once()


def test_get_collection_stats_vector_store_error_returns_500():
    mock_vector_store = MagicMock()
    mock_vector_store.get_collection_info.side_effect = VectorStoreError(
        "ChromaDB connection timeout at /secret/db/path with KEY_999"
    )

    with patch("app.api.endpoints.get_vector_store_service", return_value=mock_vector_store):
        response = client.get("/api/v1/stats")

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Failed to retrieve vector store collection statistics."
    # Verify raw error details, secret keys, internal DB paths are not exposed
    assert "KEY_999" not in response.text
    assert "/secret/db/path" not in response.text
    assert "ChromaDB connection timeout" not in response.text
