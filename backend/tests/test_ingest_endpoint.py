import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.services.document_parser import DocumentExtractionError
from app.services.text_chunker import ChunkingError
from app.services.embedding_service import EmbeddedChunk, EmbeddingError
from app.services.vector_store import VectorStoreError

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def mock_services():
    mock_emb = MagicMock()
    mock_vec = MagicMock()

    mock_vec.collection_name = "test_collection"

    def fake_embed_chunks(chunks):
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

    mock_emb.embed_chunks.side_effect = fake_embed_chunks

    mock_vec.add_embedded_chunks.return_value = {
        "added_count": 2,
        "collection_name": "test_collection",
        "status": "success",
        "model_name": "text-embedding-004",
        "vector_dimension": 768,
    }

    with patch("app.api.endpoints.get_embedding_service", return_value=mock_emb), \
         patch("app.api.endpoints.get_vector_store_service", return_value=mock_vec):
        yield mock_emb, mock_vec


def test_ingest_successful_txt(mock_services):
    content = b"This is a test document content for ingestion API endpoint testing."
    files = {"file": ("test_doc.txt", io.BytesIO(content), "text/plain")}

    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.txt"
    assert data["total_pages"] == 1
    assert data["total_chunks"] > 0
    assert data["added_count"] == 2
    assert data["collection_name"] == "test_collection"
    assert data["vector_dimension"] == 768
    assert data["status"] == "success"


def test_ingest_unsupported_extension(mock_services):
    files = {"file": ("image.png", io.BytesIO(b"fake image data"), "image/png")}

    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 400
    assert "Unsupported file format '.png'" in response.json()["detail"]


def test_ingest_empty_file(mock_services):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 400
    assert "is empty (0 bytes)" in response.json()["detail"]


def test_ingest_invalid_chunk_params(mock_services):
    content = b"Some valid document text content."
    files = {"file": ("sample.txt", io.BytesIO(content), "text/plain")}

    # chunk_size <= 0
    res1 = client.post(
        "/api/v1/documents/ingest?chunk_size=0", files=files
    )
    assert res1.status_code == 400
    assert "chunk_size must be greater than 0" in res1.json()["detail"]

    # chunk_overlap < 0
    files["file"] = ("sample.txt", io.BytesIO(content), "text/plain")
    res2 = client.post(
        "/api/v1/documents/ingest?chunk_overlap=-5", files=files
    )
    assert res2.status_code == 400
    assert "chunk_overlap cannot be negative" in res2.json()["detail"]

    # chunk_overlap >= chunk_size
    files["file"] = ("sample.txt", io.BytesIO(content), "text/plain")
    res3 = client.post(
        "/api/v1/documents/ingest?chunk_size=100&chunk_overlap=100", files=files
    )
    assert res3.status_code == 400
    assert "must be strictly less than chunk_size" in res3.json()["detail"]


def test_ingest_document_extraction_error(mock_services):
    with patch("app.api.endpoints.extract_text_from_file") as mock_extract:
        mock_extract.side_effect = DocumentExtractionError("Corrupt file content")
        files = {"file": ("sample.txt", io.BytesIO(b"corrupt"), "text/plain")}
        response = client.post("/api/v1/documents/ingest", files=files)

        assert response.status_code == 400
        assert "Corrupt file content" in response.json()["detail"]


def test_ingest_chunking_error(mock_services):
    with patch("app.api.endpoints.chunk_document") as mock_chunk:
        mock_chunk.side_effect = ChunkingError("Chunking error occurred")
        files = {"file": ("sample.txt", io.BytesIO(b"Valid text content"), "text/plain")}
        response = client.post("/api/v1/documents/ingest", files=files)

        assert response.status_code == 400
        assert "Chunking error occurred" in response.json()["detail"]


def test_ingest_embedding_error(mock_services):
    mock_emb, _ = mock_services
    mock_emb.embed_chunks.side_effect = EmbeddingError("SECRET_API_KEY_OR_INTERNAL_TRACE")

    files = {"file": ("sample.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Embedding generation failed. Please try again later."
    assert "SECRET_API_KEY_OR_INTERNAL_TRACE" not in detail


def test_ingest_vector_store_duplicate_error(mock_services):
    _, mock_vec = mock_services
    mock_vec.add_embedded_chunks.side_effect = VectorStoreError(
        "Duplicate chunk IDs found in collection: ['chunk_1_secret_id']. Set overwrite_duplicates=True"
    )

    files = {"file": ("sample.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail == "Duplicate document chunks already exist. Set overwrite_duplicates=true to replace them."
    assert "chunk_1_secret_id" not in detail


def test_ingest_general_vector_store_error(mock_services):
    _, mock_vec = mock_services
    mock_vec.add_embedded_chunks.side_effect = VectorStoreError("Internal SQLite/ChromaDB connection broken at line 42")

    files = {"file": ("sample.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Vector store operation failed. Please try again later."
    assert "ChromaDB" not in detail
    assert "SQLite" not in detail


def test_ingest_temp_file_cleanup(mock_services):
    created_temp_paths = []

    with patch("app.api.endpoints.extract_text_from_file") as mock_extract:
        def capture_and_raise(path):
            created_temp_paths.append(Path(path))
            raise DocumentExtractionError("Parsing failed intentionally for test")

        mock_extract.side_effect = capture_and_raise

        files = {"file": ("sample.txt", io.BytesIO(b"Some text for cleanup test"), "text/plain")}
        response = client.post("/api/v1/documents/ingest", files=files)

        assert response.status_code == 400
        assert len(created_temp_paths) == 1
        temp_path = created_temp_paths[0]
        # Confirm temporary file was deleted by finally block
        assert not temp_path.exists()
