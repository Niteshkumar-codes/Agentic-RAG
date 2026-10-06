import pytest
from pydantic import ValidationError
from app.schemas import (
    CollectionStatsResponse,
    DocumentIngestionResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceReferenceItem,
)
from app.api.endpoints import router


def test_search_request_validation():
    # Valid query
    req = SearchRequest(query="  test query  ", top_k=5)
    assert req.query == "test query"
    assert req.top_k == 5

    # Invalid empty query
    with pytest.raises(ValidationError):
        SearchRequest(query="   ", top_k=4)

    # Invalid top_k <= 0
    with pytest.raises(ValidationError):
        SearchRequest(query="valid", top_k=0)


def test_query_request_validation():
    # Valid question
    req = QueryRequest(question="What is RAG?", top_k=3)
    assert req.question == "What is RAG?"
    assert req.top_k == 3

    # Invalid blank question
    with pytest.raises(ValidationError):
        QueryRequest(question="", top_k=4)

    # Invalid top_k <= 0
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question?", top_k=-1)


def test_search_result_item_compatibility():
    item = SearchResultItem(
        chunk_id="chunk_1",
        text="Sample text",
        source_filename="doc.pdf",
        page_number=1,
        chunk_index=1,
        model_name="text-embedding-004",
        distance=0.12,
    )
    assert item.chunk_id == "chunk_1"
    assert item.distance == 0.12


def test_source_reference_item_compatibility():
    ref = SourceReferenceItem(
        source_label="[S1]",
        source_filename="doc.pdf",
        page_number=2,
        chunk_id="chunk_2",
    )
    assert ref.source_label == "[S1]"
    assert ref.chunk_id == "chunk_2"


def test_router_routes_registered():
    routes = {route.path: route.methods for route in router.routes}
    assert "/api/v1/documents/ingest" in routes
    assert "POST" in routes["/api/v1/documents/ingest"]
    assert "/api/v1/search" in routes
    assert "POST" in routes["/api/v1/search"]
    assert "/api/v1/query" in routes
    assert "POST" in routes["/api/v1/query"]
    assert "/api/v1/stats" in routes
    assert "GET" in routes["/api/v1/stats"]
