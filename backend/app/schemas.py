from typing import List
from pydantic import BaseModel, Field, field_validator


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion operation."""
    filename: str
    total_pages: int
    total_chunks: int
    added_count: int
    collection_name: str
    vector_dimension: int
    status: str


class SearchRequest(BaseModel):
    """Request model for semantic vector search."""
    query: str = Field(..., description="User query for semantic search")
    top_k: int = Field(4, gt=0, description="Maximum number of relevant chunks to retrieve")

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v.strip()


class SearchResultItem(BaseModel):
    """Response item representing a single retrieved search chunk."""
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    model_name: str
    distance: float


class SearchResponse(BaseModel):
    """Response model for semantic vector search."""
    query: str
    results_count: int
    results: List[SearchResultItem]


class QueryRequest(BaseModel):
    """Request model for RAG question answering."""
    question: str = Field(..., description="User question for RAG answer generation")
    top_k: int = Field(4, gt=0, description="Maximum number of context chunks to retrieve")

    @field_validator("question")
    @classmethod
    def validate_question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return v.strip()


class SourceReferenceItem(BaseModel):
    """Response item representing a single source reference for generated answer."""
    source_label: str
    source_filename: str
    page_number: int
    chunk_id: str


class QueryResponse(BaseModel):
    """Response model for RAG question answering."""
    question: str
    answer: str
    sources: List[SourceReferenceItem]
    sufficient_context: bool
    model_name: str


class CollectionStatsResponse(BaseModel):
    """Response model for vector store collection statistics."""
    collection_name: str
    persist_directory: str
    total_records: int
