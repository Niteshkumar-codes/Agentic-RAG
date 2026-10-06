import os
from pathlib import Path
import tempfile
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import (
    CollectionStatsResponse,
    DocumentIngestionResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.document_parser import DocumentExtractionError, extract_text_from_file
from app.services.text_chunker import ChunkingError, chunk_document
from app.services.embedding_service import EmbeddingError, EmbeddingService
from app.services.vector_store import VectorStoreError, VectorStoreService

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])

# Shared service instances
_embedding_service: Optional[EmbeddingService] = None
_vector_store_service: Optional[VectorStoreService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store_service() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service


@router.post(
    "/documents/ingest",
    response_model=DocumentIngestionResponse,
    summary="Ingest a document file (.pdf or .txt)",
)
def ingest_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    overwrite_duplicates: bool = False,
) -> DocumentIngestionResponse:
    """
    Ingests a document file (.pdf or .txt), extracts text, chunks it, generates embeddings,
    and stores vectors in ChromaDB.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be provided.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported.",
        )

    # Pre-validate chunk parameters
    if chunk_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_size must be greater than 0, got {chunk_size}.",
        )
    if chunk_overlap < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_overlap cannot be negative, got {chunk_overlap}.",
        )
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}).",
        )

    embedding_service = get_embedding_service()
    vector_store_service = get_vector_store_service()

    # Save uploaded file stream to temporary location on disk
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = Path(temp_file.name)

    try:
        content = file.file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty (0 bytes).",
            )
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()

        # 1. Document parsing
        try:
            extracted_doc = extract_text_from_file(temp_path)
        except DocumentExtractionError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Preserve original uploaded filename in extracted document
        extracted_doc.filename = file.filename

        # 2. Text chunking
        try:
            chunks = chunk_document(
                extracted_doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        except ChunkingError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        if not chunks:
            return DocumentIngestionResponse(
                filename=file.filename,
                total_pages=extracted_doc.total_pages,
                total_chunks=0,
                added_count=0,
                collection_name=vector_store_service.collection_name,
                vector_dimension=0,
                status="no_op",
            )

        # 3. Embedding generation
        try:
            embedded_chunks = embedding_service.embed_chunks(chunks)
        except EmbeddingError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed. Please try again later.",
            )

        # 4. Vector storage
        try:
            store_result = vector_store_service.add_embedded_chunks(
                embedded_chunks, overwrite_duplicates=overwrite_duplicates
            )
        except VectorStoreError as e:
            err_msg = str(e)
            if "Duplicate chunk IDs found" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Duplicate document chunks already exist. Set overwrite_duplicates=true to replace them.",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector store operation failed. Please try again later.",
            )

        return DocumentIngestionResponse(
            filename=file.filename,
            total_pages=extracted_doc.total_pages,
            total_chunks=len(chunks),
            added_count=store_result.get("added_count", len(chunks)),
            collection_name=store_result.get("collection_name", vector_store_service.collection_name),
            vector_dimension=store_result.get("vector_dimension", 0),
            status=store_result.get("status", "success"),
        )
    finally:
        # Guarantee cleanup of temporary file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute semantic vector similarity search",
)
def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Executes semantic vector similarity search for a user query.

    (Service integration stub for Phase 2.10 Step 2)
    """
    return SearchResponse(
        query=request.query,
        results_count=0,
        results=[],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Generate grounded answer for a question using RAG",
)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Retrieves context and generates a grounded answer for a user question.

    (Service integration stub for Phase 2.10 Step 2)
    """
    return QueryResponse(
        question=request.question,
        answer="",
        sources=[],
        sufficient_context=False,
        model_name="gemini-2.5-flash",
    )


@router.get(
    "/stats",
    response_model=CollectionStatsResponse,
    summary="Get vector store collection statistics",
)
def get_collection_stats() -> CollectionStatsResponse:
    """
    Returns basic statistics about the vector store collection.

    (Service integration stub for Phase 2.10 Step 2)
    """
    return CollectionStatsResponse(
        collection_name="agentic_rag_collection",
        persist_directory="chroma_db",
        total_records=0,
    )
