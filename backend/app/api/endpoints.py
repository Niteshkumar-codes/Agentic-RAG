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
    SearchResultItem,
    SourceReferenceItem,
)
from app.services.document_parser import DocumentExtractionError, extract_text_from_file
from app.services.text_chunker import ChunkingError, chunk_document
from app.services.embedding_service import EmbeddingError, EmbeddingService
from app.services.vector_store import VectorStoreError, VectorStoreService
from app.services.retrieval_service import (
    RetrievalError,
    RetrievalService,
    SearchResult,
)
from app.services.answer_generation_service import (
    AnswerGenerationError,
    AnswerGenerationService,
    AnswerResult,
    SourceReference,
)

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])

# Shared service instances
_embedding_service: Optional[EmbeddingService] = None
_vector_store_service: Optional[VectorStoreService] = None
_retrieval_service: Optional[RetrievalService] = None
_answer_generation_service: Optional[AnswerGenerationService] = None


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


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(
            embedding_service=get_embedding_service(),
            vector_store_service=get_vector_store_service(),
        )
    return _retrieval_service


def get_answer_generation_service() -> AnswerGenerationService:
    global _answer_generation_service
    if _answer_generation_service is None:
        _answer_generation_service = AnswerGenerationService()
    return _answer_generation_service


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
    """
    retrieval_service = get_retrieval_service()

    try:
        results = retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
        )
    except RetrievalError as e:
        err_msg = str(e)
        if (
            "Model mismatch" in err_msg
            or "Query cannot be empty" in err_msg
            or "top_k must be" in err_msg
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid search parameters or embedding model mismatch.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic vector search failed. Please try again later.",
        )

    mapped_results = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            text=r.text,
            source_filename=r.source_filename,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            model_name=r.model_name,
            distance=r.distance,
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        results_count=len(mapped_results),
        results=mapped_results,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Generate grounded answer for a question using RAG",
)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Retrieves context and generates a grounded answer for a user question.
    """
    retrieval_service = get_retrieval_service()
    answer_gen_service = get_answer_generation_service()

    # 1. Retrieve vector search context
    try:
        results = retrieval_service.search(
            query=request.question,
            top_k=request.top_k,
        )
    except RetrievalError as e:
        err_msg = str(e)
        if (
            "Model mismatch" in err_msg
            or "Query cannot be empty" in err_msg
            or "top_k must be" in err_msg
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid search parameters or embedding model mismatch.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Context retrieval failed. Please try again later.",
        )

    # 2. Generate grounded answer
    try:
        answer_result = answer_gen_service.generate_answer(
            question=request.question,
            search_results=results,
        )
    except AnswerGenerationError as e:
        err_msg = str(e)
        if "Question cannot be empty" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid question parameter.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed. Please try again later.",
        )

    # 3. Map SourceReference objects into SourceReferenceItem
    mapped_sources = [
        SourceReferenceItem(
            source_label=s.source_label,
            source_filename=s.source_filename,
            page_number=s.page_number,
            chunk_id=s.chunk_id,
        )
        for s in answer_result.sources
    ]

    return QueryResponse(
        question=request.question,
        answer=answer_result.answer,
        sources=mapped_sources,
        sufficient_context=answer_result.sufficient_context,
        model_name=answer_result.model_name,
    )


@router.get(
    "/stats",
    response_model=CollectionStatsResponse,
    summary="Get vector store collection statistics",
)
def get_collection_stats() -> CollectionStatsResponse:
    """
    Returns basic statistics about the vector store collection.
    """
    vector_store_service = get_vector_store_service()

    try:
        info = vector_store_service.get_collection_info()
    except VectorStoreError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vector store collection statistics.",
        )

    return CollectionStatsResponse(
        collection_name=info.get("collection_name", vector_store_service.collection_name),
        persist_directory=info.get("persist_directory", vector_store_service.persist_directory),
        total_records=info.get("total_records", 0),
    )
