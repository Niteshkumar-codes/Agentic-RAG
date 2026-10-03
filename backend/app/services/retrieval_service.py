from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.services.embedding_service import EmbeddingService, EmbeddingError
from app.services.vector_store import VectorStoreService, VectorStoreError


class RetrievalError(Exception):
    """Custom exception raised during semantic retrieval failures."""
    pass


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    model_name: str
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "model_name": self.model_name,
            "distance": self.distance,
        }


class RetrievalService:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
    ):
        """
        Initializes RetrievalService with an EmbeddingService and VectorStoreService instance.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[SearchResult]:
        """
        Executes semantic vector similarity search for a user query.

        :param query: User natural language search query.
        :param top_k: Maximum number of relevant chunks to retrieve (default: 4).
        :return: List of SearchResult objects sorted by vector similarity distance.
        :raises RetrievalError: For invalid inputs, empty query, or model mismatch.
        """
        # 1. Input Validation
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise RetrievalError(f"top_k must be a positive integer > 0, got {top_k}.")

        # 2. Check for empty collection
        try:
            info = self.vector_store_service.get_collection_info()
        except VectorStoreError as e:
            raise RetrievalError(f"Failed to inspect collection before search: {str(e)}")

        total_records = info.get("total_records", 0)
        if total_records == 0:
            return []

        # 3. Generate query embedding using the configured model
        try:
            query_vector = self.embedding_service.embed_text(query.strip())
        except EmbeddingError as e:
            raise RetrievalError(f"Failed to generate query embedding: {str(e)}")

        # 4. Perform vector similarity search in ChromaDB
        collection = self.vector_store_service.collection
        n_results = min(top_k, total_records)

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise RetrievalError(f"ChromaDB vector search query failed: {str(e)}")

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results.get("documents") else []
        metas_list = results["metadatas"][0] if results.get("metadatas") else []
        dists_list = results["distances"][0] if results.get("distances") else []

        search_results: List[SearchResult] = []

        # 5. Parse and validate results
        for idx in range(len(ids_list)):
            c_id = ids_list[idx]
            text = docs_list[idx] if idx < len(docs_list) else ""
            meta = metas_list[idx] if idx < len(metas_list) else {}
            dist = dists_list[idx] if idx < len(dists_list) else 0.0

            # Enforce embedding model consistency
            record_model = meta.get("model_name", "")
            if record_model and record_model != self.embedding_service.model_name:
                raise RetrievalError(
                    f"Model mismatch error: stored chunk '{c_id}' uses model '{record_model}', "
                    f"but query retrieval uses model '{self.embedding_service.model_name}'."
                )

            search_results.append(
                SearchResult(
                    chunk_id=c_id,
                    text=text,
                    source_filename=meta.get("source_filename", "unknown"),
                    page_number=meta.get("page_number", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    model_name=record_model or self.embedding_service.model_name,
                    distance=float(dist),
                )
            )

        return search_results
