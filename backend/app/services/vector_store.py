import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb

from app.services.embedding_service import EmbeddedChunk


class VectorStoreError(Exception):
    """Custom exception raised during ChromaDB vector store operations."""
    pass


class VectorStoreService:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initializes persistent ChromaDB client and gets or creates the target collection.
        
        :param persist_directory: Path to store persistent database files.
        :param collection_name: Name of the vector collection.
        """
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "chroma_db"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "agentic_rag_collection"
        )

        try:
            # Ensure directory path exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB store: {str(e)}")

    def add_embedded_chunks(
        self,
        chunks: List[EmbeddedChunk],
        overwrite_duplicates: bool = False,
    ) -> Dict[str, Any]:
        """
        Stores a batch of EmbeddedChunk objects into ChromaDB.

        :param chunks: List of EmbeddedChunk objects.
        :param overwrite_duplicates: If True, uses upsert to overwrite existing chunk IDs.
                                     If False, raises VectorStoreError if duplicate IDs exist.
        :return: Summary dictionary of operation results.
        """
        if not chunks:
            return {"added_count": 0, "status": "no_op", "message": "No chunks provided"}

        # 1. Validate vector dimension consistency across batch
        sample_dim = len(chunks[0].embedding)
        sample_model = chunks[0].model_name

        for idx, chunk in enumerate(chunks):
            if not chunk.embedding:
                raise VectorStoreError(f"Chunk '{chunk.chunk_id}' has empty embedding vector.")
            if len(chunk.embedding) != sample_dim:
                raise VectorStoreError(
                    f"Incompatible vector dimensions in batch: chunk '{chunk.chunk_id}' "
                    f"has dimension {len(chunk.embedding)}, expected {sample_dim}."
                )

        # 2. Check for duplicate IDs in existing collection if overwrite_duplicates is False
        incoming_ids = [c.chunk_id for c in chunks]
        existing_records = self.collection.get(ids=incoming_ids)
        existing_ids = set(existing_records.get("ids", [])) if existing_records else set()

        if existing_ids and not overwrite_duplicates:
            raise VectorStoreError(
                f"Duplicate chunk IDs found in collection: {list(existing_ids)}. "
                "Set overwrite_duplicates=True to overwrite existing records."
            )

        # 3. Prepare ChromaDB payloads
        ids = incoming_ids
        embeddings = [c.embedding for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "model_name": c.model_name,
            }
            for c in chunks
        ]

        # 4. Perform Insert / Upsert
        try:
            if overwrite_duplicates:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB insert failed: {str(e)}")

        return {
            "added_count": len(chunks),
            "collection_name": self.collection_name,
            "status": "success",
            "model_name": sample_model,
            "vector_dimension": sample_dim,
        }

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single record by chunk_id from ChromaDB for verification.
        """
        if not chunk_id:
            return None

        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if not result or not result.get("ids"):
                return None

            raw_emb = result["embeddings"][0] if result.get("embeddings") is not None else []
            if hasattr(raw_emb, "tolist"):
                emb_list = raw_emb.tolist()
            else:
                emb_list = list(raw_emb)

            return {
                "chunk_id": result["ids"][0],
                "text": result["documents"][0] if result.get("documents") else "",
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                "embedding": emb_list,
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to retrieve chunk '{chunk_id}': {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns basic statistics about the current collection.
        """
        try:
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_records": self.collection.count(),
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to fetch collection info: {str(e)}")
