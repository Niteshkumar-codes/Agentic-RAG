from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
from google import genai
from google.genai.types import EmbedContentConfig

from app.services.text_chunker import DocumentChunk


class EmbeddingError(Exception):
    """Custom exception raised during text embedding generation failures."""
    pass


@dataclass
class EmbeddedChunk:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    embedding: List[float]
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "embedding_length": len(self.embedding),
            "model_name": self.model_name,
        }


class EmbeddingService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dimensionality: Optional[int] = 768,
    ):
        """
        Initializes the EmbeddingService.
        Reads GEMINI_API_KEY and EMBEDDING_MODEL from environment variables if not provided.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        self.output_dimensionality = output_dimensionality
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        """Instantiates or returns the cached Google GenAI Client."""
        if self._client is None:
            if not self.api_key:
                raise EmbeddingError(
                    "GEMINI_API_KEY environment variable is missing or empty. Please configure GEMINI_API_KEY."
                )
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize Gemini Client: {str(e)}")
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single text string using Gemini API.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty or whitespace-only text.")

        client = self._get_client()

        try:
            config = None
            if self.output_dimensionality:
                config = EmbedContentConfig(output_dimensionality=self.output_dimensionality)

            response = client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config,
            )

            # Robust response parsing across SDK versions
            if hasattr(response, "embedding") and response.embedding and hasattr(response.embedding, "values"):
                return list(response.embedding.values)
            elif hasattr(response, "embeddings") and response.embeddings:
                first_emb = response.embeddings[0]
                if hasattr(first_emb, "values"):
                    return list(first_emb.values)
                return list(first_emb)
            else:
                raise EmbeddingError("Received unexpected or empty embedding response structure from Gemini API.")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Gemini API embedding request failed for model '{self.model_name}': {str(e)}")

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """
        Generates embeddings for a list of DocumentChunk objects while preserving metadata.
        """
        if not chunks:
            return []

        embedded_chunks: List[EmbeddedChunk] = []

        for chunk in chunks:
            vector = self.embed_text(chunk.text)
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_filename=chunk.source_filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    embedding=vector,
                    model_name=self.model_name,
                )
            )

        return embedded_chunks
