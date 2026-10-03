import os
import shutil
import tempfile
import unittest

from app.services.embedding_service import EmbeddedChunk
from app.services.vector_store import VectorStoreService, VectorStoreError


class TestVectorStoreService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.persist_path = self.temp_dir.name

        self.service = VectorStoreService(
            persist_directory=self.persist_path,
            collection_name="test_collection",
        )

        self.sample_chunk_1 = EmbeddedChunk(
            chunk_id="doc1.pdf_p1_c1_abc123",
            text="First chunk content for testing vector store.",
            source_filename="doc1.pdf",
            page_number=1,
            chunk_index=1,
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="text-embedding-004",
        )

        self.sample_chunk_2 = EmbeddedChunk(
            chunk_id="doc1.pdf_p2_c2_xyz789",
            text="Second chunk content from page two.",
            source_filename="doc1.pdf",
            page_number=2,
            chunk_index=2,
            embedding=[0.5, 0.6, 0.7, 0.8],
            model_name="text-embedding-004",
        )

    def tearDown(self):
        # Gracefully handle Windows SQLite file locks during temp dir cleanup
        del self.service
        try:
            self.temp_dir.cleanup()
        except Exception:
            shutil.rmtree(self.persist_path, ignore_errors=True)

    def test_add_and_retrieve_chunk(self):
        """Verify inserting embedded chunks and retrieving record by ID."""
        result = self.service.add_embedded_chunks([self.sample_chunk_1, self.sample_chunk_2])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["added_count"], 2)

        # Retrieve record 1
        record = self.service.get_chunk_by_id("doc1.pdf_p1_c1_abc123")
        self.assertIsNotNone(record)
        self.assertEqual(record["chunk_id"], "doc1.pdf_p1_c1_abc123")
        self.assertEqual(record["text"], "First chunk content for testing vector store.")
        self.assertEqual(record["metadata"]["source_filename"], "doc1.pdf")
        self.assertEqual(record["metadata"]["page_number"], 1)
        self.assertEqual(record["metadata"]["chunk_index"], 1)
        self.assertEqual(record["metadata"]["model_name"], "text-embedding-004")
        self.assertEqual([round(x, 4) for x in record["embedding"]], [0.1, 0.2, 0.3, 0.4])

    def test_duplicate_id_handling(self):
        """Verify duplicate chunk ID handling (raise error vs upsert)."""
        self.service.add_embedded_chunks([self.sample_chunk_1])

        # Attempting duplicate insert with overwrite_duplicates=False should raise error
        with self.assertRaises(VectorStoreError) as ctx:
            self.service.add_embedded_chunks([self.sample_chunk_1], overwrite_duplicates=False)
        self.assertIn("Duplicate chunk IDs found", str(ctx.exception))

        # Attempting duplicate insert with overwrite_duplicates=True should succeed
        updated_chunk = EmbeddedChunk(
            chunk_id="doc1.pdf_p1_c1_abc123",
            text="Updated text for duplicate chunk.",
            source_filename="doc1.pdf",
            page_number=1,
            chunk_index=1,
            embedding=[0.9, 0.9, 0.9, 0.9],
            model_name="text-embedding-004",
        )
        res = self.service.add_embedded_chunks([updated_chunk], overwrite_duplicates=True)
        self.assertEqual(res["status"], "success")

        # Verify updated text retrieved
        record = self.service.get_chunk_by_id("doc1.pdf_p1_c1_abc123")
        self.assertEqual(record["text"], "Updated text for duplicate chunk.")

    def test_persistence_across_reopen(self):
        """Verify data persists when collection is reopened from disk."""
        self.service.add_embedded_chunks([self.sample_chunk_1, self.sample_chunk_2])
        info_1 = self.service.get_collection_info()
        self.assertEqual(info_1["total_records"], 2)

        # Re-open database from same directory path
        new_service_instance = VectorStoreService(
            persist_directory=self.persist_path,
            collection_name="test_collection",
        )
        info_2 = new_service_instance.get_collection_info()
        self.assertEqual(info_2["total_records"], 2)

        retrieved = new_service_instance.get_chunk_by_id("doc1.pdf_p2_c2_xyz789")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["text"], "Second chunk content from page two.")

    def test_empty_chunks_handling(self):
        """Verify adding empty chunk list returns no-op status."""
        res = self.service.add_embedded_chunks([])
        self.assertEqual(res["added_count"], 0)
        self.assertEqual(res["status"], "no_op")

    def test_incompatible_vector_dimensions_in_batch(self):
        """Verify batch with mismatched vector dimensions raises error."""
        incompatible_chunk = EmbeddedChunk(
            chunk_id="incompatible_1",
            text="Bad dimensions",
            source_filename="bad.pdf",
            page_number=1,
            chunk_index=1,
            embedding=[0.1, 0.2],  # 2 dims vs sample_chunk_1's 4 dims
            model_name="text-embedding-004",
        )

        with self.assertRaises(VectorStoreError) as ctx:
            self.service.add_embedded_chunks([self.sample_chunk_1, incompatible_chunk])
        self.assertIn("Incompatible vector dimensions", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
