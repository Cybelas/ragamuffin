from pathlib import Path
import tempfile
import unittest

from knowledge_base.store import KnowledgeStore, cosine_similarity


class CosineSimilarityTests(unittest.TestCase):
    def test_same_direction_scores_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2], [2, 4]), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "test.db"
        self.store = KnowledgeStore(database_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary_directory.cleanup()

    def test_search_ranks_closest_vector_first(self) -> None:
        self.store.replace_document(
            source="notes.md",
            content_hash="abc",
            embedding_model="test-model",
            chunks=["about databases", "about baking"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        results = self.store.search(
            [0.9, 0.1], embedding_model="test-model", top_k=2
        )

        self.assertEqual(results[0].text, "about databases")
        self.assertGreater(results[0].score, results[1].score)

    def test_replacing_document_removes_old_chunks(self) -> None:
        self.store.replace_document(
            source="notes.md",
            content_hash="old",
            embedding_model="test-model",
            chunks=["old one", "old two"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        self.store.replace_document(
            source="notes.md",
            content_hash="new",
            embedding_model="test-model",
            chunks=["new only"],
            embeddings=[[1.0, 0.0]],
        )

        self.assertEqual(self.store.stats(), (1, 1))
        self.assertFalse(
            self.store.document_is_current("notes.md", "old", "test-model")
        )
        self.assertTrue(
            self.store.document_is_current("notes.md", "new", "test-model")
        )


if __name__ == "__main__":
    unittest.main()
