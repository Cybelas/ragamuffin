import unittest

from knowledge_base.chunking import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_has_no_chunks(self) -> None:
        self.assertEqual(chunk_text("   \n"), [])

    def test_chunks_overlap(self) -> None:
        chunks = chunk_text(
            "one two three four five six seven",
            max_words=4,
            overlap_words=2,
        )
        self.assertEqual(
            chunks,
            ["one two three four", "three four five six", "five six seven"],
        )

    def test_overlap_must_be_smaller_than_chunk(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("some words", max_words=4, overlap_words=4)


if __name__ == "__main__":
    unittest.main()
