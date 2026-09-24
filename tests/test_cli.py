import unittest
from unittest.mock import Mock, patch
from knowledge_base.cli import build_parser, retrieve, run_ask, run_search
from argparse import Namespace

class CliParserTests(unittest.TestCase):
    def test_ask_accepts_minimum_score(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["ask", "What is RAG?", "--min-score", "0.2"]
        )

        self.assertEqual(args.min_score, 0.2)

    def test_search_accepts_minimum_score(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["search", "What is RAG?", "--min-score", "0.2"]
        )

        self.assertEqual(args.min_score, 0.2)

class RetrievalTests(unittest.TestCase):
    @patch("knowledge_base.cli.OpenAIProvider")
    def test_retrieve_passes_minimum_score_to_store(
        self,
        provider_class: Mock,
    ) -> None:
        provider = provider_class.return_value
        provider.embedding_model = "test-model"
        provider.embed.return_value = [[1.0, 0.0]]

        store = Mock()
        store.search.return_value = []

        retrieve(
            "What is RAG?",
            2,
            store,
            min_score=0.5,
        )

        store.search.assert_called_once_with(
            [1.0, 0.0],
            embedding_model="test-model",
            top_k=2,
            min_score=0.5,
        )

class CommandHandlerTests(unittest.TestCase):
    @patch("knowledge_base.cli.retrieve")
    def test_run_search_passes_minimum_score_to_retrieve(
        self,
        retrieve_mock: Mock,
    ) -> None:
        retrieve_mock.return_value = (Mock(), [])

        args = Namespace(
            query="What is RAG?",
            top_k=2,
            min_score=0.5,
        )
        store = Mock()

        run_search(args, store)

        retrieve_mock.assert_called_once_with(
            "What is RAG?",
            2,
            store,
            min_score=0.5,
        )
    @patch("knowledge_base.cli.retrieve")
    def test_run_ask_passes_minimum_score_to_retrieve(
        self,
        retrieve_mock: Mock,
    ) -> None:
        retrieve_mock.return_value = (Mock(), [])

        args = Namespace(
            question="What is RAG?",
            top_k=2,
            min_score=0.5,
        )
        store = Mock()

        run_ask(args, store)

        retrieve_mock.assert_called_once_with(
            "What is RAG?",
            2,
            store,
            min_score=0.5,
        )
    def test_run_ask_skips_generation_when_threshold_removes_results(
        self,
    ) -> None:
        with patch("knowledge_base.cli.retrieve") as retrieve_mock:
            with patch("builtins.print") as print_mock:
                provider = Mock()
                retrieve_mock.return_value = (provider, [])

                args = Namespace(
                    question="How do I replace a lightbulb?",
                    top_k=2,
                    min_score=0.2,
                )
                store = Mock()

                exit_code = run_ask(args, store)

        self.assertEqual(exit_code, 1)
        provider.answer.assert_not_called()
        print_mock.assert_called_once_with(
            "No chunks met --min-score 0.2000. "
            "Check the index or try a lower threshold."
        )
    def test_run_search_receives_no_results_and_min_score(
        self,
    ) -> None:

        with patch("knowledge_base.cli.retrieve") as retrieve_mock:
            with patch("builtins.print") as print_mock:
                provider = Mock()
                retrieve_mock.return_value = (provider, [])
                args = Namespace(
                    query="How do I replace a lightbulb?",
                    top_k=2,
                    min_score=0.2,
                )
                store = Mock()
                exit_code = run_search(args, store)
        self.assertEqual(exit_code, 1)
        print_mock.assert_called_once_with(
            "No chunks met --min-score 0.2000. "
            "Check the index or try a lower threshold."
        )


    def test_run_ask_generates_answer_when_results_exist(self) -> None:
        with patch("knowledge_base.cli.retrieve") as retrieve_mock:
            with patch("builtins.print") as print_mock:
                provider = Mock()
                provider.answer.return_value = "Grounded answer [S1]"

                result = Mock(
                    source="notes.md",
                    chunk_index=0,
                    score=0.8,
                )
                results = [result]
                retrieve_mock.return_value = (provider, results)

                args = Namespace(
                    question="What is RAG?",
                    top_k=2,
                    min_score=0.2,
                )
                store = Mock()

                exit_code = run_ask(args, store)

        self.assertEqual(exit_code, 0)
        provider.answer.assert_called_once_with(
            "What is RAG?",
            results,
        )
        print_mock.assert_any_call("Grounded answer [S1]")


if __name__ == "__main__":
    unittest.main()