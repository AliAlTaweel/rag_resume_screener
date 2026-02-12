"""
Tests for src/rag/chain.py
Covers: build_rag_chain, ask_question
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.chain import HR_PROMPT_TEMPLATE, ask_question, build_rag_chain


# ---------------------------------------------------------------------------
# build_rag_chain
# ---------------------------------------------------------------------------

class TestBuildRagChain:
    def test_returns_a_runnable(self, mock_vector_store, mock_llm):
        """build_rag_chain must return something invokable."""
        chain = build_rag_chain(mock_vector_store, mock_llm)
        assert callable(getattr(chain, "invoke", None))

    def test_sets_retriever_with_correct_k(self, mock_vector_store, mock_llm):
        """as_retriever must be called with the provided k value."""
        build_rag_chain(mock_vector_store, mock_llm, k=7)
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 7})

    def test_default_k_is_five(self, mock_vector_store, mock_llm):
        """When k is not supplied the default should be 5."""
        build_rag_chain(mock_vector_store, mock_llm)
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    def test_custom_prompt_template_is_accepted(self, mock_vector_store, mock_llm):
        """A custom prompt template should not raise."""
        custom_tpl = "Custom prompt: {context}\nQ: {input}"
        chain = build_rag_chain(
            mock_vector_store, mock_llm, prompt_template=custom_tpl
        )
        assert chain is not None

    def test_hr_prompt_template_contains_required_slots(self):
        """The default HR prompt must include {context} and {input} placeholders."""
        assert "{context}" in HR_PROMPT_TEMPLATE
        assert "{input}" in HR_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# ask_question
# ---------------------------------------------------------------------------

class TestAskQuestion:
    def test_returns_string(self, mock_vector_store, mock_llm):
        """ask_question should return a plain string."""
        # Patch the entire chain so we don't hit external services
        with patch("src.rag.chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Alice is the best Python developer."
            mock_build.return_value = mock_chain

            result = ask_question("Who knows Python?", mock_vector_store, mock_llm)

        assert isinstance(result, str)

    def test_invokes_chain_with_query(self, mock_vector_store, mock_llm):
        """The query must be forwarded to chain.invoke."""
        query = "Which candidate has Django experience?"
        with patch("src.rag.chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Alice does."
            mock_build.return_value = mock_chain

            ask_question(query, mock_vector_store, mock_llm)

            mock_chain.invoke.assert_called_once_with(query)

    def test_passes_k_to_build_rag_chain(self, mock_vector_store, mock_llm):
        """The k parameter must be forwarded when building the chain."""
        with patch("src.rag.chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Answer"
            mock_build.return_value = mock_chain

            ask_question("Q?", mock_vector_store, mock_llm, k=3)

            _, kwargs = mock_build.call_args
            assert kwargs["k"] == 3

    def test_returns_answer_from_chain(self, mock_vector_store, mock_llm):
        """The exact string returned by the chain should be passed through."""
        expected = "Bob has the strongest data science background."
        with patch("src.rag.chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = expected
            mock_build.return_value = mock_chain

            result = ask_question("Best data scientist?", mock_vector_store, mock_llm)

        assert result == expected

    def test_builds_new_chain_on_each_call(self, mock_vector_store, mock_llm):
        """A new chain should be built for each ask_question call."""
        with patch("src.rag.chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "ok"
            mock_build.return_value = mock_chain

            ask_question("Q1", mock_vector_store, mock_llm)
            ask_question("Q2", mock_vector_store, mock_llm)

        assert mock_build.call_count == 2