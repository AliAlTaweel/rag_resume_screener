"""
Tests for src/embeddings/embedding_model.py
Covers: get_embeddings
"""
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import get_embeddings
from src.embeddings.embedding_model import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION


class TestGetEmbeddings:
    def test_returns_huggingface_embeddings_instance(self):
        """get_embeddings should return a HuggingFaceEmbeddings object."""
        from langchain_huggingface import HuggingFaceEmbeddings

        with patch(
            "src.embeddings.embedding_model.HuggingFaceEmbeddings"
        ) as MockEmbeddings:
            mock_instance = MagicMock(spec=HuggingFaceEmbeddings)
            MockEmbeddings.return_value = mock_instance

            result = get_embeddings()

        assert result is mock_instance

    def test_uses_default_model_name(self):
        """Default call should use DEFAULT_EMBEDDING_MODEL."""
        with patch(
            "src.embeddings.embedding_model.HuggingFaceEmbeddings"
        ) as MockEmbeddings:
            MockEmbeddings.return_value = MagicMock()
            get_embeddings()
            MockEmbeddings.assert_called_once_with(model_name=DEFAULT_EMBEDDING_MODEL)

    def test_accepts_custom_model_name(self):
        """Passing a custom model name should forward it to HuggingFaceEmbeddings."""
        custom_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        with patch(
            "src.embeddings.embedding_model.HuggingFaceEmbeddings"
        ) as MockEmbeddings:
            MockEmbeddings.return_value = MagicMock()
            get_embeddings(model_name=custom_model)
            MockEmbeddings.assert_called_once_with(model_name=custom_model)

    def test_default_model_constant_is_correct(self):
        """Sanity-check the default model constant value."""
        assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dimension_constant_is_correct(self):
        """Sanity-check the dimension constant for the default model."""
        assert EMBEDDING_DIMENSION == 384

    def test_get_embeddings_called_once_per_invocation(self):
        """Each call to get_embeddings should instantiate the model exactly once."""
        with patch(
            "src.embeddings.embedding_model.HuggingFaceEmbeddings"
        ) as MockEmbeddings:
            MockEmbeddings.return_value = MagicMock()
            get_embeddings()
            get_embeddings()
        assert MockEmbeddings.call_count == 2