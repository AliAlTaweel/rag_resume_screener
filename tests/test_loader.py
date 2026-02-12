"""
Tests for src/loaders/document_loader.py
Covers: load_docs, split_docs
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.loaders import load_docs, split_docs


# ---------------------------------------------------------------------------
# load_docs
# ---------------------------------------------------------------------------

class TestLoadDocs:
    def test_creates_directory_if_missing(self, tmp_path):
        """load_docs should create the directory when it does not exist."""
        target = str(tmp_path / "new_dir")
        assert not os.path.exists(target)
        load_docs(target)
        assert os.path.exists(target)

    def test_returns_empty_list_for_empty_dir(self, empty_dir):
        """An empty directory should yield zero documents."""
        docs = load_docs(empty_dir)
        assert docs == []

    def test_returns_documents_for_pdf_dir(self, tmp_pdf_dir):
        """A directory with a valid PDF should return at least one Document."""
        # The stub PDF is empty content-wise; PyPDFLoader may return 0 pages.
        # We mock DirectoryLoader to return predictable results.
        fake_doc = Document(page_content="Alice resume", metadata={"source": "test.pdf"})
        with patch("src.loaders.document_loader.DirectoryLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.load.return_value = [fake_doc]
            docs = load_docs(tmp_pdf_dir)

        assert len(docs) == 1
        assert docs[0].page_content == "Alice resume"

    def test_returns_list_of_document_objects(self, tmp_pdf_dir):
        """All returned items must be LangChain Document instances."""
        fake_docs = [
            Document(page_content="Doc A", metadata={}),
            Document(page_content="Doc B", metadata={}),
        ]
        with patch("src.loaders.document_loader.DirectoryLoader") as MockLoader:
            MockLoader.return_value.load.return_value = fake_docs
            docs = load_docs(tmp_pdf_dir)

        for doc in docs:
            assert isinstance(doc, Document)

    def test_load_docs_existing_dir_is_not_recreated(self, tmp_path):
        """No OSError should be raised when the directory already exists."""
        existing = str(tmp_path)
        with patch("src.loaders.document_loader.DirectoryLoader") as MockLoader:
            MockLoader.return_value.load.return_value = []
            load_docs(existing)  # Should not raise


# ---------------------------------------------------------------------------
# split_docs
# ---------------------------------------------------------------------------

class TestSplitDocs:
    def test_splits_into_multiple_chunks(self, sample_documents):
        """Long documents should produce more than one chunk."""
        # Use very small chunk size to force splitting
        chunks = split_docs(sample_documents, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= len(sample_documents)

    def test_chunks_are_document_objects(self, sample_documents):
        """Every chunk must be a LangChain Document."""
        chunks = split_docs(sample_documents)
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_chunk_size_is_respected(self, sample_documents):
        """No chunk's content should exceed the configured chunk_size."""
        chunk_size = 50
        chunks = split_docs(sample_documents, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_empty_input_returns_empty_list(self):
        """Passing an empty list should return an empty list."""
        chunks = split_docs([])
        assert chunks == []

    def test_metadata_is_preserved(self, sample_documents):
        """Source metadata from the original document should be present in chunks."""
        chunks = split_docs(sample_documents, chunk_size=20, chunk_overlap=0)
        sources = {c.metadata.get("source") for c in chunks}
        # Both source filenames should appear in at least one chunk
        assert "alice_resume.pdf" in sources
        assert "bob_resume.pdf" in sources

    def test_default_parameters_produce_chunks(self, sample_documents):
        """Calling split_docs with only documents should still work."""
        chunks = split_docs(sample_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0