"""
Shared pytest fixtures used across all test modules.
"""
import os
import tempfile

import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Document / chunk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """Two minimal Document objects mimicking loaded PDF pages."""
    return [
        Document(
            page_content="Alice is a Python developer with 5 years experience in Django and FastAPI.",
            metadata={"source": "alice_resume.pdf", "page": 0},
        ),
        Document(
            page_content="Bob specialises in data science, using pandas, numpy and scikit-learn daily.",
            metadata={"source": "bob_resume.pdf", "page": 0},
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents):
    """Pre-split chunks derived from sample_documents."""
    from src.loaders import split_docs
    return split_docs(sample_documents, chunk_size=100, chunk_overlap=10)


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_pdf_dir(tmp_path):
    """A temporary directory that contains one stub PDF file."""
    pdf_path = tmp_path / "stub_resume.pdf"
    # Minimal valid PDF so PyPDFLoader does not crash
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )
    return str(tmp_path)


@pytest.fixture
def empty_dir(tmp_path):
    """An empty temporary directory (no PDFs)."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Mock LLM fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """A MagicMock that quacks like a LangChain chat LLM."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mocked LLM answer.")
    return llm


# ---------------------------------------------------------------------------
# Mock vector store fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(sample_documents):
    """A MagicMock PineconeVectorStore with a working retriever."""
    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = sample_documents
    store.as_retriever.return_value = retriever
    return store