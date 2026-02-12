"""
Pinecone vector store creation and retrieval utilities.
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.embeddings.embedding_model import get_embeddings

logger = logging.getLogger(__name__)


def _ensure_index_exists(
    pc: Pinecone,
    index_name: str,
    dimension: int = 384,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """
    Create the Pinecone index if it does not already exist.

    Args:
        pc:         Initialized Pinecone client.
        index_name: Name for the vector index.
        dimension:  Embedding vector dimension (must match embedding model).
        metric:     Distance metric ('cosine', 'euclidean', 'dotproduct').
        cloud:      Cloud provider for ServerlessSpec.
        region:     Region for ServerlessSpec.
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Index '%s' not found. Creating it...", index_name)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info("Index '%s' created.", index_name)
    else:
        logger.info("Index '%s' already exists. Skipping creation.", index_name)


def create_vector_store(
    chunks: List[Document],
    index_name: str,
    pinecone_api_key: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> PineconeVectorStore:
    """
    Embed document chunks and upsert them into a Pinecone vector store.

    Args:
        chunks:               List of document chunks to embed and store.
        index_name:           Name of the Pinecone index.
        pinecone_api_key:     Pinecone API key.
        embedding_model_name: HuggingFace model for embeddings.

    Returns:
        PineconeVectorStore instance populated with the given chunks.
    """
    pc = Pinecone(api_key=pinecone_api_key)
    _ensure_index_exists(pc, index_name)

    embeddings = get_embeddings(embedding_model_name)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    logger.info("Upserted %d chunks into index '%s'.", len(chunks), index_name)
    return vector_store


def load_vector_store(
    index_name: str,
    pinecone_api_key: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> PineconeVectorStore:
    """
    Load an existing Pinecone vector store without upserting new documents.

    Args:
        index_name:           Name of the Pinecone index.
        pinecone_api_key:     Pinecone API key.
        embedding_model_name: HuggingFace model for embeddings.

    Returns:
        PineconeVectorStore instance wrapping the existing index.
    """
    # Initialise client just to validate the key is usable (optional guard)
    Pinecone(api_key=pinecone_api_key)
    embeddings = get_embeddings(embedding_model_name)
    logger.info("Loading existing vector store from index '%s'.", index_name)
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)