"""
Standalone ingestion script.
Run this once (or whenever you add new resumes) to embed and upload
all PDFs in RESUMES_DIR to your Pinecone index.

Usage:
    python ingest.py
    # or via Makefile:
    make ingest
"""
import logging
import os
import sys

from dotenv import load_dotenv

from src.loaders import load_docs, split_docs
from src.retriever import create_vector_store
from src.utils import authenticate_huggingface, setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        logger.error("❌  Required env var '%s' not set. Check your .env file.", name)
        sys.exit(1)
    return value


def main() -> None:
    pinecone_api_key = _require_env("PINECONE_API_KEY")
    hf_token         = _require_env("HUGGINGFACEHUB_API_TOKEN")
    index_name       = os.getenv("PINECONE_INDEX_NAME", "resumes-index")
    resumes_dir      = os.getenv("RESUMES_DIR", "./resumes")

    logger.info("🔐 Authenticating with HuggingFace...")
    authenticate_huggingface(token=hf_token)

    logger.info("📂 Loading PDFs from '%s'...", resumes_dir)
    docs = load_docs(resumes_dir)

    if not docs:
        logger.error(
            "❌  No PDF files found in '%s'. "
            "Put your resume PDFs there and re-run.",
            resumes_dir,
        )
        sys.exit(1)

    logger.info("✅  Loaded %d document(s).", len(docs))

    logger.info("✂️   Splitting into chunks...")
    chunks = split_docs(docs)
    logger.info("✅  Created %d chunks.", len(chunks))

    logger.info("📤  Uploading to Pinecone index '%s'...", index_name)
    create_vector_store(
        chunks=chunks,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key,
    )

    logger.info(
        "🎉  Done! %d resume(s) → %d chunks → Pinecone index '%s'.",
        len(docs), len(chunks), index_name,
    )


if __name__ == "__main__":
    main()
