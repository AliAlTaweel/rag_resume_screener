"""
Resume Screener — main entry point.
Bootstraps the RAG pipeline and launches the Gradio web UI.
"""
import logging
import os
import sys

import gradio as gr
from dotenv import load_dotenv

from src.loaders import load_docs, split_docs
from src.rag import ask_question, get_llm
from src.retriever import create_vector_store, load_vector_store
from src.utils import authenticate_huggingface, setup_logging

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Exit with a clear message if a required env var is missing."""
    value = os.getenv(name, "").strip()
    if not value:
        logger.error(
            "❌  Required environment variable '%s' is not set. "
            "Copy .env.example → .env and fill in your keys.",
            name,
        )
        sys.exit(1)
    return value


PINECONE_API_KEY = _require_env("PINECONE_API_KEY")
HF_TOKEN        = _require_env("HUGGINGFACEHUB_API_TOKEN")
INDEX_NAME      = os.getenv("PINECONE_INDEX_NAME", "resumes-index")
RESUMES_DIR     = os.getenv("RESUMES_DIR", "./resumes")

# ---------------------------------------------------------------------------
# Auth & LLM
# ---------------------------------------------------------------------------
logger.info("🔐 Authenticating with HuggingFace...")
authenticate_huggingface(token=HF_TOKEN)

logger.info("🤖 Connecting to LLM...")
llm = get_llm(hf_token=HF_TOKEN)

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
logger.info("📂 Loading resumes from '%s'...", RESUMES_DIR)
docs = load_docs(RESUMES_DIR)

if not docs:
    logger.warning(
        "⚠️  No PDF files found in '%s'. "
        "Add resumes to that folder and re-run. "
        "Falling back to existing Pinecone index (queries may return nothing).",
        RESUMES_DIR,
    )
    vector_db = load_vector_store(
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
else:
    logger.info("✂️  Splitting %d document(s) into chunks...", len(docs))
    chunks = split_docs(docs)
    logger.info("📤 Uploading %d chunks to Pinecone index '%s'...", len(chunks), INDEX_NAME)
    vector_db = create_vector_store(
        chunks=chunks,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    logger.info("✅ %d resume(s) → %d chunks uploaded to Pinecone.", len(docs), len(chunks))

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def screen_resume(user_question: str) -> str:
    """Handler wired to the Gradio submit button."""
    if not user_question.strip():
        return "Please enter a question."
    try:
        return ask_question(user_question, vector_db, llm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during question answering.")
        return f"Error: {exc}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Cloud-Powered Resume Screener (Pinecone Edition)")
    question = gr.Textbox(
        label="Question",
        placeholder="Which candidate is best for Python?",
    )
    submit_btn = gr.Button("Analyze", variant="primary")
    output = gr.Textbox(label="AI Report", lines=10)
    submit_btn.click(fn=screen_resume, inputs=question, outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)
