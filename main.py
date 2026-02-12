"""
Resume Screener — main entry point.
Bootstraps the RAG pipeline and launches the Gradio web UI.
"""
import logging
import os

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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resumes-index")
RESUMES_DIR = os.getenv("RESUMES_DIR", "./resumes")

# ---------------------------------------------------------------------------
# Auth & LLM
# ---------------------------------------------------------------------------
authenticate_huggingface()
llm = get_llm()

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
logger.info("Processing resumes from '%s'...", RESUMES_DIR)
docs = load_docs(RESUMES_DIR)

if docs:
    chunks = split_docs(docs)
    vector_db = create_vector_store(
        chunks=chunks,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    logger.info("Loaded %d resumes into Pinecone.", len(docs))
else:
    logger.warning("No resumes found. Loading existing Pinecone index.")
    vector_db = load_vector_store(
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def screen_resume(user_question: str) -> str:
    """Handler wired to the Gradio submit button."""
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