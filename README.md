# 📄 Resume Screener

A cloud-powered, RAG-based resume screening tool built with **LangChain**, **Pinecone**, and **HuggingFace**. Drop PDF resumes into a folder, ask natural-language questions, and get instant AI-generated answers via a Gradio web UI.

---

## Architecture

```
resume_screener/
├── main.py                  # Entry point — bootstraps pipeline & launches UI
├── src/
│   ├── loaders/
│   │   └── document_loader.py   # PDF loading & text splitting
│   ├── embeddings/
│   │   └── embedding_model.py   # HuggingFace sentence-transformer wrapper
│   ├── retriever/
│   │   └── vector_store.py      # Pinecone vector store: create & load
│   ├── rag/
│   │   ├── llm.py               # LLM factory (HuggingFace Inference API)
│   │   └── chain.py             # RAG chain construction & query execution
│   └── utils/
│       └── __init__.py          # Logging setup & HuggingFace auth
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_loader.py           # Tests for document loading & splitting
│   ├── test_embeddings.py       # Tests for the embedding model
│   ├── test_retriever.py        # Tests for vector store creation & loading
│   └── test_rag_chain.py        # Tests for RAG chain & ask_question
├── resumes/                     # Place PDF resumes here (git-ignored)
├── requirements.txt
├── pytest.ini
├── Makefile
└── .env.example
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd resume_screener
python -m venv .venv && source .venv/bin/activate
make install
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env with your real API keys
```

| Variable | Description |
|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)) |
| `PINECONE_API_KEY` | Pinecone API key (from [app.pinecone.io](https://app.pinecone.io)) |
| `PINECONE_INDEX_NAME` | Name for the Pinecone index (default: `resumes-index`) |
| `RESUMES_DIR` | Path to the folder containing PDF resumes (default: `./resumes`) |

### 3. Add resumes

```bash
cp /path/to/*.pdf resumes/
```

### 4. Launch the app

```bash
make run
# → http://127.0.0.1:7860
```

---

## Usage

Open the Gradio UI and type questions such as:

- *"Which candidate has the most Python experience?"*
- *"Who has worked with machine learning frameworks?"*
- *"List all candidates with a Masters degree."*

---

## Development

### Install dev dependencies

```bash
make dev-install
```

### Run tests

```bash
make test            # all tests
make test-loader     # loader tests only
make test-embeddings # embedding tests only
make test-retriever  # retriever tests only
make test-rag        # RAG chain tests only
make test-cov        # tests + HTML coverage report
```

### Lint & format

```bash
make lint            # ruff linter
make format          # black formatter
make format-check    # check without modifying
```

### Clean build artefacts

```bash
make clean
```

---

## Models

| Component | Default model |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| LLM | `meta-llama/Llama-3.2-3B-Instruct` via HF Inference API |

Both can be overridden by passing different `model_name` / `model_id` arguments to `get_embeddings()` and `get_llm()`.

---

## Requirements

- Python 3.11+
- A [HuggingFace](https://huggingface.co) account with API token
- A [Pinecone](https://www.pinecone.io) account with API key

---

## License

MIT