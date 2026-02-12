# ============================================================
#  Resume Screener — Makefile
# ============================================================

.PHONY: help install dev-install ingest run test test-cov \
        test-loader test-embeddings test-retriever test-rag \
        lint format format-check clean

PYTHON   := python
PIP      := pip
PYTEST   := pytest
SRC_DIR  := src
TEST_DIR := tests

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────
help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Installation
# ──────────────────────────────────────────────
install:        ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install:    ## Install production + dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov ruff black

# ──────────────────────────────────────────────
# Running
# ──────────────────────────────────────────────
ingest:         ## Embed & upload resumes/ PDFs to Pinecone (run this first!)
	$(PYTHON) ingest.py

run:            ## Launch the Gradio web UI (ingests if resumes/ has new PDFs)
	$(PYTHON) main.py

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────
test:           ## Run all tests
	$(PYTEST) $(TEST_DIR)

test-cov:       ## Run tests with HTML coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html:htmlcov --cov-report=term-missing
	@echo "Coverage report → htmlcov/index.html"

test-loader:    ## Run only loader tests
	$(PYTEST) $(TEST_DIR)/test_loader.py -v

test-embeddings: ## Run only embedding tests
	$(PYTEST) $(TEST_DIR)/test_embeddings.py -v

test-retriever: ## Run only retriever tests
	$(PYTEST) $(TEST_DIR)/test_retriever.py -v

test-rag:       ## Run only RAG chain tests
	$(PYTEST) $(TEST_DIR)/test_rag_chain.py -v

# ──────────────────────────────────────────────
# Code quality
# ──────────────────────────────────────────────
lint:           ## Lint with ruff
	ruff check $(SRC_DIR) $(TEST_DIR)

format:         ## Auto-format with black
	black $(SRC_DIR) $(TEST_DIR) main.py ingest.py

format-check:   ## Check formatting without making changes
	black --check $(SRC_DIR) $(TEST_DIR) main.py ingest.py

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
clean:          ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	@echo "Cleaned."
