"""
Shared utility helpers.
"""
import logging
import os

from huggingface_hub import login


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a standard format."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=level,
    )


def authenticate_huggingface(token: str | None = None) -> None:
    """
    Log in to HuggingFace Hub.

    Args:
        token: HF token. Falls back to HUGGINGFACEHUB_API_TOKEN env var.

    Raises:
        ValueError: If no token can be found.
    """
    resolved_token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not resolved_token:
        raise ValueError(
            "HuggingFace token not found. "
            "Set HUGGINGFACEHUB_API_TOKEN or pass token explicitly."
        )
    login(token=resolved_token)
    logging.getLogger(__name__).info("Authenticated with HuggingFace Hub.")
