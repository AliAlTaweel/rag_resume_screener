"""
LLM configuration and factory for the resume screener.
"""
import logging
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def get_llm(
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    hf_token: str | None = None,
) -> ChatHuggingFace:
    """
    Build and return a ChatHuggingFace LLM connected to the HF Inference API.

    Args:
        model_id:       HuggingFace model repository identifier.
        max_new_tokens: Maximum tokens the model may generate per call.
        temperature:    Sampling temperature (lower = more deterministic).
        hf_token:       HuggingFace API token. Falls back to the
                        HUGGINGFACEHUB_API_TOKEN environment variable.

    Returns:
        A ChatHuggingFace instance ready to use in a LangChain chain.
    """
    token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "No HuggingFace token provided. "
            "Set HUGGINGFACEHUB_API_TOKEN or pass hf_token explicitly."
        )

    logger.info("Connecting to HuggingFace Inference API for model: %s", model_id)

    raw_llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
    )
    llm = ChatHuggingFace(llm=raw_llm)
    logger.info("LLM connection ready.")
    return llm