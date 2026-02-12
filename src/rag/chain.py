"""
RAG chain construction and query execution.
"""
import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

logger = logging.getLogger(__name__)

HR_PROMPT_TEMPLATE = (
    "You are an expert HR assistant. "
    "Answer based ONLY on the provided resumes:\n"
    "Context: {context}\n\n"
    "Question: {input}"
)


def build_rag_chain(
    vector_store: PineconeVectorStore,
    llm: Any,
    k: int = 5,
    prompt_template: str = HR_PROMPT_TEMPLATE,
) -> Any:
    """
    Assemble a retrieval-augmented generation (RAG) chain.

    Args:
        vector_store:    Populated PineconeVectorStore used as the retriever.
        llm:             LangChain-compatible chat LLM.
        k:               Number of documents to retrieve per query.
        prompt_template: Jinja-style template with {context} and {input} slots.

    Returns:
        A LangChain runnable chain that accepts a query string and returns a
        plain-text answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain built with k=%d.", k)
    return chain


def ask_question(
    query: str,
    vector_store: PineconeVectorStore,
    llm: Any,
    k: int = 5,
) -> str:
    """
    Run a single question through the RAG chain and return the answer.

    Args:
        query:        Natural-language question about the resumes.
        vector_store: Populated PineconeVectorStore.
        llm:          LangChain-compatible chat LLM.
        k:            Number of documents to retrieve.

    Returns:
        Plain-text answer string produced by the LLM.
    """
    chain = build_rag_chain(vector_store, llm, k=k)
    logger.info("Invoking RAG chain with query: %s", query)
    answer = chain.invoke(query)
    return answer