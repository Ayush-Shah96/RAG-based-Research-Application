"""
rag_engine.py — Retrieval-Augmented Generation engine.
Loads the FAISS vectorstore, retrieves relevant context, and
generates an answer via OpenAI's chat API.
"""

from openai import OpenAI

from src.config.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    TOP_K_RESULTS,
)
from src.processing.embeddings import load_vectorstore

_client = OpenAI(api_key=OPENAI_API_KEY)


def _build_prompt(query: str, context: str) -> str:
    return f"""You are a research assistant. Use ONLY the context below to answer the
question. If the context does not contain enough information, say so clearly.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""


def run_rag(query: str) -> str:
    """
    Retrieves the most relevant document chunks for `query` and
    generates a grounded answer using the LLM.

    Args:
        query: The user's research question.

    Returns:
        LLM-generated answer string.
    """
    # 1. Load vectorstore and retrieve top-k chunks
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as e:
        return f"[RAG Error] {e}"

    docs = vectorstore.similarity_search(query, k=TOP_K_RESULTS)
    if not docs:
        return "No relevant documents found in the local knowledge base."

    # 2. Concatenate retrieved context
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    # 3. Call LLM
    prompt = _build_prompt(query, context)
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()