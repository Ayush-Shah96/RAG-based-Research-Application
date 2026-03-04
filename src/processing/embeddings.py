"""
embeddings.py — Builds and persists a FAISS vectorstore from document chunks.
Uses sentence-transformers/all-MiniLM-L6-v2 for local, cost-free embeddings.
"""

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config.settings import EMBEDDING_MODEL, FAISS_INDEX_NAME, VECTORSTORE_DIR


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Returns an initialised HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embeds document chunks and saves the FAISS index to disk.

    Args:
        chunks: Chunked documents from the splitter.

    Returns:
        The in-memory FAISS vectorstore.
    """
    print(f"[Embeddings] Building vectorstore with {len(chunks)} chunk(s)…")
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR / FAISS_INDEX_NAME))
    print(f"[Embeddings] Vectorstore saved to {VECTORSTORE_DIR / FAISS_INDEX_NAME}")
    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Loads a previously saved FAISS index from disk.

    Returns:
        The loaded FAISS vectorstore.

    Raises:
        FileNotFoundError: If the index does not exist yet.
    """
    index_path = VECTORSTORE_DIR / FAISS_INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {index_path}. "
            "Run build_index.ipynb (or embeddings.build_vectorstore) first."
        )

    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[Embeddings] Vectorstore loaded from {index_path}")
    return vectorstore