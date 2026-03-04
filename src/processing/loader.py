"""
loader.py — Loads documents from the /data directory.
Supports .txt, .pdf, .md, and .csv files via LangChain loaders.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config.settings import DATA_DIR


def load_documents() -> list[Document]:
    """
    Loads all supported documents from the DATA_DIR.

    Supported formats: .txt, .md, .pdf

    Returns:
        A flat list of LangChain Document objects.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    documents: list[Document] = []

    # Load plain text and markdown
    for glob_pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(DATA_DIR),
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        documents.extend(loader.load())

    # Load PDFs file-by-file for better error handling
    for pdf_path in DATA_DIR.rglob("*.pdf"):
        try:
            pdf_loader = PyPDFLoader(str(pdf_path))
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"[Loader] Warning: could not load {pdf_path.name}: {e}")

    print(f"[Loader] Loaded {len(documents)} document(s) from {DATA_DIR}")
    return documents