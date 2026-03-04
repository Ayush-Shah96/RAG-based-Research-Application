import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS index name
FAISS_INDEX_NAME = "research_index"

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# Splitter settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG settings
TOP_K_RESULTS = 4

# Web scraper settings
SCRAPER_TIMEOUT = 10
SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}