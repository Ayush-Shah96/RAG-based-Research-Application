# RAG Research Agent

A lightweight research agent that answers questions from your own documents using Retrieval-Augmented Generation (RAG), and falls back to live web scraping for queries about recent or current events.

---

## How It Works

```
User Query
    │
    ▼
 router.py ──── contains "latest/recent/today/current"? ──► web_scraper.py ──► LLM Summary
    │                                                                               │
    └─── otherwise ──────────────────────────────────────► rag_engine.py ─────────┘
                                                              (FAISS + LLM)
                                                                    │
                                                              Answer returned
```

---

## Project Structure

```
rag-research-agent/
├── data/                     ← Drop your .txt / .pdf / .md files here
├── vectorstore/              ← Auto-generated FAISS index (git-ignored)
├── src/
│   ├── main.py               ← CLI entry point
│   ├── agent/
│   │   ├── router.py         ← Keyword-based query routing
│   │   ├── rag_engine.py     ← RAG retrieval + LLM generation
│   │   ├── web_scraper.py    ← requests + BeautifulSoup scraper
│   │   └── agent_core.py     ← Orchestration layer
│   ├── processing/
│   │   ├── loader.py         ← LangChain DirectoryLoader
│   │   ├── splitter.py       ← RecursiveCharacterTextSplitter
│   │   └── embeddings.py     ← FAISS build + load (MiniLM-L6-v2)
│   └── config/
│       └── settings.py       ← All tuneable constants
├── notebooks/
│   └── build_index.ipynb     ← One-time index builder
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd rag-research-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

---

## Adding Your Documents

Drop any `.txt`, `.md`, or `.pdf` files into the `data/` directory:

```
data/
├── my_research_paper.pdf
├── notes.txt
└── summary.md
```

---

## Building the Vector Index

Run the notebook **once** (or after adding new documents):

```bash
jupyter notebook notebooks/build_index.ipynb
```

Or run the steps directly in Python:

```python
from src.processing.loader import load_documents
from src.processing.splitter import split_documents
from src.processing.embeddings import build_vectorstore

docs = load_documents()
chunks = split_documents(docs)
build_vectorstore(chunks)
```

The FAISS index will be saved to `vectorstore/research_index/`.

---

## Running the Agent

```bash
python -m src.main
```

```
============================================================
  RAG Research Agent
  Type 'exit' or 'quit' to stop.
============================================================

Ask your research question: What is quantum entanglement?

[Agent] Route: LOCAL — Query: "What is quantum entanglement?"
[Agent] Querying local RAG knowledge base…

------------------------------------------------------------
Quantum entanglement is a phenomenon where two particles ...
------------------------------------------------------------

Ask your research question: What are the latest AI developments today?

[Agent] Route: WEB — Query: "What are the latest AI developments today?"
[Agent] Scraping web for live information…

------------------------------------------------------------
Recent AI developments include ...
------------------------------------------------------------
```

---

## Configuration

All settings live in `src/config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |

---

## Notes

- The web scraper hits Google Search directly. For production use, consider a proper search API (SerpAPI, Brave Search, Tavily) for more reliable results.
- The FAISS index is stored locally; add `vectorstore/` to `.gitignore` for large datasets.
- No external vector database is required — everything runs locally.