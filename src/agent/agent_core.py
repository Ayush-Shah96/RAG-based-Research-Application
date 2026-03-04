"""
agent_core.py — Orchestrates routing, retrieval, and generation.
"""

from openai import OpenAI

from src.agent.rag_engine import run_rag
from src.agent.router import route_query
from src.agent.web_scraper import scrape_query
from src.config.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)

_client = OpenAI(api_key=OPENAI_API_KEY)

_SUMMARISE_PROMPT = """\
You are a research assistant. A user asked the following question:

"{query}"

Below is raw text scraped from a web search. Summarise the most relevant
information to answer the question. Be concise and factual.

--- SCRAPED CONTENT ---
{content}
--- END ---

Summary:"""


def _summarise_web_content(query: str, content: str) -> str:
    """Uses the LLM to distil scraped web content into a focused answer."""
    # Truncate very long pages to avoid token limits
    truncated = content[:6000]
    prompt = _SUMMARISE_PROMPT.format(query=query, content=truncated)
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


class ResearchAgent:
    """
    High-level agent that routes queries and returns answers.

    Usage:
        agent = ResearchAgent()
        answer = agent.run("What is quantum entanglement?")
    """

    def run(self, query: str) -> str:
        """
        Processes a research question end-to-end.

        Args:
            query: The user's research question.

        Returns:
            A string answer.
        """
        route = route_query(query)
        print(f"[Agent] Route: {route.upper()} — Query: \"{query}\"")

        if route == "web":
            print("[Agent] Scraping web for live information…")
            raw_content = scrape_query(query)
            if raw_content.startswith("[Scraper error]"):
                return raw_content
            return _summarise_web_content(query, raw_content)

        # route == "local"
        print("[Agent] Querying local RAG knowledge base…")
        return run_rag(query)