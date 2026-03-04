"""
web_scraper.py — Fetches and extracts clean text from a URL.
Also provides a helper to build a Google search URL from a query string.
"""

import urllib.parse

import requests
from bs4 import BeautifulSoup

from src.config.settings import SCRAPER_HEADERS, SCRAPER_TIMEOUT


def scrape_url(url: str) -> str:
    """
    Fetches a URL and returns the visible text content.

    Args:
        url: The page to scrape.

    Returns:
        Extracted text as a plain string, or an error message.
    """
    try:
        response = requests.get(url, headers=SCRAPER_HEADERS, timeout=SCRAPER_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"[Scraper error] Could not fetch {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Extract and clean text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def build_google_search_url(query: str) -> str:
    """
    Returns a Google search URL for the given query string.

    Args:
        query: The search query.

    Returns:
        A formatted Google search URL.
    """
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.google.com/search?q={encoded}"


def scrape_query(query: str) -> str:
    """
    Convenience wrapper: builds a Google search URL from `query`
    and returns the scraped text.

    Args:
        query: The research question.

    Returns:
        Scraped text from the Google search results page.
    """
    url = build_google_search_url(query)
    return scrape_url(url)