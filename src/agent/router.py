"""
router.py — Decides whether a query should be answered via
local RAG or live web scraping.
"""

WEB_KEYWORDS = {"latest", "recent", "today", "current", "now", "new", "breaking"}


def route_query(query: str) -> str:
    """
    Returns 'web' if the query contains any web-trigger keywords,
    otherwise returns 'local'.

    Args:
        query: The user's research question.

    Returns:
        'web' or 'local'
    """
    lowered = query.lower()
    for keyword in WEB_KEYWORDS:
        if keyword in lowered:
            return "web"
    return "local"