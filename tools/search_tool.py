"""
tools/search_tool.py
--------------------
Two search utilities used by the Novelty and Fact-check agents:
  1. Semantic Scholar API  — finds related academic papers
  2. DuckDuckGo search     — general web fact-checking (free, no key)
  3. SerpAPI               — optional upgrade for better web search
"""

import os
import time
import requests
from typing import Optional
from ddgs import DDGS


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def search_related_papers(query: str, limit: int = 8) -> list[dict]:
    """
    Search Semantic Scholar for papers related to the given query.
    Returns a list of dicts: {title, authors, year, abstract, url}
    """
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,externalIds,openAccessPdf",
    }

    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params=params,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for paper in data.get("data", []):
            results.append({
                "title":    paper.get("title", ""),
                "authors":  ", ".join(a["name"] for a in paper.get("authors", [])[:3]),
                "year":     paper.get("year", ""),
                "abstract": (paper.get("abstract") or "")[:300],
                "url":      f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
            })
        return results

    except Exception as e:
        print(f"[Search] Semantic Scholar error: {e}")
        return []


def get_paper_citations(title: str) -> int:
    """
    Get approximate citation count for a paper by title.
    Useful for estimating novelty context.
    """
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY

    try:
        params = {"query": title, "limit": 1, "fields": "citationCount"}
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params=params,
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        papers = data.get("data", [])
        if papers:
            return papers[0].get("citationCount", 0)
    except Exception as e:
        print(f"[Search] Citation count error: {e}")
    return 0


# ── DuckDuckGo web search ─────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Free DuckDuckGo search. Returns list of {title, url, snippet}.
    Used by Fact-check agent to verify claims.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", "")[:400],
            }
            for r in results
        ]
    except Exception as e:
        print(f"[Search] DuckDuckGo error: {e}")
        return []


def fact_check_claim(claim: str) -> dict:
    """
    Convenience wrapper: search for a specific factual claim.
    Returns {claim, results, verdict_hint}
    """
    results = web_search(f"verify fact: {claim}", max_results=4)

    # Simple heuristic: if we got results, the claim is at least searchable
    verdict_hint = "found_references" if results else "no_references_found"

    return {
        "claim":        claim,
        "results":      results,
        "verdict_hint": verdict_hint,
    }


# ── SerpAPI upgrade (optional) ────────────────────────────────────────────────

def serpapi_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Optional: use SerpAPI for higher-quality web search.
    Falls back to DuckDuckGo if SERPAPI_KEY not set.
    """
    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key:
        return web_search(query, max_results)

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q":       query,
                "api_key": api_key,
                "num":     max_results,
                "engine":  "google",
            },
            timeout=15,
        )
        data = resp.json()
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("link", ""),
                "snippet": r.get("snippet", "")[:400],
            }
            for r in data.get("organic_results", [])[:max_results]
        ]
    except Exception as e:
        print(f"[Search] SerpAPI error: {e}, falling back to DuckDuckGo")
        return web_search(query, max_results)
