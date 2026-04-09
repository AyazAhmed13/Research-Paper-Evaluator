"""
tools/scraper.py
----------------
Hybrid arXiv scraper:
  1. Try arXiv experimental HTML full-text  (/html/{id})
  2. Fallback → download PDF and extract with pymupdf
  3. Last resort → abstract only from /abs/ page
"""

import re
import os
import time
import requests
import arxiv
import fitz  # pymupdf
from bs4 import BeautifulSoup
from typing import Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_arxiv_id(url: str) -> str:
    """
    Extract clean arXiv ID from various URL formats.
    Handles:
      https://arxiv.org/abs/2301.00001
      https://arxiv.org/pdf/2301.00001
      https://arxiv.org/abs/2301.00001v2
      2301.00001   (raw ID)
    """
    url = url.strip().rstrip("/")
    # Match pattern like 2301.00001 or 2301.00001v2
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arXiv ID from: {url}")


def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # strip non-ASCII
    return text.strip()


# ── Scraping strategies ───────────────────────────────────────────────────────

def scrape_html_fulltext(arxiv_id: str) -> str:
    """
    Attempt to fetch the full paper from arXiv's experimental HTML endpoint.
    Returns empty string if unavailable or too short.
    """
    base_id = arxiv_id.split("v")[0]  # strip version suffix for HTML URL
    url = f"https://arxiv.org/html/{base_id}"

    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "ArxivEvaluator/1.0"})
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove clutter
        for tag in soup(["script", "style", "nav", "footer", "head",
                          "button", "figure", "figcaption"]):
            tag.decompose()

        # Try to get main content div first
        main = soup.find("div", {"class": re.compile(r"ltx_page_main|ltx_document")})
        text = main.get_text(separator="\n", strip=True) if main else \
               soup.get_text(separator="\n", strip=True)

        return clean_text(text)

    except Exception as e:
        print(f"[Scraper] HTML fetch failed: {e}")
        return ""


def extract_from_pdf(arxiv_id: str) -> str:
    """
    Download PDF via arxiv library and extract text with pymupdf.
    Most reliable fallback.
    """
    try:
        print(f"[Scraper] Downloading PDF for {arxiv_id} ...")
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        pdf_path = f"/tmp/{arxiv_id.replace('/', '_')}.pdf"
        paper.download_pdf(filename=pdf_path)

        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()

        # Clean up temp file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return clean_text("\n".join(pages_text))

    except Exception as e:
        print(f"[Scraper] PDF extraction failed: {e}")
        return ""


def scrape_abstract_only(arxiv_id: str) -> dict:
    """
    Last resort: fetch metadata + abstract from /abs/ page.
    Returns a dict with title, authors, abstract.
    """
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")

        title = soup.find("h1", {"class": "title"})
        title = title.get_text(strip=True).replace("Title:", "").strip() if title else "Unknown"

        authors = soup.find("div", {"class": "authors"})
        authors = authors.get_text(strip=True).replace("Authors:", "").strip() if authors else ""

        abstract = soup.find("blockquote", {"class": "abstract"})
        abstract = abstract.get_text(strip=True).replace("Abstract:", "").strip() if abstract else ""

        return {"title": title, "authors": authors, "abstract": abstract}
    except Exception as e:
        print(f"[Scraper] Abstract-only fetch failed: {e}")
        return {"title": "Unknown", "authors": "", "abstract": ""}


# ── Main entry point ──────────────────────────────────────────────────────────

def fetch_paper(url: str) -> dict:
    """
    Main function. Returns a dict:
    {
        "arxiv_id": str,
        "title": str,
        "authors": str,
        "full_text": str,
        "source": str   # "html" | "pdf" | "abstract_only"
    }
    """
    arxiv_id = extract_arxiv_id(url)
    print(f"[Scraper] Processing arXiv ID: {arxiv_id}")

    # --- Fetch metadata separately (always reliable) ---
    meta = scrape_abstract_only(arxiv_id)

    # --- Strategy 1: HTML full text ---
    full_text = scrape_html_fulltext(arxiv_id)
    source = "html"

    # --- Strategy 2: PDF fallback ---
    if len(full_text) < 2000:
        print("[Scraper] HTML too short or unavailable, falling back to PDF ...")
        full_text = extract_from_pdf(arxiv_id)
        source = "pdf"

    # --- Strategy 3: Abstract only (emergency) ---
    if len(full_text) < 500:
        print("[Scraper] PDF extraction also failed, using abstract only.")
        full_text = meta.get("abstract", "")
        source = "abstract_only"

    print(f"[Scraper] Done. Source={source}, Length={len(full_text)} chars")

    return {
        "arxiv_id": arxiv_id,
        "title": meta.get("title", "Unknown"),
        "authors": meta.get("authors", ""),
        "full_text": full_text,
        "source": source,
    }
