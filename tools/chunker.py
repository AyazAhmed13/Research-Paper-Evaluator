"""
tools/chunker.py
----------------
Splits raw paper text into named sections, then routes only
the relevant sections to each specialized agent.

Key design decisions (from architecture review):
  - Chunking is for SECTION ROUTING, not just token limits.
  - Even with Gemini's 1M context, each agent only sees what it needs.
  - Token counting guards against the 16k-per-call assignment limit.
"""

import re
import tiktoken
from typing import Optional


# ── Section detection ─────────────────────────────────────────────────────────

# Keywords that signal the start of a new section.
# Order matters: more specific patterns first.
SECTION_KEYWORDS = {
    "abstract":     ["abstract"],
    "introduction": ["introduction", "1. introduction", "1 introduction", "background"],
    "related_work": ["related work", "prior work", "literature review", "previous work"],
    "methodology":  ["method", "methodology", "approach", "proposed method",
                     "model", "framework", "architecture", "system design",
                     "experimental setup", "implementation"],
    "results":      ["result", "experiment", "evaluation", "performance",
                     "analysis", "findings", "benchmark", "comparison"],
    "discussion":   ["discussion", "ablation", "limitation", "future work"],
    "conclusion":   ["conclusion", "concluding", "summary"],
    "references":   ["reference", "bibliography"],
}

# Which sections each agent needs (from architecture diagram routing map)
AGENT_SECTION_MAP = {
    "consistency":   ["methodology", "results", "discussion"],
    "grammar":       ["abstract", "introduction", "methodology", "results",
                      "discussion", "conclusion"],
    "novelty":       ["abstract", "introduction", "related_work", "conclusion"],
    "factcheck":     ["results", "methodology", "references"],
    "authenticity":  ["abstract", "results", "conclusion", "discussion"],
}

MAX_TOKENS_PER_CALL = int(__import__("os").getenv("MAX_TOKENS_PER_CALL", 8000))


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (compatible with most modern LLMs)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def chunk_text(text: str, chunk_size: int = 10000, overlap: int = 500) -> list[str]:
    """
    Split text into overlapping chunks of chunk_size tokens.
    Overlap ensures context isn't lost at boundaries.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text, disallowed_special=())

    if len(token_ids) <= chunk_size:
        return [text]  # fits in one call, no chunking needed

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(token_ids):
            break
        start += chunk_size - overlap  # slide forward with overlap

    print(f"[Chunker] Split into {len(chunks)} chunks of ~{chunk_size} tokens each")
    return chunks


# ── Section splitter ──────────────────────────────────────────────────────────

def _detect_section(line: str) -> Optional[str]:
    """
    Check if a line looks like a section heading.
    Returns section key or None.
    """
    stripped = line.strip().lower()

    # Must be short (headings are rarely > 80 chars)
    if len(stripped) > 80:
        return None

    # Remove leading numbers like "2.", "2.1", "III."
    cleaned = re.sub(r"^[\d\.]+\s*", "", stripped)
    cleaned = re.sub(r"^[ivxlcdm]+\.\s*", "", cleaned)  # Roman numerals

    for section_key, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if cleaned.startswith(kw) or stripped.startswith(kw):
                return section_key

    return None


def split_into_sections(text: str) -> dict:
    """
    Split full paper text into a dict of {section_name: section_text}.
    Falls back gracefully: unidentified lines go to the current section.
    """
    sections = {key: [] for key in SECTION_KEYWORDS}
    sections["preamble"] = []  # text before abstract

    current_section = "preamble"

    for line in text.split("\n"):
        detected = _detect_section(line)
        if detected:
            current_section = detected
        else:
            sections.setdefault(current_section, []).append(line)

    # Join and clean each section
    result = {}
    for key, lines in sections.items():
        body = "\n".join(lines).strip()
        if body:
            result[key] = body

    return result


# ── Agent context builder ─────────────────────────────────────────────────────

def get_context_for_agent(agent_name: str, sections: dict) -> str:
    """
    Build context for an agent. If content fits in one call, return as-is.
    If it exceeds the limit, return chunked text with a header so the
    agent knows it's processing part N of M.
    Returns a single string — multi-chunk papers get a summary header injected.
    """
    relevant_keys = AGENT_SECTION_MAP.get(agent_name, list(sections.keys()))

    parts = []
    for key in relevant_keys:
        if key in sections and sections[key].strip():
            parts.append(f"[{key.upper().replace('_', ' ')}]\n{sections[key]}")

    combined = "\n\n".join(parts)

    if not combined.strip():
        combined = "\n\n".join(sections.values())

    total_tokens = count_tokens(combined)
    print(f"[Chunker] Agent '{agent_name}': {total_tokens} tokens — {'CHUNKED' if total_tokens > MAX_TOKENS_PER_CALL else 'fits in one call'}")


    if total_tokens <= MAX_TOKENS_PER_CALL:
        return combined  # fits fine, return as-is

    # Too large — chunk it and return with aggregation instructions
    chunks = chunk_text(combined, chunk_size=8000, overlap=500)

    if len(chunks) == 1:
        return chunks[0]

    # Build a multi-chunk context string with clear section markers
    # The agent prompt already asks for structured output so it handles this naturally
    aggregated_parts = []
    for i, chunk in enumerate(chunks, 1):
        aggregated_parts.append(
            f"[PAPER SEGMENT {i} OF {len(chunks)}]\n{chunk}"
        )

    merged = "\n\n---\n\n".join(aggregated_parts)

    # Final safety check — if merged is still too big somehow, take first 2 chunks only
    if count_tokens(merged) > MAX_TOKENS_PER_CALL * len(chunks):
        merged = "\n\n---\n\n".join(aggregated_parts[:2])

    print(f"[Chunker] Agent '{agent_name}': {total_tokens} tokens → {len(chunks)} chunks merged")
    return merged


# ── Summary stats ─────────────────────────────────────────────────────────────

def get_section_stats(sections: dict) -> dict:
    """Return token count per section (useful for debugging)."""
    return {
        section: {
            "chars": len(text),
            "tokens": count_tokens(text),
        }
        for section, text in sections.items()
    }


# ── Main convenience function ─────────────────────────────────────────────────

def prepare_paper(full_text: str) -> dict:
    """
    Full pipeline: split → stats → return structured paper.
    Returns:
    {
        "sections": dict,
        "stats": dict,
        "total_tokens": int,
    }
    """
    sections = split_into_sections(full_text)
    stats = get_section_stats(sections)
    total_tokens = sum(v["tokens"] for v in stats.values())

    print(f"[Chunker] Sections found: {list(sections.keys())}")
    print(f"[Chunker] Total tokens: {total_tokens}")

    return {
        "sections": sections,
        "stats": stats,
        "total_tokens": total_tokens,
    }
