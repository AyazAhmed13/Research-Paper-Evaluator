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

MAX_TOKENS_PER_CALL = int(__import__("os").getenv("MAX_TOKENS_PER_CALL", 14000))


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (compatible with most modern LLMs)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def truncate_to_token_limit(text: str, limit: int = MAX_TOKENS_PER_CALL) -> str:
    """Truncate text to fit within token limit, preserving start and end."""
    tokens = count_tokens(text)
    if tokens <= limit:
        return text

    # Keep first 70% and last 30% of limit to preserve intro and conclusions
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    keep_start = int(limit * 0.70)
    keep_end   = limit - keep_start

    truncated_ids = token_ids[:keep_start] + token_ids[-keep_end:]
    result = enc.decode(truncated_ids)
    print(f"[Chunker] Truncated from {tokens} → {limit} tokens")
    return result


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
    Build the context string for a specific agent by concatenating
    only its relevant sections. Truncates if over token limit.
    """
    relevant_keys = AGENT_SECTION_MAP.get(agent_name, list(sections.keys()))

    parts = []
    for key in relevant_keys:
        if key in sections and sections[key].strip():
            parts.append(f"[{key.upper().replace('_', ' ')}]\n{sections[key]}")

    combined = "\n\n".join(parts)

    if not combined.strip():
        # Fallback: give agent the full text (truncated)
        all_text = "\n\n".join(sections.values())
        return truncate_to_token_limit(all_text)

    return truncate_to_token_limit(combined)


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
