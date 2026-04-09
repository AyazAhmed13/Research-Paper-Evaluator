"""
crew.py
-------
Main orchestrator. Assembles the CrewAI crew with all 5 specialized agents
and runs them against a given arXiv paper.

Usage:
    from crew import run_evaluation
    result = run_evaluation("https://arxiv.org/abs/2301.00001")
"""

import os
from dotenv import load_dotenv
from crewai import Crew, Process
from crewai import LLM

from tools.scraper import fetch_paper
from tools.chunker import prepare_paper, get_context_for_agent

from agents.consistency_agent  import build_consistency_agent,  build_consistency_task
from agents.grammar_agent      import build_grammar_agent,      build_grammar_task
from agents.novelty_agent      import build_novelty_agent,      build_novelty_task
from agents.factcheck_agent    import build_factcheck_agent,    build_factcheck_task
from agents.authenticity_agent import build_authenticity_agent, build_authenticity_task

load_dotenv()


# ── LLM Setup ─────────────────────────────────────────────────────────────────
import os
from crewai import LLM


def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Get it from https://openrouter.ai/keys")
    
    # Use a working free model (Gemma 4 26B is powerful, 262K context)
    return LLM(
        model="openrouter/nvidia/nemotron-nano-12b-v2-vl:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    )
'''
def get_llm():
    """
    Returns a CrewAI LLM instance for Gemini 1.5 Flash (free tier, 1M context).
    Uses CrewAI's native LLM class – no LangChain dependencies.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. "
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then add it to your .env file."
        )

    # CrewAI's LLM class uses LiteLLM under the hood
    return LLM(
        model="gemini/gemini-2.5-flash",   # LiteLLM model string
        temperature=0.1,                   # Low temp for analytical tasks
        max_tokens=4096,                   # Reasonable output limit
        api_key=api_key,                   # Explicitly pass the key
    )'''



# ── Main evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(arxiv_url: str, progress_callback=None) -> dict:
    """
    Full evaluation pipeline for a given arXiv URL.

    Args:
        arxiv_url:          URL like https://arxiv.org/abs/2301.00001
        progress_callback:  Optional callable(step: str) for UI progress updates

    Returns:
        dict with keys:
          paper_meta     — title, authors, arxiv_id, source
          sections       — dict of detected sections
          section_stats  — token counts per section
          results        — dict of agent outputs (raw text)
          scores         — parsed scores dict
    """

    def log(msg):
        print(f"\n{'='*60}\n{msg}\n{'='*60}")
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Fetch paper ────────────────────────────────────────
    log("Step 1/4 — Fetching paper from arXiv ...")
    paper = fetch_paper(arxiv_url)

    if not paper["full_text"]:
        raise ValueError(f"Could not retrieve paper content from: {arxiv_url}")

    log(f"Paper fetched: '{paper['title']}' (source: {paper['source']})")

    # ── Step 2: Chunk and decompose ────────────────────────────────
    log("Step 2/4 — Decomposing paper into sections ...")
    prepared = prepare_paper(paper["full_text"])
    sections = prepared["sections"]

    log(f"Sections found: {list(sections.keys())}")
    log(f"Total tokens: {prepared['total_tokens']}")

    # ── Step 3: Build agents and tasks ────────────────────────────
    log("Step 3/4 — Initializing specialized agents ...")
    llm = get_llm()
    title = paper["title"]

    # Build agents
    consistency_agent  = build_consistency_agent(llm)
    grammar_agent      = build_grammar_agent(llm)
    novelty_agent      = build_novelty_agent(llm)
    factcheck_agent    = build_factcheck_agent(llm)
    authenticity_agent = build_authenticity_agent(llm)

    # Build tasks with relevant paper sections for each agent
    consistency_task = build_consistency_task(
        consistency_agent,
        get_context_for_agent("consistency", sections),
        title,
    )
    grammar_task = build_grammar_task(
        grammar_agent,
        get_context_for_agent("grammar", sections),
        title,
    )
    novelty_task = build_novelty_task(
        novelty_agent,
        get_context_for_agent("novelty", sections),
        title,
    )
    factcheck_task = build_factcheck_task(
        factcheck_agent,
        get_context_for_agent("factcheck", sections),
        title,
    )
    authenticity_task = build_authenticity_task(
        authenticity_agent,
        get_context_for_agent("authenticity", sections),
        title,
    )

    # ── Step 4: Run the crew ───────────────────────────────────────
    log("Step 4/4 — Running multi-agent evaluation ...")

    crew = Crew(
        agents=[
            consistency_agent,
            grammar_agent,
            novelty_agent,
            factcheck_agent,
            authenticity_agent,
        ],
        tasks=[
            consistency_task,
            grammar_task,
            novelty_task,
            factcheck_task,
            authenticity_task,
        ],
        process=Process.sequential,   # Run agents one after another
        verbose=True,
        memory=False,                 # No cross-agent memory needed
    )

    crew_output = crew.kickoff()

    # ── Parse results ──────────────────────────────────────────────
    raw_results = {
        "consistency":  str(consistency_task.output  or ""),
        "grammar":      str(grammar_task.output      or ""),
        "novelty":      str(novelty_task.output      or ""),
        "factcheck":    str(factcheck_task.output    or ""),
        "authenticity": str(authenticity_task.output or ""),
    }

    scores = parse_scores(raw_results)

    log("Evaluation complete!")

    return {
        "paper_meta":    paper,
        "sections":      sections,
        "section_stats": prepared["stats"],
        "results":       raw_results,
        "scores":        scores,
    }


# ── Score parser ───────────────────────────────────────────────────────────────

def parse_scores(results: dict) -> dict:
    """
    Extract numeric/categorical scores from raw agent text outputs.
    Returns a clean dict of scores for report generation.
    """
    import re

    scores = {
        "consistency_score":       None,
        "grammar_rating":          None,
        "novelty_index":           None,
        "fabrication_probability": None,
        "fact_check_summary":      {},
        "overall_verdict":         None,
    }

    # Consistency score (0-100)
    m = re.search(r"CONSISTENCY_SCORE:\s*(\d+)", results["consistency"], re.IGNORECASE)
    if m:
        scores["consistency_score"] = int(m.group(1))

    # Grammar rating
    m = re.search(r"GRAMMAR_RATING:\s*(High|Medium|Low)", results["grammar"], re.IGNORECASE)
    if m:
        scores["grammar_rating"] = m.group(1).capitalize()

    # Novelty index
    m = re.search(
        r"NOVELTY_INDEX:\s*(Groundbreaking|Significant|Moderate|Marginal|Derivative)",
        results["novelty"], re.IGNORECASE | re.DOTALL
    )
    if not m:
        # fallback: search anywhere in text
        m = re.search(
            r"(Groundbreaking|Significant|Moderate|Marginal|Derivative)",
            results["novelty"], re.IGNORECASE
        )
    if m:
        scores["novelty_index"] = m.group(1).capitalize()

    # Fabrication probability
    m = re.search(r"FABRICATION_PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%", results["authenticity"], re.IGNORECASE)
    if m:
        scores["fabrication_probability"] = float(m.group(1))

    # Fact check counts
    for key in ["VERIFIED_COUNT", "UNVERIFIED_COUNT", "CONTRADICTED_COUNT", "SUSPICIOUS_COUNT"]:
        m = re.search(rf"{key}:\s*(\d+)", results["factcheck"], re.IGNORECASE)
        if m:
            scores["fact_check_summary"][key.lower()] = int(m.group(1))

    # Overall verdict (computed from scores)
    scores["overall_verdict"] = compute_verdict(scores)

    return scores


def compute_verdict(scores: dict) -> str:
    """
    Compute a PASS / CONDITIONAL PASS / FAIL recommendation
    based on all agent scores.
    """
    issues = []

    c_score = scores.get("consistency_score")
    if c_score is not None and c_score < 50:
        issues.append(f"Low consistency score ({c_score}/100)")

    grammar = scores.get("grammar_rating")
    if grammar == "Low":
        issues.append("Poor grammar/language quality")

    novelty = scores.get("novelty_index")
    if novelty in ("Marginal", "Derivative"):
        issues.append(f"Insufficient novelty ({novelty})")

    fab_prob = scores.get("fabrication_probability")
    if fab_prob is not None and fab_prob > 60:
        issues.append(f"High fabrication risk ({fab_prob}%)")

    fc = scores.get("fact_check_summary", {})
    contradicted = fc.get("contradicted_count", 0)
    if contradicted > 0:
        issues.append(f"{contradicted} contradicted fact(s) found")

    if not issues:
        return "PASS"
    elif len(issues) <= 2 and (fab_prob is None or fab_prob < 60):
        return "CONDITIONAL PASS"
    else:
        return "FAIL"