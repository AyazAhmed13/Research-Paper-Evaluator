"""
agents/novelty_agent.py
-----------------------
Searches for existing literature to evaluate how novel/unique the paper's
contributions are relative to the current state of the art.
Receives: Abstract + Introduction + Related Work + Conclusion
Uses: Semantic Scholar API to find related papers
Outputs: Novelty Index (qualitative) + related works comparison
"""

import os
from crewai import Agent, Task
from crewai.tools import tool
from textwrap import dedent
from tools.search_tool import search_related_papers


# ── CrewAI tool wrapper ───────────────────────────────────────────────────────

@tool("Search Academic Literature")
def search_literature_tool(query: str) -> str:
    """
    Search Semantic Scholar for academic papers related to the given query.
    Returns a formatted list of related papers with titles, authors, and abstracts.
    Use this to find prior work related to the paper being evaluated.
    """
    results = search_related_papers(query, limit=6)
    if not results:
        return "No related papers found for this query."

    lines = []
    for i, paper in enumerate(results, 1):
        lines.append(
            f"{i}. {paper['title']} ({paper['year']})\n"
            f"   Authors: {paper['authors']}\n"
            f"   Abstract: {paper['abstract'][:200]}...\n"
            f"   URL: {paper['url']}"
        )
    return "\n\n".join(lines)


# ── Agent + Task builders ─────────────────────────────────────────────────────

def build_novelty_agent(llm) -> Agent:
    return Agent(
        role="Research Novelty and Originality Assessor",
        goal=dedent("""
            Determine how novel and original the paper's contributions are by
            searching for related prior work and comparing the paper's claims
            against what already exists in the literature.
        """),
        backstory=dedent("""
            You are a research scientist who has served on program committees
            for NeurIPS, ICML, ICLR, ACL, and CVPR for over a decade.
            Your specialty is identifying incremental papers that overstate their
            novelty versus genuinely groundbreaking work. You are rigorous but fair:
            incremental work is not bad — only when it's misrepresented as revolutionary.
            You always ground your assessment in specific related papers you find.
        """),
        llm=llm,
        tools=[search_literature_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,  # Allow a few search iterations
    )


def build_novelty_task(agent: Agent, paper_context: str, title: str) -> Task:
    return Task(
        description=dedent(f"""
            NOTE: If the paper is divided into [PAPER SEGMENT X OF Y] sections below,
            analyze ALL segments and provide ONE unified assessment in your final output.

            Assess the novelty and originality of the research paper titled:
            "{title}"

            --- PAPER SECTIONS ---
            {paper_context}
            --- END OF PAPER ---

            Follow this process:

            STEP 1 — IDENTIFY CLAIMED CONTRIBUTIONS
            Read the abstract, introduction, and conclusion. List the specific
            contributions the authors claim (e.g., "we propose a new architecture",
            "we achieve state-of-the-art on X benchmark", "we introduce dataset Y").

            STEP 2 — SEARCH FOR RELATED WORK
            Use the "Search Academic Literature" tool to search for papers related
            to the core topics of this paper. Run 2-3 searches with different queries
            (e.g., the paper's core method name, the problem being solved, the dataset used).

            STEP 3 — COMPARE
            For each claimed contribution, check whether similar work already exists
            in the literature you found. Be specific about similarities and differences.

            STEP 4 — ASSESS NOVELTY INDEX
            Assign a Novelty Index from this scale:
              - Groundbreaking: Introduces a fundamentally new concept or approach
              - Significant:    Clear advancement over prior work with novel elements
              - Moderate:       Solid incremental contribution with some novelty
              - Marginal:       Minor extension of existing work, limited new ideas
              - Derivative:     Essentially replicates existing work with minor changes

            FORMAT YOUR RESPONSE EXACTLY AS:
            NOVELTY_INDEX: [Groundbreaking/Significant/Moderate/Marginal/Derivative]

            CLAIMED_CONTRIBUTIONS:
            - [contribution 1]
            - [contribution 2]
            (list all identified contributions)

            RELATED_WORK_FOUND:
            - Paper: [title] ([year]) — Similarity: [how it relates to this paper]
            - Paper: ...
            (list 3-5 most relevant papers found)

            NOVELTY_ANALYSIS:
            [2-3 paragraphs comparing paper to related work]

            ORIGINALITY_GAPS:
            [bullet list of any contributions that appear to already exist in literature]

            SUMMARY:
            [2 sentences max — overall novelty verdict]
        """),
        agent=agent,
        expected_output=dedent("""
            A structured novelty report containing:
            - NOVELTY_INDEX rating
            - CLAIMED_CONTRIBUTIONS list
            - RELATED_WORK_FOUND list with similarity notes
            - NOVELTY_ANALYSIS paragraphs
            - ORIGINALITY_GAPS list
            - SUMMARY verdict
        """),
    )
