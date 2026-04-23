"""
agents/factcheck_agent.py
--------------------------
Verifies cited constants, formulas, historical facts, and dataset claims
in the paper using web search.
Receives: Results + Methodology + References sections
Uses: DuckDuckGo / SerpAPI web search
Outputs: Fact Check Log (verified vs unverified claims)
"""

from crewai import Agent, Task
from crewai.tools import tool
from textwrap import dedent
from tools.search_tool import web_search, serpapi_search


# ── CrewAI tool wrapper ───────────────────────────────────────────────────────

@tool("Web Fact Check")
def web_fact_check_tool(claim: str) -> str:
    """
    Search the web to verify a specific factual claim from a research paper.
    Use this to check: mathematical constants, dataset statistics, benchmark scores,
    historical facts, or any specific numerical claim in the paper.
    Input should be a concise, searchable claim or question.
    """
    results = serpapi_search(f"verify: {claim}", max_results=4)
    if not results:
        return f"No web results found for: {claim}"

    lines = [f"Search results for: '{claim}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n"
            f"   {r['snippet']}\n"
            f"   Source: {r['url']}"
        )
    return "\n\n".join(lines)


# ── Agent + Task builders ─────────────────────────────────────────────────────

def build_factcheck_agent(llm) -> Agent:
    return Agent(
        role="Scientific Fact-Checker and Claims Verifier",
        goal=dedent("""
            Identify and verify specific factual claims in the research paper,
            including mathematical constants, benchmark dataset statistics,
            cited results, and quantitative assertions. Produce a clear log
            of which claims are verified, unverified, or flagged as suspicious.
        """),
        backstory=dedent("""
            You are a scientific fact-checker who has worked with Retraction Watch
            and major journal editorial offices for 10 years. You specialize in
            catching numerical errors, misattributed results, incorrect benchmark
            statistics, and fabricated or misquoted figures. You are methodical:
            you check specific claims one at a time and document your findings clearly.
            You only flag something as "false" when you have clear contradicting evidence.
        """),
        llm=llm,
        tools=[web_fact_check_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=4,
    )


def build_factcheck_task(agent: Agent, paper_context: str, title: str) -> Task:
    return Task(
        description=dedent(f"""
            NOTE: If the paper is divided into [PAPER SEGMENT X OF Y] sections below,
            analyze ALL segments and provide ONE unified assessment in your final output.

            Fact-check the claims in the research paper titled:
            "{title}"

            --- PAPER SECTIONS ---
            {paper_context}
            --- END OF PAPER ---

            Follow this systematic process:

            STEP 1 — EXTRACT VERIFIABLE CLAIMS
            Scan the paper and list 5-8 specific, verifiable factual claims. Focus on:
              a) Mathematical constants or formulas cited (e.g., "using Adam optimizer
                 with β1=0.9, β2=0.999")
              b) Dataset statistics (e.g., "ImageNet contains 1.2 million images")
              c) Benchmark results cited from other papers
              d) Historical or foundational facts ("BERT was introduced in 2018")
              e) Specific numerical results or thresholds

            STEP 2 — VERIFY EACH CLAIM
            For each claim you identified, use the "Web Fact Check" tool to search
            for verification. Use concise, specific queries.

            STEP 3 — CLASSIFY EACH CLAIM
            For each claim, assign a status:
              ✓ VERIFIED      — confirmed by reliable sources
              ? UNVERIFIED    — couldn't find enough evidence either way
              ✗ CONTRADICTED  — found evidence suggesting the claim is wrong
              ! SUSPICIOUS    — claim is technically unverifiable but seems off

            FORMAT YOUR RESPONSE EXACTLY AS:
            FACT_CHECK_LOG:
            | # | Claim | Status | Source/Note |
            |---|-------|--------|-------------|
            | 1 | [claim text] | ✓ VERIFIED | [source or note] |
            | 2 | [claim text] | ? UNVERIFIED | [why unverifiable] |
            | 3 | [claim text] | ✗ CONTRADICTED | [contradicting source] |
            (list all 5-8 claims)

            VERIFIED_COUNT: [number]
            UNVERIFIED_COUNT: [number]
            CONTRADICTED_COUNT: [number]
            SUSPICIOUS_COUNT: [number]

            KEY_FINDINGS:
            [bullet list of the most important fact-check findings]

            SUMMARY:
            [2 sentences max — overall fact-check verdict]
        """),
        agent=agent,
        expected_output=dedent("""
            A structured fact-check report containing:
            - FACT_CHECK_LOG table with all claims and their verification status
            - VERIFIED_COUNT, UNVERIFIED_COUNT, CONTRADICTED_COUNT, SUSPICIOUS_COUNT
            - KEY_FINDINGS bullet list
            - SUMMARY verdict
        """),
    )
