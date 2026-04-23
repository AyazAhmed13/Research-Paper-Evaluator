"""
agents/consistency_agent.py
----------------------------
Checks whether the paper's methodology actually supports its claimed results.
Receives: [METHODOLOGY] + [RESULTS] + [DISCUSSION] sections.
Outputs: Consistency Score (0-100) + detailed findings.
"""

from crewai import Agent, Task
from textwrap import dedent


def build_consistency_agent(llm) -> Agent:
    return Agent(
        role="Scientific Consistency Reviewer",
        goal=dedent("""
            Rigorously evaluate whether the methodology described in the paper
            logically and statistically supports the results and claims made.
            Identify any logical leaps, missing validation steps, unsupported
            conclusions, or mismatches between experimental design and reported outcomes.
        """),
        backstory=dedent("""
            You are a senior peer reviewer with 20 years of experience at top
            academic journals (Nature, Science, NeurIPS). You have a sharp eye
            for methodological flaws — you've caught fabricated benchmarks,
            cherry-picked metrics, and overstated conclusions in hundreds of papers.
            You are systematic, fair, and back every critique with specific evidence
            from the paper text.
        """),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


def build_consistency_task(agent: Agent, paper_context: str, title: str) -> Task:
    return Task(
        description=dedent(f"""
            NOTE: If the paper is divided into [PAPER SEGMENT X OF Y] sections below,
            analyze ALL segments and provide ONE unified assessment in your final output.

            Analyze the following sections of the research paper titled:
            "{title}"

            --- PAPER SECTIONS ---
            {paper_context}
            --- END OF PAPER ---

            Perform a consistency audit by answering these questions:

            1. CLAIM-METHOD ALIGNMENT
               - List the top 3-5 main claims/contributions stated in the paper.
               - For each claim, identify whether the methodology provides
                 sufficient evidence to support it. Be specific.

            2. EXPERIMENTAL VALIDITY
               - Are the evaluation metrics appropriate for the task?
               - Is the baseline comparison fair and comprehensive?
               - Are there missing ablations or control experiments?

            3. LOGICAL GAPS
               - Are there any conclusions that go beyond what the data shows?
               - Any statistical anomalies (e.g., suspiciously perfect results)?
               - Any assumptions stated but not validated?

            4. SCORE
               - Assign a Consistency Score from 0 to 100.
                 (100 = methodology fully supports all claims,
                  0 = fundamental mismatch between method and results)

            FORMAT YOUR RESPONSE EXACTLY AS:
            CONSISTENCY_SCORE: [number 0-100]

            CLAIMS_ANALYZED:
            - Claim 1: [claim text] → Support: [Supported/Partially/Unsupported] — [reason]
            - Claim 2: [claim text] → Support: [Supported/Partially/Unsupported] — [reason]
            (add more as needed)

            EXPERIMENTAL_VALIDITY:
            [2-3 paragraphs]

            LOGICAL_GAPS:
            [bullet list of specific gaps found, or "None identified" if clean]

            SUMMARY:
            [2 sentences max — overall consistency verdict]
        """),
        agent=agent,
        expected_output=dedent("""
            A structured consistency report containing:
            - CONSISTENCY_SCORE (0-100)
            - CLAIMS_ANALYZED list
            - EXPERIMENTAL_VALIDITY assessment
            - LOGICAL_GAPS list
            - SUMMARY verdict
        """),
    )
