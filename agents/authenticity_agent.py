"""
agents/authenticity_agent.py
-----------------------------
Detects signs of data fabrication, result manipulation, or statistical
anomalies. Calculates a "Fabrication Probability" risk score.
Receives: Abstract + Results + Conclusion + Discussion
Outputs: Accuracy/Fabrication Score (percentage risk)
"""

from crewai import Agent, Task
from textwrap import dedent


def build_authenticity_agent(llm) -> Agent:
    return Agent(
        role="Research Integrity and Authenticity Auditor",
        goal=dedent("""
            Detect statistical anomalies, implausible results, logical inconsistencies,
            and signs of potential data fabrication or manipulation. Calculate a
            Fabrication Probability risk score based on specific red flags found.
        """),
        backstory=dedent("""
            You are a research integrity specialist trained in statistical methods
            for detecting data fabrication. You have studied cases from the Committee
            on Publication Ethics (COPE) and are familiar with techniques like
            GRIM test, SPRITE analysis, and Benford's Law applied to research data.
            You understand that most papers are legitimate and approach your work
            with appropriate skepticism — you flag concerns, not accusations.
            Your risk score is a calibrated probability, not a verdict of guilt.
        """),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


def build_authenticity_task(agent: Agent, paper_context: str, title: str) -> Task:
    return Task(
        description=dedent(f"""
            Assess the authenticity and research integrity of the paper titled:
            "{title}"

            --- PAPER SECTIONS ---
            {paper_context}
            --- END OF PAPER ---

            Conduct a thorough integrity audit across these dimensions:

            1. RESULT PLAUSIBILITY
               - Are the reported results within the expected range for this field?
               - Do improvements over baselines seem realistic (e.g., a 50% improvement
                 over state-of-the-art without a major architectural change is suspicious)?
               - Are variance/standard deviation values reported? If results are
                 suspiciously "clean" (no variance), flag this.

            2. STATISTICAL RED FLAGS
               - Are p-values reported? Do they seem artificially precise?
               - Are confidence intervals appropriate?
               - Any signs of p-hacking (e.g., only reporting the best runs)?
               - Round numbers in results can indicate fabrication — check for this.

            3. INTERNAL CONSISTENCY
               - Do numbers in tables match numbers mentioned in text?
               - Are the same experiments described consistently throughout the paper?
               - Any contradictions between abstract claims and detailed results?

            4. LOGICAL LEAPS
               - Are conclusions much stronger than what the evidence supports?
               - Any "too good to be true" claims?
               - Are limitations acknowledged honestly?

            5. REPRODUCIBILITY SIGNALS
               - Is enough detail provided to reproduce the experiments?
               - Is code/data promised or provided?
               - Are hyperparameters, seeds, and hardware specs reported?

            6. FABRICATION PROBABILITY SCORE
               Calculate a risk score from 0% to 100%.
               Be conservative — most academic papers are legitimate.
               A paper with code, hardware specs, and verified benchmarks
               should score 5-15% even if minor issues exist.
               Only score above 50% if you find strong direct evidence of fraud.

                 0-15%:   Very likely authentic — no significant red flags
                 16-30%:  Low risk — 1-2 minor concerns (e.g. small numerical discrepancy)
                 31-50%:  Moderate risk — multiple red flags, missing validations
                 51-75%:  High risk — significant integrity concerns, major inconsistencies
                 76-100%: Very high risk — strong indicators of fabrication or fraud

            FORMAT YOUR RESPONSE EXACTLY AS:
            FABRICATION_PROBABILITY: [number]%

            RESULT_PLAUSIBILITY:
            [assessment with specific examples from paper]

            STATISTICAL_RED_FLAGS:
            - [flag 1, or "None identified"]
            - [flag 2]
            ...

            INTERNAL_CONSISTENCY_ISSUES:
            - [issue 1, or "None identified"]
            ...

            LOGICAL_LEAPS:
            - [leap 1, or "None identified"]
            ...

            REPRODUCIBILITY_SCORE: [High/Medium/Low]
            REPRODUCIBILITY_NOTES:
            [1 paragraph on reproducibility]

            RISK_FACTORS_SUMMARY:
            [bullet list of the main factors that drove the fabrication probability score]

            SUMMARY:
            [2 sentences max — overall authenticity verdict]
        """),
        agent=agent,
        expected_output=dedent("""
            A structured authenticity report containing:
            - FABRICATION_PROBABILITY percentage
            - RESULT_PLAUSIBILITY assessment
            - STATISTICAL_RED_FLAGS list
            - INTERNAL_CONSISTENCY_ISSUES list
            - LOGICAL_LEAPS list
            - REPRODUCIBILITY_SCORE and notes
            - RISK_FACTORS_SUMMARY
            - SUMMARY verdict
        """),
    )