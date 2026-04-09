"""
agents/grammar_agent.py
-----------------------
Evaluates the paper's language quality, professional tone, clarity,
and academic writing standards.
Receives: All sections (full paper).
Outputs: Grammar Rating (High/Medium/Low) + detailed findings.
"""

from crewai import Agent, Task
from textwrap import dedent


def build_grammar_agent(llm) -> Agent:
    return Agent(
        role="Academic Language and Style Reviewer",
        goal=dedent("""
            Evaluate the research paper's writing quality, including grammar,
            vocabulary, sentence structure, clarity of expression, academic tone,
            and adherence to professional scientific writing standards.
        """),
        backstory=dedent("""
            You are a professional academic editor who has worked with major
            publishers including Elsevier, Springer, and IEEE for 15 years.
            You specialize in reviewing papers written by non-native English
            speakers and have a deep understanding of what separates clear,
            publishable writing from unclear or unprofessional prose.
            You are constructive in your feedback and specific in your examples.
        """),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


def build_grammar_task(agent: Agent, paper_context: str, title: str) -> Task:
    return Task(
        description=dedent(f"""
            Evaluate the language quality of the research paper titled:
            "{title}"

            --- PAPER SECTIONS ---
            {paper_context}
            --- END OF PAPER ---

            Perform a comprehensive language and style audit:

            1. GRAMMAR & SYNTAX
               - Identify grammatical errors (subject-verb agreement, tense consistency,
                 article usage, punctuation).
               - Quote 3-5 specific problematic sentences with corrections.

            2. CLARITY & READABILITY
               - Are technical concepts explained clearly?
               - Is the paper's structure logical and easy to follow?
               - Are there ambiguous sentences that could confuse readers?

            3. ACADEMIC TONE
               - Is the language formal and appropriate for a research paper?
               - Any informal language, colloquialisms, or overly casual phrasing?
               - Is hedging language used appropriately (e.g., "suggests", "indicates"
                 vs. overclaiming with "proves", "demonstrates definitively")?

            4. VOCABULARY & PRECISION
               - Is domain-specific terminology used correctly?
               - Any imprecise or vague terms where technical language would be better?

            5. ABSTRACT QUALITY
               - Does the abstract clearly summarize the problem, method, and contribution?
               - Is it self-contained?

            6. RATING
               - Assign an overall Grammar Rating: High / Medium / Low
                 High   = publication-ready with minor or no edits needed
                 Medium = noticeable issues but generally readable
                 Low    = significant revision required

            FORMAT YOUR RESPONSE EXACTLY AS:
            GRAMMAR_RATING: [High/Medium/Low]

            GRAMMAR_ISSUES:
            - Issue 1: "[original text]" → Suggestion: "[corrected text]"
            - Issue 2: ...
            (list up to 5 specific examples)

            CLARITY_ASSESSMENT:
            [1-2 paragraphs]

            ACADEMIC_TONE_ASSESSMENT:
            [1-2 paragraphs]

            ABSTRACT_QUALITY:
            [1 paragraph]

            RECOMMENDATIONS:
            [bullet list of 3-5 actionable improvements]

            SUMMARY:
            [2 sentences max — overall language verdict]
        """),
        agent=agent,
        expected_output=dedent("""
            A structured grammar report containing:
            - GRAMMAR_RATING (High/Medium/Low)
            - GRAMMAR_ISSUES with specific examples
            - CLARITY_ASSESSMENT
            - ACADEMIC_TONE_ASSESSMENT
            - ABSTRACT_QUALITY evaluation
            - RECOMMENDATIONS list
            - SUMMARY verdict
        """),
    )
