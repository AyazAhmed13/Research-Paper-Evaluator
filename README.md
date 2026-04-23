# 🔬 Agentic Research Paper Evaluator

An AI-powered **multi-agent system** that autonomously peer-reviews any arXiv research paper and produces a comprehensive **Judgement Report** — with scores across 5 evaluation dimensions, a Pass/Fail verdict, and downloadable PDF/Markdown output.

Built with **CrewAI** + **OpenRouter** (free tier).

---

## What It Does

Paste any arXiv URL → get a full AI peer review in minutes:

| Agent | What it evaluates | Output |
|-------|-------------------|--------|
| 🔍 **Consistency** | Does the methodology actually prove the claims? | Score 0–100 |
| ✍️ **Grammar** | Language quality, tone, academic writing standard | High / Medium / Low |
| 🆕 **Novelty** | How original is this vs existing literature? | Groundbreaking → Derivative |
| ✅ **Fact-check** | Are cited numbers, datasets, benchmarks accurate? | Verified / Unverified / Contradicted log |
| 🛡️ **Authenticity** | Any signs of data fabrication or statistical anomalies? | Fabrication Probability % |

Final output: **PASS / CONDITIONAL PASS / FAIL** verdict + full PDF and Markdown report.

---

## Architecture

```
User enters arXiv URL
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  tools/scraper.py                    │
│  1. Try HTML full-text  →  arxiv.org/html/{id}      │
│  2. Fallback: PDF download  →  pymupdf extraction   │
│  3. Last resort: abstract only  →  /abs/ page       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  tools/chunker.py                    │
│  Split into sections: Abstract, Introduction,        │
│  Methodology, Results, Conclusion, References        │
│  Count tokens with tiktoken (enforce 16k limit)      │
│  Chunk oversized sections + route to each agent      │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              crew.py — CrewAI Orchestrator           │
│         Process.sequential — agents run 1 by 1      │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
Agent 1    Agent 2    Agent 3    Agent 4    Agent 5
Consistency Grammar   Novelty   Fact-check Authenticity
   │          │          │          │          │
   └──────────┴──────────┴──────────┴──────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            report_generator.py                       │
│  Parse scores → clean output → generate MD + PDF    │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                   app.py (Streamlit)                 │
│  Score cards · Verdict banner · Agent tabs · Download│
└─────────────────────────────────────────────────────┘
```

---

## Agent Details

### Agent 1 — Consistency (runs first)
- **Input:** Methodology + Results + Discussion sections
- **Task:** Lists every major claim, checks if methodology proves each one, flags logical gaps
- **Output:** Score 0–100 + Claims Analyzed list + Experimental Validity + Logical Gaps
- **Tools:** None (pure LLM reasoning)

### Agent 2 — Grammar (runs second)
- **Input:** Full paper (all sections)
- **Task:** Evaluates grammar errors, academic tone, clarity, abstract quality
- **Output:** High/Medium/Low rating + specific issue examples with corrections + recommendations
- **Tools:** None (pure LLM reasoning)

### Agent 3 — Novelty (runs third)
- **Input:** Abstract + Introduction + Related Work + Conclusion
- **Task:** Searches Semantic Scholar for related papers, compares claimed contributions against literature
- **Output:** Novelty Index (Groundbreaking/Significant/Moderate/Marginal/Derivative) + related papers found
- **Tools:** search_literature_tool → Semantic Scholar API (2–3 search rounds, max_iter=3)

### Agent 4 — Fact-check (runs fourth)
- **Input:** Results + Methodology + References
- **Task:** Extracts 5–8 specific verifiable claims, web-searches each to verify
- **Output:** Fact Check Log table (✓ Verified / ? Unverified / ✗ Contradicted / ! Suspicious)
- **Tools:** web_fact_check_tool → DuckDuckGo or SerpAPI (max_iter=4)

### Agent 5 — Authenticity (runs fifth)
- **Input:** Abstract + Results + Conclusion + Discussion
- **Task:** Checks result plausibility, statistical red flags, internal consistency, logical leaps, reproducibility
- **Output:** Fabrication Probability % + risk factor breakdown + Reproducibility Score
- **Tools:** None (pure LLM reasoning)

---

## Execution Order

Agents run **sequentially** (one after another), not in parallel:

```python
crew = Crew(process=Process.sequential, ...)
```

**Why sequential?** Free OpenRouter models have rate limits and context window constraints. Running 5 agents simultaneously would hit those limits and fail. Sequential execution is slower (5–10 min per paper) but reliable and stable.

---

## Section Routing Map

The chunker routes only relevant sections to each agent — not the full paper:

```
Consistency  ←  Methodology + Results + Discussion
Grammar      ←  Abstract + Introduction + Methodology + Results + Conclusion
Novelty      ←  Abstract + Introduction + Related Work + Conclusion
Fact-check   ←  Results + Methodology + References
Authenticity ←  Abstract + Results + Conclusion + Discussion
```

---

## Token Budget Enforcement

The system enforces a hard token limit per agent call via `tools/chunker.py`:

1. Each section's tokens are counted with `tiktoken` (`cl100k_base` encoding)
2. Each agent receives only its relevant sections (section routing map above)
3. If combined context exceeds the limit, it is **split into overlapping chunks** with 300-token overlap — no content is dropped
4. Chunks are merged with `[PAPER SEGMENT X OF Y]` markers so the agent knows it is reading a partial section and produces one unified output

This is proper **chunk + aggregate** handling, not truncation.

---

## Project Structure

```
arxiv-evaluator/
│
├── agents/
│   ├── __init__.py               # Package exports
│   ├── consistency_agent.py      # Agent 1 — methodology vs claims checker
│   ├── grammar_agent.py          # Agent 2 — language quality evaluator
│   ├── novelty_agent.py          # Agent 3 — literature search + originality assessor
│   ├── factcheck_agent.py        # Agent 4 — web-based fact verifier
│   └── authenticity_agent.py     # Agent 5 — fabrication risk assessor
│
├── tools/
│   ├── __init__.py               # Package exports
│   ├── scraper.py                # Hybrid HTML → PDF → abstract arXiv scraper
│   ├── chunker.py                # Section splitter + token counter + chunk router
│   └── search_tool.py            # Semantic Scholar + DuckDuckGo + SerpAPI
│
├── reports/                      # Auto-created — generated reports saved here
│
├── crew.py                       # Main orchestrator — assembles and runs CrewAI crew
├── report_generator.py           # Builds clean Markdown + PDF from agent outputs
├── app.py                        # Streamlit UI
├── run.py                        # CLI runner (no UI needed)
├── requirements.txt              # All dependencies
├── .env.example                  # API key template
└── README.md                     # This file
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/arxiv-evaluator
cd arxiv-evaluator
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your OpenRouter API key — **this is the only required key**:

```env
OPENROUTER_API_KEY=your_key_here
```

Get a free key at: https://openrouter.ai/keys

Optional keys (everything works without them):
```env
SERPAPI_KEY=              # Better web search — 100 free calls/month at serpapi.com
SEMANTIC_SCHOLAR_API_KEY= # Higher rate limits for novelty search
```

### 3. Run

**Option A — Streamlit UI (recommended for demo)**
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

**Option B — CLI (quick testing)**
```bash
python run.py https://arxiv.org/abs/1706.03762
python run.py https://arxiv.org/abs/1810.04805 --no-pdf
```

**Option C — Python API (integrate into other code)**
```python
from crew import run_evaluation
from report_generator import generate_all_reports

evaluation   = run_evaluation("https://arxiv.org/abs/1706.03762")
report_paths = generate_all_reports(evaluation)
# report_paths = {"markdown": "./reports/...", "pdf": "./reports/..."}
```

---

## Output — Judgement Report

Every evaluation produces two files in `./reports/`:

**Filename format:** `Paper_Title_evaluation_report.pdf` / `.md`

**Report sections:**

| Section | Content |
|---------|---------|
| Executive Summary | PASS / CONDITIONAL PASS / FAIL + score table |
| Consistency Analysis | Score 0–100 + per-claim breakdown |
| Grammar & Language | High/Medium/Low + specific corrections |
| Novelty Assessment | Index + related papers found + originality gaps |
| Fact Check Log | Table of all claims with verification status |
| Authenticity & Fabrication Risk | % risk + red flags + reproducibility score |
| Paper Structure Statistics | Token count per section |

**Verdict logic:**
- **PASS** — No major issues across all 5 dimensions
- **CONDITIONAL PASS** — 1–2 minor issues, fabrication risk under 60%
- **FAIL** — Low consistency score, poor grammar, high fabrication risk, or contradicted facts

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `crewai` | 1.13.0 | Agentic framework — orchestrates agents and tasks |
| `crewai-tools` | 1.13.0 | Tool decorators for agent web search |
| `arxiv` | latest | Download arXiv PDFs via official API |
| `pymupdf` | latest | Extract text from PDF files |
| `requests` | latest | HTTP requests for HTML scraping |
| `beautifulsoup4` | latest | Parse arXiv HTML full-text |
| `lxml` | latest | Fast HTML parser for BeautifulSoup |
| `tiktoken` | latest | Token counting to enforce token limit |
| `semanticscholar` | latest | Academic literature search API |
| `duckduckgo-search` | latest | Free web search for fact-checking |
| `fpdf2` | latest | Generate PDF reports |
| `streamlit` | latest | Web UI framework |
| `python-dotenv` | latest | Load API keys from .env file |

---

## Key Design Decisions

**1. Hybrid scraper with 3-tier fallback**
arXiv HTML (`/html/{id}`) → PDF download + pymupdf → abstract only. Avoids the common mistake of only scraping the abstract from `/abs/` which gives ~200 words instead of the full 8,000–15,000 word paper.

**2. Section routing instead of full paper per agent**
Sending the full paper to every agent is wasteful and makes prompts noisy. The Consistency agent does not need References. Grammar needs everything. Smart routing produces cleaner, more focused agent outputs.

**3. Chunk + aggregate for large papers**
If a paper's relevant sections exceed the token limit, the chunker splits them into overlapping chunks with segment markers. The agent processes all segments in one call and produces a single unified output. No content is ever dropped.

**4. CrewAI native LLM class**
CrewAI 1.x has its own `LLM` class that connects to OpenRouter via LiteLLM internally. No LangChain dependency needed — this eliminates an entire class of version conflict errors.

**5. Conservative fabrication scoring**
The Authenticity agent is explicitly instructed to be conservative. A paper with released code, hardware specs, and verified benchmarks should score 5–15% even with minor issues. This prevents over-flagging legitimate papers.

**6. Sequential execution over parallel**
Free OpenRouter models have context and rate limits. Sequential execution is slower but guaranteed stable — no race conditions, no simultaneous rate limit hits.

---

## Limitations

- Free OpenRouter models have context window limits — very large papers may have sections trimmed to fit the model's capacity.
- Novelty search quality depends on Semantic Scholar coverage — very new papers may not appear yet.
- Fabrication probability is a risk indicator based on LLM judgment, not a definitive fraud detection system.
- Section detection uses keyword matching — unusual paper structures may cause misclassification.
- Non-English papers will have reduced grammar analysis accuracy.
- Mathematical formulas extracted from PDF may have encoding artifacts (e.g., `1 10 4` instead of `1e-4`).

---

## Future Improvements

- **Parallel execution** with rate limit handling (exponential backoff + retry) — would cut evaluation time from ~8 min to ~2 min
- **Paper caching** — store scraped text in a local DB so re-running the same URL is instant
- **Direct PDF upload** — support papers not on arXiv
- **Citations Agent** — 6th agent that verifies every reference actually exists
- **Comparison mode** — evaluate two papers side by side
- **ML-based section detection** — replace keyword matching with a trained classifier for more accurate section boundaries
- **Batch evaluation** — submit multiple URLs and get reports for all of them

---

## Example Papers to Test

| Paper | arXiv URL |
|-------|-----------|
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| BERT | https://arxiv.org/abs/1810.04805 |
| GPT-4 Technical Report | https://arxiv.org/abs/2303.08774 |
| LLaMA | https://arxiv.org/abs/2302.13971 |
| Stable Diffusion | https://arxiv.org/abs/2112.10752 |

---

