"""
app.py
------
Streamlit UI for the Agentic Research Paper Evaluator.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — works on both light and dark Streamlit themes ───────────────────────

st.markdown("""
<style>
    /* Verdict banners */
    .verdict-pass { background:#1a3d2b; border-left:5px solid #34a853; padding:20px 24px; border-radius:10px; }
    .verdict-cond { background:#3d3200; border-left:5px solid #fbbc04; padding:20px 24px; border-radius:10px; }
    .verdict-fail { background:#3d1a1a; border-left:5px solid #ea4335; padding:20px 24px; border-radius:10px; }
    .verdict-pass h2, .verdict-cond h2, .verdict-fail h2 { color:#ffffff !important; margin:0; }
    .verdict-pass p,  .verdict-cond p,  .verdict-fail p  { color:#cccccc !important; margin:8px 0 0; }

    /* Score cards */
    .score-card {
        background: #1e1e2e;
        border: 1px solid #333355;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .score-number { font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
    .score-label  { font-size: 0.82rem; color: #aaaacc; margin-top: 6px; }

    /* Agent output box — dark bg with light text */
    .agent-output {
        background: #0f1117;
        color: #e0e0e0;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.84rem;
        line-height: 1.65;
        max-height: 520px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Novelty index — smaller font so long words fit */
    .novelty-number { font-size: 1.25rem !important; font-weight: 700; }

    /* Sidebar about section */
    .about-agent { display:flex; align-items:flex-start; gap:10px; margin:8px 0; }
    .about-agent-icon { font-size:1.2rem; min-width:24px; }
    .about-agent-text { font-size:0.9rem; color:#cccccc; line-height:1.4; }
    .about-agent-text strong { color:#ffffff; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — About only ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Research Paper Evaluator")
    st.divider()
    st.markdown("### About")
    st.markdown("""
This tool uses **5 specialized AI agents** to autonomously peer-review any arXiv research paper and generate a **Judgement Report**.
    """)

    agents_info = [
        ("🔍", "Consistency", "Checks if methodology supports the claims"),
        ("✍️", "Grammar",     "Evaluates language quality and tone"),
        ("🆕", "Novelty",     "Compares against existing literature"),
        ("✅", "Fact-check",  "Verifies citations, formulas & data"),
        ("🛡️", "Authenticity","Estimates fabrication risk %"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(f"""
        <div class="about-agent">
            <div class="about-agent-icon">{icon}</div>
            <div class="about-agent-text"><strong>{name}</strong><br>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

   


# ── Main header ───────────────────────────────────────────────────────────────

st.title("🔬 Agentic Research Paper Evaluator")
st.markdown("Paste any **arXiv URL** below to get a full AI peer-review with scores across 5 dimensions.")

col1, col2 = st.columns([5, 1])
with col1:
    arxiv_url = st.text_input(
        "arXiv URL",
        value=st.session_state.get("arxiv_url", ""),
        placeholder="https://arxiv.org/abs/1706.03762",
        label_visibility="collapsed",
    )
with col2:
    run_button = st.button("🚀 Evaluate", type="primary", use_container_width=True)

# Quick example buttons
st.markdown("<small style='color:#888'>Try an example:</small>", unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
examples = {
    "Attention Is All You Need": "https://arxiv.org/abs/1706.03762",
    "BERT":                       "https://arxiv.org/abs/1810.04805",
    "GPT-4 Report":               "https://arxiv.org/abs/2303.08774",
}
for col, (name, url) in zip([ex1, ex2, ex3], examples.items()):
    with col:
        if st.button(f"📄 {name}", use_container_width=True):
            st.session_state["arxiv_url"] = url
            st.rerun()


# ── Run evaluation ────────────────────────────────────────────────────────────

if run_button:
    if not arxiv_url.strip():
        st.error("Please enter an arXiv URL.")
        st.stop()

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found. Make sure it is set in your .env file.")
        st.stop()

    st.markdown("### ⏳ Running Evaluation...")
    progress_bar = st.progress(0)
    status_text  = st.empty()
    current_step = [0]
    total_steps  = 8

    def update_progress(msg: str):
        step = current_step[0]
        progress_bar.progress(min(int((step / total_steps) * 100), 99))
        status_text.info(f"⏳ {msg}")
        current_step[0] += 1

    try:
        from crew import run_evaluation
        from report_generator import generate_all_reports

        update_progress("Fetching paper from arXiv ...")
        evaluation = run_evaluation(arxiv_url, progress_callback=update_progress)

        update_progress("Generating reports ...")
        report_paths = generate_all_reports(evaluation)

        progress_bar.progress(100)
        status_text.success("✅ Evaluation complete!")

        st.session_state["evaluation"]   = evaluation
        st.session_state["report_paths"] = report_paths

    except Exception as e:
        status_text.error(f"❌ Evaluation failed: {str(e)}")
        st.exception(e)
        st.stop()


# ── Display results ───────────────────────────────────────────────────────────

if "evaluation" in st.session_state:
    evaluation   = st.session_state["evaluation"]
    report_paths = st.session_state.get("report_paths", {})
    scores       = evaluation["scores"]
    results      = evaluation["results"]
    meta         = evaluation["paper_meta"]

    st.divider()

    # ── Paper info ────────────────────────────────────────────────
    st.markdown(f"## 📄 {meta.get('title', 'Unknown Paper')}")
    arxiv_id = meta.get('arxiv_id', '')
    st.markdown(
        f"**Authors:** {meta.get('authors', 'Unknown')} &nbsp;|&nbsp; "
        f"**arXiv:** [arxiv.org/abs/{arxiv_id}](https://arxiv.org/abs/{arxiv_id}) &nbsp;|&nbsp; "
        f"**Source:** `{meta.get('source', 'unknown')}`"
    )
    st.divider()

    # ── Verdict banner ────────────────────────────────────────────
    verdict = scores.get("overall_verdict", "UNKNOWN")
    verdict_class = {
        "PASS":             "verdict-pass",
        "CONDITIONAL PASS": "verdict-cond",
        "FAIL":             "verdict-fail",
    }.get(verdict, "verdict-cond")
    verdict_icon = {"PASS": "✅", "CONDITIONAL PASS": "⚠️", "FAIL": "❌"}.get(verdict, "❓")

    st.markdown(f"""
    <div class="{verdict_class}">
        <h2>{verdict_icon} Overall Verdict: <strong>{verdict}</strong></h2>
        <p>Based on analysis across consistency, grammar, novelty, fact-checking, and authenticity.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Score cards ───────────────────────────────────────────────
    st.markdown("### Score Summary")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        score = scores.get("consistency_score", "N/A")
        color = "#34a853" if isinstance(score, int) and score >= 70 \
           else "#fbbc04" if isinstance(score, int) and score >= 50 \
           else "#ea4335"
        st.markdown(f"""<div class="score-card">
            <div class="score-number" style="color:{color}">{score}</div>
            <div class="score-label">Consistency Score<br><small>/100</small></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        rating = scores.get("grammar_rating", "N/A")
        color  = "#34a853" if rating == "High" else "#fbbc04" if rating == "Medium" else "#ea4335"
        st.markdown(f"""<div class="score-card">
            <div class="score-number" style="color:{color}">{rating}</div>
            <div class="score-label">Grammar Rating</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        novelty = scores.get("novelty_index", "N/A")
        n_color = {
            "Groundbreaking": "#34a853",
            "Significant":    "#34a853",
            "Moderate":       "#fbbc04",
            "Marginal":       "#ea4335",
            "Derivative":     "#ea4335",
            "None":           "#888888",
        }.get(novelty, "#aaaacc")
        st.markdown(f"""<div class="score-card">
            <div class="score-number novelty-number" style="color:{n_color}">{novelty}</div>
            <div class="score-label">Novelty Index</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        fab = scores.get("fabrication_probability", "N/A")
        fab_display = f"{fab}%" if isinstance(fab, (int, float)) else fab
        color = "#34a853" if isinstance(fab, (int, float)) and fab < 20 \
           else "#fbbc04" if isinstance(fab, (int, float)) and fab < 50 \
           else "#ea4335"
        st.markdown(f"""<div class="score-card">
            <div class="score-number" style="color:{color}">{fab_display}</div>
            <div class="score-label">Fabrication Risk</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Fact check quick stats ────────────────────────────────────
    fc = scores.get("fact_check_summary", {})
    if fc:
        st.markdown("### Fact Check Summary")
        f1, f2, f3, f4 = st.columns(4)
        for col, label, key, color in [
            (f1, "✓ Verified",     "verified_count",     "#34a853"),
            (f2, "? Unverified",   "unverified_count",   "#fbbc04"),
            (f3, "✗ Contradicted", "contradicted_count", "#ea4335"),
            (f4, "! Suspicious",   "suspicious_count",   "#ff9500"),
        ]:
            with col:
                val = fc.get(key, 0)
                st.markdown(f"""<div class="score-card">
                    <div class="score-number" style="color:{color}">{val}</div>
                    <div class="score-label">{label}</div>
                </div>""", unsafe_allow_html=True)
        st.divider()

    # ── Agent report tabs ─────────────────────────────────────────
    st.markdown("### Detailed Agent Reports")
    tabs = st.tabs(["🔍 Consistency", "✍️ Grammar", "🆕 Novelty", "✅ Fact-check", "🛡️ Authenticity"])
    agent_keys = ["consistency", "grammar", "novelty", "factcheck", "authenticity"]

    for tab, key in zip(tabs, agent_keys):
        with tab:
            output = results.get(key, "No output generated.")
            if output and output.strip():
                # Use st.markdown for proper rendering instead of raw HTML
                st.markdown(output)
            else:
                st.warning("No output was generated for this agent.")

    st.divider()

    # ── Downloads ─────────────────────────────────────────────────
    st.markdown("### 📥 Download Report")
    dl1, dl2 = st.columns(2)

    with dl1:
        md_path = report_paths.get("markdown")
        if md_path and os.path.exists(md_path):
            with open(md_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Markdown Report",
                    data=f.read(),
                    file_name=os.path.basename(md_path),
                    mime="text/markdown",
                    use_container_width=True,
                )

    with dl2:
        pdf_path = report_paths.get("pdf")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True,
                )

    # ── Paper sections debug expander ────────────────────────────
    with st.expander("📑 View Detected Paper Sections"):
        sections = evaluation.get("sections", {})
        stats    = evaluation.get("section_stats", {})
        for section, text in sections.items():
            token_count = stats.get(section, {}).get("tokens", "?")
            with st.expander(f"{section.replace('_', ' ').title()} — {token_count} tokens"):
                st.code(text[:2000] + ("..." if len(text) > 2000 else ""), language=None)