#!/usr/bin/env python3
"""
Streamlit demo for Meta-Review AI — CSV-only (no model required).

Run: streamlit run src/demo_csv.py
"""

import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "predictions" / "pegasus_x_meta_review_predictions.csv"
TEST_JSONL = PROJECT_ROOT / "data" / "processed" / "test.jsonl"

# CSS must come after set_page_config (which is first in main())
CSS = """
<style>
    /* Hero */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .hero p { font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0; }
    /* Decision badges */
    .decision-accept {
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    .decision-reject {
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    .decision-unknown {
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 600; background: #e5e7eb; color: #4b5563;
    }
    /* Cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    .result-card h4 {
        font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #6366f1; margin-bottom: 0.75rem;
    }
    .result-card .content {
        font-size: 0.95rem; line-height: 1.7; color: #374151;
    }
    /* Match badge */
    .match-yes {
        display: inline-block; padding: 0.4rem 0.9rem; border-radius: 8px;
        font-weight: 600; background: #d1fae5; color: #065f46;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }
    .match-no {
        display: inline-block; padding: 0.4rem 0.9rem; border-radius: 8px;
        font-weight: 600; background: #fee2e2; color: #991b1b;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    /* Hide Streamlit branding for cleaner demo */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""


def extract_meta_review(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if "META_REVIEW:" in text:
        return text.split("META_REVIEW:", 1)[1].strip()
    return text


def truncate_repetition(text: str) -> str:
    """Truncate obvious repetition (e.g. 'basedbasedbased' or 'paper, paper, paper') for display."""
    if not text or len(text) < 50:
        return text
    # Pattern 1: "word, word, word" (comma-separated repeat)
    m = re.search(r"(\b\w+\b)(\s*,\s*\1){8,}", text)
    if m:
        return text[: m.start()].rstrip(" ,") + "\n\n_[Repetitive output truncated]_"
    # Pattern 2: "wordwordword" (no spaces, e.g. basedbasedbased)
    m = re.search(r"(\w{3,})\1{8,}", text)
    if m:
        return text[: m.start()].rstrip() + "\n\n_[Repetitive output truncated]_"
    return text


def decision_badge(decision: str) -> str:
    d = (decision or "").upper()
    cls = "decision-accept" if "ACCEPT" in d else "decision-reject" if "REJECT" in d else "decision-unknown"
    return f'<span class="{cls}">{decision or "—"}</span>'


def safe_display(text: str) -> str:
    return html.escape(text).replace("\n", "<br>") if text else ""


def parse_title_abstract(input_text: str) -> tuple[str, str]:
    """Extract title and abstract from input_text prompt."""
    title, abstract = "N/A", "N/A"
    if not input_text:
        return title, abstract
    if "PAPER TITLE:" in input_text:
        after_title = input_text.split("PAPER TITLE:", 1)[1]
        title = after_title.split("\n", 1)[0].strip() or "N/A"
    if "ABSTRACT:" in input_text:
        after_abs = input_text.split("ABSTRACT:", 1)[1]
        abstract = after_abs.split("AGGREGATES:", 1)[0].strip() or "N/A"
    return title, abstract


def load_paper_metadata() -> dict:
    """Load paper_id -> {title, abstract} from test.jsonl."""
    meta = {}
    if not TEST_JSONL.exists():
        return meta
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                title, abstract = parse_title_abstract(ex.get("input_text", ""))
                meta[ex["paper_id"]] = {"title": title, "abstract": abstract}
            except Exception:
                continue
    return meta


def main():
    st.set_page_config(
        page_title="Meta-Review AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🤖 Meta-Review AI</h1>
        <p>AI that writes ICLR-style meta-reviews & accept/reject decisions</p>
        <p style="font-size: 0.9rem; opacity: 0.85;">Pegasus-X Base · 862 papers</p>
    </div>
    """, unsafe_allow_html=True)

    if not CSV_PATH.exists():
        st.error(f"Predictions file not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    paper_ids = df["paper_id"].tolist()
    paper_meta = load_paper_metadata()

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Stats")
        n_total = len(df)
        n_correct = (df["true_decision"] == df["pred_decision"]).sum()
        acc = 100 * n_correct / n_total if n_total else 0
        st.metric("Accuracy", f"{acc:.1f}%")
        st.metric("Papers", n_total)

    # Paper selector
    idx = st.selectbox(
        "Select paper",
        range(len(paper_ids)),
        format_func=lambda i: f"{paper_ids[i]} — {df[df['paper_id'] == paper_ids[i]]['pred_decision'].values[0]}",
    )

    paper_id = paper_ids[idx]
    row = df[df["paper_id"] == paper_id].iloc[0]
    meta = paper_meta.get(paper_id, {})
    title = meta.get("title", "N/A")
    abstract = meta.get("abstract", "N/A")
    true_dec = row["true_decision"]
    pred_dec = row["pred_decision"]
    is_match = true_dec == pred_dec

    # Paper info
    st.markdown("---")
    st.markdown("### Paper")
    st.markdown(f"**{title}**")
    st.caption(f"ID: {paper_id}")
    with st.expander("Abstract"):
        st.write(abstract)

    # Decision row
    st.markdown("### Decision")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("**Ground truth**")
        st.markdown(decision_badge(true_dec), unsafe_allow_html=True)
    with c2:
        st.markdown("**Model prediction**")
        st.markdown(decision_badge(pred_dec), unsafe_allow_html=True)
    with c3:
        st.markdown("**Match**")
        cls = "match-yes" if is_match else "match-no"
        txt = "✓ Correct" if is_match else "✗ Incorrect"
        st.markdown(f'<span class="{cls}">{txt}</span>', unsafe_allow_html=True)

    # Side-by-side meta-reviews
    st.markdown("---")
    st.markdown("### Meta-review")
    left, right = st.columns(2)
    pred_text = extract_meta_review(row["pred_meta_review"])
    true_text = extract_meta_review(row["true_meta_review"])

    with left:
        st.markdown("#### 🤖 AI-generated")
        if pred_text:
            display_text = truncate_repetition(pred_text)
            st.markdown(
                f'<div class="result-card"><h4>Model output</h4><div class="content">{safe_display(display_text)}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No meta-review extracted.")

    with right:
        st.markdown("#### 👤 Human (ground truth)")
        if true_text:
            st.markdown(
                f'<div class="result-card"><h4>Area Chair</h4><div class="content">{safe_display(true_text)}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No meta-review extracted.")


if __name__ == "__main__":
    main()
