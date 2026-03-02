#!/usr/bin/env python3
"""
Streamlit demo for Peermind (Meta-Review AI) — CSV-only (no model required).

Run:
  streamlit run src/demo_csv.py
"""

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "predictions" / "pegasus_x_meta_review_predictions.csv"
RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw"
LOGO_PATH = PROJECT_ROOT / "assets" / "peermind_logo.png"

# Make logo look good in header: give it a fixed "card" and a larger, crisp size.
LOGO_PX = 96  # try 88–110 for best look on projectors

# CSS must come after set_page_config (which is first in main())
CSS = f"""
<style>
    /* Header row: logo card + hero card */
    .header-row {{
        display: flex;
        align-items: stretch;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    /* Logo card */
    .logo-card {{
        width: 120px;
        padding: 0.9rem 0.8rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(6px);
    }}
    .logo-card img {{
        width: {LOGO_PX}px;
        height: {LOGO_PX}px;
        object-fit: contain;
        image-rendering: -webkit-optimize-contrast;
        filter: drop-shadow(0 6px 18px rgba(0,0,0,0.35));
    }}

    /* Hero */
    .hero {{
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.30);
        border: 1px solid rgba(255,255,255,0.10);
    }}
    .hero h1 {{ font-size: 2.2rem; font-weight: 800; margin: 0; }}
    .hero p {{ font-size: 1.05rem; opacity: 0.95; margin: 0.55rem 0 0 0; }}

    /* Decision badges */
    .decision-accept {{
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }}
    .decision-reject {{
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 700; font-size: 1rem;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }}
    .decision-unknown {{
        display: inline-block; padding: 0.5rem 1rem; border-radius: 10px;
        font-weight: 600; background: #e5e7eb; color: #4b5563;
    }}

    /* Cards */
    .result-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }}
    .result-card h4 {{
        font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #6366f1; margin-bottom: 0.75rem;
    }}
    .result-card .content {{
        font-size: 0.95rem; line-height: 1.7; color: #374151;
    }}

    /* Match badge */
    .match-yes {{
        display: inline-block; padding: 0.4rem 0.9rem; border-radius: 8px;
        font-weight: 600; background: #d1fae5; color: #065f46;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }}
    .match-no {{
        display: inline-block; padding: 0.4rem 0.9rem; border-radius: 8px;
        font-weight: 600; background: #fee2e2; color: #991b1b;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }}

    /* Hide Streamlit branding for cleaner demo */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Mobile/Small screen fallback */
    @media (max-width: 900px) {{
        .header-row {{
            flex-direction: column;
        }}
        .logo-card {{
            width: 100%;
            justify-content: center;
        }}
    }}
</style>
"""


def _safe(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def extract_meta_review(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if "META_REVIEW:" in text:
        return text.split("META_REVIEW:", 1)[1].strip()
    return text


def truncate_repetition(text: str) -> str:
    """Truncate obvious repetition for display (presentation guardrail)."""
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


def safe_display(text: str) -> str:
    return html.escape(text).replace("\n", "<br>") if text else ""


def normalize_decision(x) -> str:
    """
    Normalize decision strings so your metrics and match badge don't get wrecked
    by casing/whitespace/variants. Keeps it simple for class demo.
    """
    s = _safe(x).upper()
    if not s:
        return "UNKNOWN"
    # Common variants
    if "ACCEPT" in s:
        return "ACCEPT"
    if "REJECT" in s:
        return "REJECT"
    return "UNKNOWN"


def decision_badge(decision_norm: str) -> str:
    d = (decision_norm or "").upper()
    cls = "decision-accept" if d == "ACCEPT" else "decision-reject" if d == "REJECT" else "decision-unknown"
    label = "ACCEPT" if d == "ACCEPT" else "REJECT" if d == "REJECT" else "—"
    return f'<span class="{cls}">{label}</span>'


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize paper_id for consistent joins
    df["paper_id"] = df["paper_id"].astype(str)

    # Normalized decisions for metrics + matching
    df["true_dec_norm"] = df.get("true_decision", pd.Series(dtype=object)).apply(normalize_decision)
    df["pred_dec_norm"] = df.get("pred_decision", pd.Series(dtype=object)).apply(normalize_decision)

    return df


@st.cache_data(show_spinner=False)
def load_paper_metadata(raw_dir: Path) -> dict:
    """
    Load paper_id -> {title, abstract, reviews} from ALL raw CSVs in data/raw/.
    This avoids the 'first file only' demo failure.
    """
    meta = {}
    if not raw_dir.exists():
        return meta

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    if not csv_files:
        return meta

    try:
        dfs = []
        for fp in csv_files:
            try:
                d = pd.read_csv(fp)
                if "paper_id" in d.columns:
                    dfs.append(d)
            except Exception:
                continue

        if not dfs:
            return meta

        df_raw = pd.concat(dfs, ignore_index=True)
        df_raw["paper_id"] = df_raw["paper_id"].astype(str)

        for paper_id, g in df_raw.groupby("paper_id"):
            title = "N/A"
            abstract = "N/A"

            if "title" in g.columns:
                val = g["title"].iloc[0]
                if pd.notna(val) and str(val).strip():
                    title = str(val).strip()

            if "abstract" in g.columns:
                val = g["abstract"].iloc[0]
                if pd.notna(val) and str(val).strip():
                    abstract = str(val).strip()

            reviews = []
            for _, r in g.iterrows():
                reviews.append(
                    {
                        "final_rating": r.get("final_rating"),
                        "summary": _safe(r.get("summary")),
                        "strengths": _safe(r.get("strengths")),
                        "weaknesses": _safe(r.get("weaknesses")),
                        "questions": _safe(r.get("questions")),
                    }
                )

            meta[str(paper_id)] = {"title": title, "abstract": abstract, "reviews": reviews}

    except Exception:
        # Don't crash presentation; just show fewer details.
        return {}

    return meta


def jump_to_case(df: pd.DataFrame, case: str) -> str | None:
    """
    Returns a paper_id for a case type:
      - "correct_accept"
      - "correct_reject"
      - "incorrect"
    """
    if df.empty:
        return None

    correct = df["true_dec_norm"] == df["pred_dec_norm"]
    incorrect = ~correct

    if case == "incorrect":
        candidates = df[incorrect]
    elif case == "correct_accept":
        candidates = df[correct & (df["true_dec_norm"] == "ACCEPT")]
    elif case == "correct_reject":
        candidates = df[correct & (df["true_dec_norm"] == "REJECT")]
    else:
        return None

    if candidates.empty:
        return None

    # deterministic pick for class demo (stable every run)
    return str(candidates.iloc[0]["paper_id"])


def main():
    st.set_page_config(
        page_title="Peermind — Meta-Review AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Load data
    if not CSV_PATH.exists():
        st.error(f"Predictions file not found: {CSV_PATH}")
        st.info("Expected at: data/predictions/pegasus_x_meta_review_predictions.csv")
        return

    df = load_predictions(CSV_PATH)
    paper_meta = load_paper_metadata(RAW_CSV_DIR)

    # Header: hero only (logo in sidebar)
    hero_html = f"""
        <div class="hero">
            <h1>Peermind</h1>
            <p>AI powered meta-review for decision support. From peer review, to clear review, to decision.</p>
            <p style="font-size: 0.9rem; opacity: 0.85;">Pegasus-X Base · {len(df)} papers</p>
        </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # Scroll to top when a shortcut button was clicked
    if st.session_state.get("scroll_to_top"):
        st.components.v1.html(
            """
            <script>
                window.scrollTo(0, 0);
                var el = document.querySelector('[data-testid="stAppViewContainer"]');
                if (el) el.scrollTop = 0;
            </script>
            """,
            height=0,
        )
        del st.session_state["scroll_to_top"]

    # Sidebar: logo on top, then demo shortcuts + stats
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.caption("(Logo not found)")
        st.markdown("### 🎬 Demo shortcuts")
        cA, cR, cI = st.columns(3)
        if cA.button("✅ Accept", use_container_width=True):
            pid = jump_to_case(df, "correct_accept")
            if pid:
                st.session_state["selected_paper_id"] = pid
                st.session_state["scroll_to_top"] = True
        if cR.button("✅ Reject", use_container_width=True):
            pid = jump_to_case(df, "correct_reject")
            if pid:
                st.session_state["selected_paper_id"] = pid
                st.session_state["scroll_to_top"] = True
        if cI.button("❌ Wrong", use_container_width=True):
            pid = jump_to_case(df, "incorrect")
            if pid:
                st.session_state["selected_paper_id"] = pid
                st.session_state["scroll_to_top"] = True

        st.markdown("---")
        st.markdown("### 📊 Stats (normalized labels)")
        n_total = len(df)
        n_correct = int((df["true_dec_norm"] == df["pred_dec_norm"]).sum())
        acc = 100 * n_correct / n_total if n_total else 0.0

        st.metric("Accuracy", f"{acc:.1f}%")
        st.metric("Papers", n_total)

        n_accept = int((df["true_dec_norm"] == "ACCEPT").sum())
        n_reject = int((df["true_dec_norm"] == "REJECT").sum())
        st.metric("Accept (GT)", n_accept)
        st.metric("Reject (GT)", n_reject)

        correct_accept = int(((df["true_dec_norm"] == "ACCEPT") & (df["pred_dec_norm"] == "ACCEPT")).sum())
        correct_reject = int(((df["true_dec_norm"] == "REJECT") & (df["pred_dec_norm"] == "REJECT")).sum())
        st.metric("Correct Accept", correct_accept)
        st.metric("Correct Reject", correct_reject)

        if not paper_meta:
            st.warning("Raw paper metadata not loaded (titles/abstracts/reviews may be missing).")

    # --- Paper selector ---
    # Keep stable ordering for class demo
    paper_ids = df["paper_id"].astype(str).tolist()

    # If a shortcut was used, preselect it
    default_pid = st.session_state.get("selected_paper_id")
    if default_pid in set(paper_ids):
        default_index = paper_ids.index(default_pid)
    else:
        default_index = 0

    def paper_label(pid: str) -> str:
        t = paper_meta.get(pid, {}).get("title", "N/A")
        return t if t != "N/A" else pid

    selected_pid = st.selectbox(
        "Select paper",
        options=paper_ids,
        index=default_index if paper_ids else 0,
        format_func=paper_label,
    )
    st.session_state["selected_paper_id"] = selected_pid

    row = df[df["paper_id"] == selected_pid].iloc[0]
    meta = paper_meta.get(selected_pid, {})
    title = meta.get("title", "N/A")
    abstract = meta.get("abstract", "N/A")

    true_norm = row["true_dec_norm"]
    pred_norm = row["pred_dec_norm"]
    is_match = true_norm == pred_norm

    # Paper info
    st.markdown("---")
    st.markdown("### Paper")
    st.markdown(f"**{title}**")
    st.markdown(f"**Paper ID:** `{selected_pid}`")
    with st.expander("**Abstract**", expanded=True):
        if abstract and abstract != "N/A":
            st.write(abstract)
        else:
            st.caption("No abstract available for this paper.")

    # Decision row
    st.markdown("### Decision")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("**Ground truth**")
        st.markdown(decision_badge(true_norm), unsafe_allow_html=True)
    with c2:
        st.markdown("**Model prediction**")
        st.markdown(decision_badge(pred_norm), unsafe_allow_html=True)
    with c3:
        st.markdown("**Match**")
        cls = "match-yes" if is_match else "match-no"
        txt = "✓ Correct" if is_match else "✗ Incorrect"
        st.markdown(f'<span class="{cls}">{txt}</span>', unsafe_allow_html=True)

    # Meta-review
    st.markdown("---")
    st.markdown("### Meta-review")
    st.caption("Model output is shown side-by-side with the human (Area Chair) meta-review.")

    left, right = st.columns(2)
    pred_text = extract_meta_review(row.get("pred_meta_review", ""))
    true_text = extract_meta_review(row.get("true_meta_review", ""))

    with left:
        st.markdown("#### 🤖 AI-generated")
        if pred_text:
            display_text = truncate_repetition(pred_text)
            st.markdown(
                f'<div class="result-card"><h4>Model output</h4><div class="content">{safe_display(display_text)}</div></div>',
                unsafe_allow_html=True,
            )
            if "_[Repetitive output truncated]_" in display_text:
                st.warning("Displayed output was truncated due to repetition (presentation guardrail).")
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

    # Reviews
    reviews = meta.get("reviews", [])
    if reviews:
        st.markdown("---")
        st.markdown(f"### Reviews ({len(reviews)})")
        for i, rev in enumerate(reviews, 1):
            rating = rev.get("final_rating")
            try:
                rating_str = f" — Rating: {int(float(rating))}" if pd.notna(rating) and str(rating).strip() else ""
            except (ValueError, TypeError):
                rating_str = ""

            st.markdown(f"#### Review {i}{rating_str}")

            if rev.get("summary"):
                st.markdown("**Summary:**")
                st.write(rev["summary"])
            if rev.get("strengths"):
                st.markdown("**Strengths:**")
                st.write(rev["strengths"])
            if rev.get("weaknesses"):
                st.markdown("**Weaknesses:**")
                st.write(rev["weaknesses"])
            if rev.get("questions"):
                st.markdown("**Questions:**")
                st.write(rev["questions"])

            st.markdown("---")
    else:
        st.markdown("---")
        st.caption("No review text loaded from raw CSVs for this paper.")


if __name__ == "__main__":
    main()
