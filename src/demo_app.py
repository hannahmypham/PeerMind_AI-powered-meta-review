#!/usr/bin/env python3
"""
Streamlit demo for Meta-Review AI (Pegasus model).

Run: streamlit run src/demo_app.py

Requires: trained Pegasus LoRA checkpoint at pegasus_large_lora/ (or set MODEL_PATH)
"""

import json
from pathlib import Path

import streamlit as st
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "pegasus_large_lora"
TEST_JSONL = PROJECT_ROOT / "data" / "processed" / "test.jsonl"

# Truncation limits (match build_dataset.py)
TRUNC = {"title": 250, "abstract": 1200, "summary": 900, "strengths": 700, "weaknesses": 700, "questions": 500}


def trunc(s: str, max_len: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= max_len else s[: max_len - 3].rstrip() + "..."


def build_input_text(paper_id: str, title: str, abstract: str, reviews: list[dict]) -> str:
    """Build prompt in same format as training."""
    parts = [
        "You are a senior ICLR meta-reviewer.\n"
        "Based on the paper and reviews, write the final meta-review and decision.\n\n"
        "You MUST follow this exact format:\n"
        "DECISION: <ACCEPT or REJECT>\n"
        "META_REVIEW:\n"
        "<your full meta-review>\n"
        "Do not output anything outside this format.\n",
        f"PAPER ID:\n{paper_id}\n",
        f"PAPER TITLE:\n{trunc(title, TRUNC['title']) or 'N/A'}\n",
        f"ABSTRACT:\n{trunc(abstract, TRUNC['abstract']) or 'N/A'}\n",
    ]

    ratings = [r.get("rating") for r in reviews if r.get("rating") is not None]
    parts.append("AGGREGATES:")
    parts.append(f"- NumReviews: {len(reviews)}")
    if ratings:
        parts.append(f"- MeanRating: {sum(ratings) / len(ratings):.2f}")
        parts.append(f"- RatingRange: {min(ratings):.0f} to {max(ratings):.0f}")
    parts.append("")
    parts.append("REVIEWS:\n")

    for i, r in enumerate(reviews, 1):
        parts.append(f"REVIEW {i}:")
        if r.get("rating") is not None:
            parts.append(f"FinalRating: {r['rating']}")
        for key in ["summary", "strengths", "weaknesses", "questions"]:
            val = trunc(r.get(key, ""), TRUNC.get(key, 700))
            if val:
                parts.append(f"{key.title()}:\n{val}")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def extract_decision(text: str) -> str:
    text = (text or "").strip()
    if "DECISION:" in text:
        after = text.split("DECISION:", 1)[1].strip()
        first = after.split("\n", 1)[0].strip().upper()
        if first.startswith("ACCEPT"):
            return "ACCEPT"
        if first.startswith("REJECT"):
            return "REJECT"
        return first
    return "UNKNOWN"


def extract_meta_review(text: str) -> str:
    text = (text or "").strip()
    if "META_REVIEW:" in text:
        return text.split("META_REVIEW:", 1)[1].strip()
    return text


@st.cache_resource
def load_model(model_path: Path):
    """Load Pegasus + LoRA adapter (cached)."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train Pegasus via pegasus-large_lora_train_and_generate.ipynb first, "
            "or set MODEL_PATH to your checkpoint folder."
        )
    peft_config = PeftConfig.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    base = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, str(model_path))
    model = model.merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device: str, input_text: str) -> str:
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            num_beams=4,
            length_penalty=0.8,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    st.set_page_config(page_title="Meta-Review AI Demo", page_icon="📝", layout="wide")
    st.title("📝 Meta-Review AI Demo")
    st.caption("Generate ICLR-style meta-reviews and accept/reject decisions using Pegasus-Large + LoRA")

    model_path = st.sidebar.text_input(
        "Model path",
        value=str(MODEL_PATH),
        help="Path to Pegasus LoRA checkpoint (e.g. pegasus_large_lora)",
    )
    model_path = Path(model_path)

    mode = st.radio("Input mode", ["Custom input", "Load from test set"], horizontal=True)

    if mode == "Load from test set":
        if not TEST_JSONL.exists():
            st.warning(f"`{TEST_JSONL}` not found. Use Custom input or run build_dataset.py first.")
            st.stop()
        else:
            with open(TEST_JSONL, encoding="utf-8") as f:
                lines = f.readlines()
            idx = st.selectbox("Select paper", range(len(lines)), format_func=lambda i: f"Paper {i+1}")
            ex = json.loads(lines[idx])
            paper_id = ex["paper_id"]
            input_text = ex["input_text"]
            st.subheader(f"Paper: {paper_id}")
            with st.expander("View input prompt"):
                st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
    else:
        st.subheader("Custom input")
        paper_id = st.text_input("Paper ID", value="demo_001")
        title = st.text_area("Paper title", placeholder="e.g. Attention Is All You Need")
        abstract = st.text_area("Abstract", placeholder="Paste the paper abstract...", height=150)

        num_reviews = st.number_input("Number of reviews", min_value=1, max_value=8, value=2)
        reviews = []
        for i in range(int(num_reviews)):
            with st.expander(f"Review {i+1}"):
                r = {
                    "rating": st.number_input("Rating (1-10)", min_value=1, max_value=10, value=6, key=f"r{i}"),
                    "summary": st.text_area("Summary", key=f"s{i}"),
                    "strengths": st.text_area("Strengths", key=f"str{i}"),
                    "weaknesses": st.text_area("Weaknesses", key=f"w{i}"),
                    "questions": st.text_area("Questions", key=f"q{i}"),
                }
                reviews.append(r)

        input_text = build_input_text(paper_id, title, abstract, reviews)
        with st.expander("View prompt"):
            st.text(input_text[:1500] + ("..." if len(input_text) > 1500 else ""))

    if st.button("Generate meta-review", type="primary"):
        try:
            model, tokenizer, device = load_model(model_path)
            with st.spinner("Generating..."):
                pred = generate(model, tokenizer, device, input_text)
            decision = extract_decision(pred)
            meta_review = extract_meta_review(pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Decision", decision)
            with col2:
                st.caption(f"Device: {device}")

            st.subheader("Meta-review")
            st.markdown(meta_review)

            with st.expander("Raw model output"):
                st.text(pred)
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
