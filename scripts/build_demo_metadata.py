#!/usr/bin/env python3
"""
Build a small paper metadata CSV for Streamlit deployment.

Reads paper_ids from the predictions file, extracts title + abstract from the
raw CSV, and writes data/demo/paper_metadata.csv. This file is small enough
to commit so the deployed demo shows paper titles and abstracts.

Run locally before deploying:
  python scripts/build_demo_metadata.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "pegasus_x_meta_review_predictions.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "iclr_2025_detailed_reviews.csv"
OUT_DIR = PROJECT_ROOT / "data" / "demo"
OUT_PATH = OUT_DIR / "paper_metadata.csv"


def main():
    if not PRED_PATH.exists():
        print(f"Predictions not found: {PRED_PATH}")
        return 1
    if not RAW_PATH.exists():
        print(f"Raw data not found: {RAW_PATH}. Run OpenReviewDataExtract.py first.")
        return 1

    pred_df = pd.read_csv(PRED_PATH)
    paper_ids = set(pred_df["paper_id"].astype(str).tolist())

    raw_df = pd.read_csv(RAW_PATH)
    raw_df["paper_id"] = raw_df["paper_id"].astype(str)

    rows = []
    for paper_id in paper_ids:
        g = raw_df[raw_df["paper_id"] == paper_id]
        if g.empty:
            rows.append({"paper_id": paper_id, "title": "N/A", "abstract": "N/A"})
            continue
        row = g.iloc[0]
        title = row.get("title", "N/A")
        abstract = row.get("abstract", "N/A")
        if pd.isna(title) or not str(title).strip():
            title = "N/A"
        else:
            title = str(title).strip()
        if pd.isna(abstract) or not str(abstract).strip():
            abstract = "N/A"
        else:
            abstract = str(abstract).strip()
        rows.append({"paper_id": paper_id, "title": title, "abstract": abstract})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(rows)} papers, ~{OUT_PATH.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    exit(main())
