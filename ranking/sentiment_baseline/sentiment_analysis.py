import pandas as pd

# --------- Load ----------
df = pd.read_csv("flan_t5_run1_predictions.csv")

required_cols = ["paper_id", "true_decision", "pred_decision", "true_meta_review", "pred_meta_review"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

# ---------- Keyword lists (binary presence; NO decision boosting) ----------
POS = [
    "rigorous", "sound", "technically sound", "well-justified", "well supported",
    "solid", "robust", "careful", "principled", "theoretically grounded", "formal", "proof",
    "strong empirical", "empirically", "comprehensive", "thorough", "well-designed",
    "well controlled", "well validated", "reproducible", "replicable",
    "novel", "novelty", "original", "innovative", "new perspective", "fresh",
    "insightful", "insight", "interesting", "surprising", "creative",
    "significant", "important", "impactful", "high impact", "meaningful",
    "substantial", "useful", "practical", "broadly applicable", "generalizable",
    "strong contribution", "major contribution", "valuable",
    "clear", "well-written", "well written", "well-organized", "well organized",
    "easy to follow", "excellent writing", "good writing", "good presentation",
    "state-of-the-art", "state of the art", "sota", "outperforms", "beats",
    "strong results", "consistent improvements", "large gains", "improves",
    "competitive", "effective", "efficient", "scalable",
    "extensive experiments", "ablation", "ablations", "error analysis",
    "robustness", "robustness analysis"
]

NEG = [
    "fatal flaw", "critical flaw", "fundamental flaw",
    "major concern", "major concerns", "serious concern", "serious concerns",
    "deal breaker", "invalid", "incorrect", "wrong", "bug", "broken",
    "not rigorous", "lack rigor", "insufficient", "insufficient evidence",
    "weak evidence", "unsubstantiated", "unsupported", "overclaim", "overclaims",
    "exaggerated", "hand-wavy", "handwavy", "missing proof", "no proof",
    "missing baseline", "missing baselines", "weak baseline", "no baseline",
    "incomplete evaluation", "limited evaluation", "small dataset", "tiny dataset",
    "no ablation", "missing ablation", "no ablations", "missing ablations",
    "no comparison", "missing comparison", "unfair comparison",
    "cherry-picked", "cherrypicked",
    "not novel", "limited novelty", "incremental", "minor contribution",
    "low impact", "unclear contribution", "unclear novelty", "marginal",
    "unclear", "confusing", "hard to follow", "poorly written", "poor writing",
    "missing details", "insufficient details", "not enough detail",
    "cannot reproduce", "not reproducible", "not replicable", "lack of details",
    "ambiguous", "vague",
    "questionable", "dubious", "unstable", "does not work", "fails to",
    "doesn't work", "failure", "collapses",
    "weak", "limited", "concern", "concerns", "flaw", "lack", "missing",
    "not convincing", "unconvincing"
]

BORDERLINE_NEG = ["borderline", "mixed", "uncertain", "not sure", "on the fence"]
BORDERLINE_POS = ["promising", "potential", "could be", "worth considering"]

def score_meta_review(text: str) -> float:
    t = (text or "").lower()
    pos = sum(1 for w in POS if w in t)
    neg = sum(1 for w in NEG if w in t)

    major_terms = [
        "fatal flaw", "critical flaw", "fundamental flaw",
        "major concern", "major concerns", "serious concern", "serious concerns"
    ]
    minor_terms = ["minor concern", "minor concerns", "small concern", "small concerns"]

    major_pen = sum(1 for w in major_terms if w in t)
    minor_pen = sum(1 for w in minor_terms if w in t)

    borderline = 0.85 if any(w in t for w in BORDERLINE_NEG) else 1.0
    promising  = 1.05 if any(w in t for w in BORDERLINE_POS) else 1.0

    length = min(len(t) / 2000, 1.0)

    base = 0.22 * pos - 0.35 * neg - 1.0 * major_pen - 0.25 * minor_pen + 0.05 * length
    return base * borderline * promising

# --------- Compute scores ----------
df["true_meta_review_sentiment_score"] = df["true_meta_review"].fillna("").map(score_meta_review)
df["pred_meta_review_sentiment_score"] = df["pred_meta_review"].fillna("").map(score_meta_review)

# --------- Filter to ACCEPTED papers first (using pred_decision) ----------
# If pred_decision values are exactly "ACCEPT"/"REJECT", this works.
# If they include extra text, this still works because it checks substring.
accepted = df[df["pred_decision"].astype(str).str.upper().str.contains("ACCEPT", na=False)].copy()

# -------------------
accepted_out = accepted[[
    "paper_id",
    "true_decision",
    "pred_decision",
    "true_meta_review_sentiment_score",
    "pred_meta_review_sentiment_score",
    "true_meta_review",
    "pred_meta_review",

]]

# --------- Rank accepted papers by predicted score and take top 30 ----------
top30 = accepted_out.sort_values("pred_meta_review_sentiment_score", ascending=False).head(30)

out_path = "top_30_pegasus_ACCEPTED_then_ranked_by_pred_score.csv"
top30.to_csv(out_path, index=False)

print("Saved:", out_path)
print(top30[[
    "paper_id",
    "true_decision",
    "pred_decision",
    "true_meta_review_sentiment_score",
    "pred_meta_review_sentiment_score",
]].to_string(index=False))