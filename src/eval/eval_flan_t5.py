#!/usr/bin/env python3
"""
Evaluate predictions CSV using evaluation.py metrics.

Metrics computed:
  - Decision: accuracy, precision, recall, f1_macro
  - Text:     rouge1/2/L, bertscore
  - Coverage: coverage@k  (needs test.jsonl for gold key-points)
  - Hallucination: supported/unsupported/contradicted rate
                   (needs test.jsonl to get original reviews per paper)
  - Consistency: decision-review consistency

Usage:
    python src/eval/eval_flan_t5.py
    python src/eval/eval_flan_t5.py --pred_path data/predictions/flan_t5_run1_predictions.csv
    python src/eval/eval_flan_t5.py --pred_path data/predictions/bart_run2_predictions.csv
    python src/eval/eval_flan_t5.py --pred_path data/predictions/pegasus_run1_predictions.csv
    python src/eval/eval_flan_t5.py --skip_bertscore --skip_hallucination   # fast mode

Output:
    data/predictions/<run_name>_eval_results.json
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ── 环境检测：自动区分 Colab 和本地 ──────────────────────────────────────────
import os
ON_COLAB = os.path.exists("/content/drive")

if ON_COLAB:
    # Colab：evaluation.py 在 Google Drive 里
    EVAL_DIR = Path("/content/drive/MyDrive/meta_review/src/eval")
    PROJECT_ROOT = Path("/content/drive/MyDrive/meta_review")
else:
    # 本地：evaluation.py 和本文件同目录
    EVAL_DIR = Path(__file__).parent
    PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(EVAL_DIR))
from evaluation import (
    compute_rouge_all,
    decision_review_consistency_rate,
    coverage_at_k,
    hallucination_rate,
)

# ── 直接修改这里来切换评估哪个模型 ───────────────────────────────────────────
PRED_FILE          = "data/predictions/flan_t5_run1_predictions.csv"
TEST_FILE          = "data/processed/test.jsonl"
SKIP_BERTSCORE     = False   # True = 跳过 BERTScore（本地慢，Colab T4 较快）
SKIP_HALLUCINATION = True    # True = 跳过 Coverage@K 和 Hallucination（慢）
COVERAGE_K         = 5
THRESHOLD          = 0.80

# ── CLI 参数（终端运行时可覆盖上面的变量，直接运行时忽略）────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pred_path",          type=str,   default=PRED_FILE)
parser.add_argument("--test_path",          type=str,   default=TEST_FILE)
parser.add_argument("--skip_bertscore",     action="store_true", default=SKIP_BERTSCORE)
parser.add_argument("--skip_hallucination", action="store_true", default=SKIP_HALLUCINATION)
parser.add_argument("--coverage_k",         type=int,   default=COVERAGE_K)
parser.add_argument("--threshold",          type=float, default=THRESHOLD)
args = parser.parse_args()

# 统一用绝对路径，无论从哪个目录运行都能找到文件
PRED_PATH = PROJECT_ROOT / args.pred_path
TEST_PATH = PROJECT_ROOT / args.test_path
run_name  = PRED_PATH.stem
OUT_PATH  = PRED_PATH.parent / f"{run_name}_eval_results.json"

assert PRED_PATH.exists(), f"Predictions file not found: {PRED_PATH}"

# ── Helper: parse individual review texts from input_text ─────────────────────
def parse_reviews_from_input(input_text: str) -> List[str]:
    """
    input_text 里 reviews 段的格式：
        REVIEWS:

        REVIEW 1:
        FinalRating: ...
        Summary: ...
        Strengths: ...
        Weaknesses: ...

        REVIEW 2:
        ...

    用正则按 "REVIEW N:" 分割，返回每条 review 的完整文本列表。
    """
    # 只取 REVIEWS: 之后的部分
    match = re.search(r'\nREVIEWS:\n', input_text)
    if not match:
        return [input_text]
    reviews_block = input_text[match.end():]

    # 按 "REVIEW N:\n" 分割
    parts = re.split(r'\nREVIEW \d+:\n', reviews_block)
    reviews = [p.strip() for p in parts if p.strip()]
    return reviews if reviews else [reviews_block]


# ── Load predictions CSV ───────────────────────────────────────────────────────
print(f"Loading predictions: {PRED_PATH}")
with open(PRED_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"  {len(rows)} samples")

gold_decisions = [r["true_decision"]    for r in rows]
pred_decisions = [r["pred_decision"]    for r in rows]
gold_reviews   = [r["true_meta_review"] for r in rows]
pred_reviews   = [r["pred_meta_review"] for r in rows]
paper_ids      = [r["paper_id"]         for r in rows]

# ── Load test.jsonl → reviews per paper ───────────────────────────────────────
reviews_per_paper: Dict[str, List[str]] = {}
if TEST_PATH.exists():
    print(f"Loading test.jsonl: {TEST_PATH}")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            reviews_per_paper[ex["paper_id"]] = parse_reviews_from_input(ex["input_text"])
    print(f"  {len(reviews_per_paper)} papers with parsed reviews")
else:
    print(f"WARNING: test.jsonl not found at {TEST_PATH} — skipping coverage@k and hallucination")
    args.skip_hallucination = True

# ── 1. Decision classification ────────────────────────────────────────────────
print("\n📊 Decision metrics ...")

def to_binary(lbl: str) -> int:
    return 1 if str(lbl).strip().upper().startswith("ACCEPT") else 0

y_true = [to_binary(d) for d in gold_decisions]
y_pred = [to_binary(d) for d in pred_decisions]

unknown_count = sum(
    1 for p in pred_decisions
    if not str(p).strip().upper().startswith(("ACCEPT", "REJECT"))
)

decision_metrics = {
    "accuracy":        float(accuracy_score(y_true, y_pred)),
    "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "unknown_count":   unknown_count,
    "total_samples":   len(rows),
}
for k, v in decision_metrics.items():
    print(f"  {k}: {v}")

# ── 2. ROUGE ──────────────────────────────────────────────────────────────────
print("\n📊 ROUGE ...")
r1, r2, rl = [], [], []
for gen, gold in tqdm(zip(pred_reviews, gold_reviews), total=len(rows)):
    s = compute_rouge_all(gen, gold)
    r1.append(s["rouge1"])
    r2.append(s["rouge2"])
    rl.append(s["rougeL"])

rouge_metrics = {
    "rouge1":    float(np.mean(r1)),
    "rouge2":    float(np.mean(r2)),
    "rougeL":    float(np.mean(rl)),
    "rougeLsum": float(np.mean(rl)),
}
for k, v in rouge_metrics.items():
    print(f"  {k}: {v:.4f}")

# ── 3. BERTScore ──────────────────────────────────────────────────────────────
bertscore_metrics = {}
if not args.skip_bertscore:
    print("\n📊 BERTScore ...")
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(pred_reviews, gold_reviews, lang="en", verbose=False)
    bertscore_metrics = {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall":    float(R.mean()),
        "bertscore_f1":        float(F1.mean()),
    }
    for k, v in bertscore_metrics.items():
        print(f"  {k}: {v:.4f}")
else:
    print("\n⏭️  Skipping BERTScore")

# ── 4. Coverage@K & Hallucination ─────────────────────────────────────────────
coverage_metrics     = {}
hallucination_metrics = {}

if not args.skip_hallucination:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型只加载一次，所有样本共用
    print("\n📊 Loading SentenceTransformer ...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"📊 Loading NLI model (facebook/bart-large-mnli) on {device} ...")
    nli_tokenizer = HFTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model     = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).to(device)
    nli_model.eval()

    cov_scores = []
    supp_list, unsupp_list, contrad_list = [], [], []

    print("\n📊 Coverage@K & Hallucination (per sample) ...")
    for i in tqdm(range(len(rows))):
        pid  = paper_ids[i]
        gen  = pred_reviews[i]
        gold = gold_reviews[i]
        revs = reviews_per_paper.get(pid, [gen])  # fallback: 用自身

        # Coverage@K：生成的 meta-review 覆盖了 gold 的多少关键点
        cov = coverage_at_k(
            gen, gold,
            k=args.coverage_k,
            threshold=args.threshold,
            emb_model=emb_model,
        )
        cov_scores.append(cov)

        # Hallucination：生成内容有多少能被原始 reviews 支撑
        s, u, c = hallucination_rate(
            gen, revs,
            threshold=args.threshold,
            emb_model=emb_model,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model,
        )
        supp_list.append(s)
        unsupp_list.append(u)
        contrad_list.append(c)

    coverage_metrics = {
        f"coverage@{args.coverage_k}": float(np.mean(cov_scores)),
    }
    hallucination_metrics = {
        "hallucination_supported_rate":    float(np.mean(supp_list)),
        "hallucination_unsupported_rate":  float(np.mean(unsupp_list)),
        "hallucination_contradicted_rate": float(np.mean(contrad_list)),
    }

    for k, v in {**coverage_metrics, **hallucination_metrics}.items():
        print(f"  {k}: {v:.4f}")
else:
    print("\n⏭️  Skipping Coverage@K & Hallucination")

# ── 5. Decision-Review Consistency ────────────────────────────────────────────
print("\n📊 Decision-review consistency ...")
consistency = decision_review_consistency_rate(pred_reviews, pred_decisions)
print(f"  decision_review_consistency: {consistency:.4f}")

# ── 6. Save results ───────────────────────────────────────────────────────────
results = {
    **decision_metrics,
    **rouge_metrics,
    **bertscore_metrics,
    **coverage_metrics,
    **hallucination_metrics,
    "decision_review_consistency": consistency,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved to {OUT_PATH}")
print(json.dumps(results, indent=2))
