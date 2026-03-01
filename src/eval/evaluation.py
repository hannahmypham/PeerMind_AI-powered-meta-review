"""
Evaluation metrics for LLM-generated meta-review vs gold meta-review.
Each metric is a separate function.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _parse_decision(val) -> int:
    """
    Robustly map decision label to int: 1 = accept, 0 = reject, -1 = unknown.
    Accepts: 'ACCEPT', 'REJECT', '1', '0', 1, 0, '' (empty), None.
    """
    if val is None:
        return -1
    s = str(val).strip().upper()
    if s in ("ACCEPT", "1"):
        return 1
    if s in ("REJECT", "0"):
        return 0
    return -1  # empty string or unknown format


# ============== 1. ROUGE ==============
def rouge_1(generated: str, gold: str) -> float:
    """ROUGE-1 F1 score (unigram overlap)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(gold, generated)
    return scores["rouge1"].fmeasure


def rouge_2(generated: str, gold: str) -> float:
    """ROUGE-2 F1 score (bigram overlap)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    scores = scorer.score(gold, generated)
    return scores["rouge2"].fmeasure


def rouge_l(generated: str, gold: str) -> float:
    """ROUGE-L F1 score (longest common subsequence)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold, generated)
    return scores["rougeL"].fmeasure


def compute_rouge_all(generated: str, gold: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L in one call."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ============== 2. BERTScore ==============
def bertscore(generated: str, gold: str, lang: str = "en") -> float:
    """BERTScore F1 (semantic similarity)."""
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn([generated], [gold], lang=lang, verbose=False)
    return float(F1[0])


# ============== 3. Coverage@K ==============
def coverage_at_k(
    generated: str,
    gold: str,
    k: int = 5,
    threshold: float = 0.80,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    emb_model=None,
) -> float:
    """
    Key-point coverage: fraction of gold key points covered by generated text.
    Uses rule-based extraction + embedding similarity.

    Pass a pre-loaded SentenceTransformer as `emb_model` to avoid reloading it
    on every call when evaluating a large batch.
    """
    from sentence_transformers import SentenceTransformer, util
    import re

    def extract_key_points(text: str, k: int) -> List[str]:
        """Simple rule-based: split by sentence, take first k non-trivial ones."""
        sents = re.split(r"[.!?]\s+", text.strip())
        sents = [s.strip() for s in sents if len(s.strip()) >= 10][:k]
        return sents if sents else [text[:200]]  # fallback

    gold_pts = extract_key_points(gold, k)
    gen_pts = extract_key_points(generated, k + 2)  # allow more from generated

    if not gold_pts:
        return 1.0

    model = emb_model if emb_model is not None else SentenceTransformer(model_name)
    E_gold = model.encode(gold_pts, convert_to_tensor=True)
    E_gen = model.encode(gen_pts, convert_to_tensor=True)

    # cosine similarity matrix: shape (len(gold_pts), len(gen_pts))
    sim_matrix = util.cos_sim(E_gold, E_gen).cpu().numpy()

    matched = 0
    used_gen = set()
    for j in range(len(gold_pts)):
        best_sim, best_t = -1.0, -1
        for t in range(len(gen_pts)):
            if t in used_gen:
                continue
            s = float(sim_matrix[j, t])
            if s > best_sim:
                best_sim, best_t = s, t
        if best_sim >= threshold and best_t >= 0:
            matched += 1
            used_gen.add(best_t)
    return matched / len(gold_pts)


# ============== 4. Hallucination (Consistency with reviews) ==============
def _nli_entailment(premise: str, hypothesis: str, tokenizer, model, device) -> str:
    """
    NLI: returns 'entailment', 'neutral', or 'contradiction'.
    premise = evidence from reviews, hypothesis = claim from generated meta-review.
    """
    import torch
    # BART-MNLI format: premise + hypothesis (tokenizer handles separator)
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # BART-MNLI: 0=contradiction, 1=neutral, 2=entailment
    pred = logits.argmax(dim=1).item()
    return ["contradiction", "neutral", "entailment"][pred]


def hallucination_rate(
    generated: str,
    reviews: List[str],
    threshold: float = 0.80,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model_name: str = "facebook/bart-large-mnli",
    emb_model=None,
    nli_tokenizer=None,
    nli_model=None,
) -> Tuple[float, float, float]:
    """
    Returns (supported_rate, unsupported_rate, contradicted_rate).
    Extracts claims from generated meta-review, checks if supported by reviews.
    Uses NLI to distinguish supported / unsupported / contradicted.

    Pass pre-loaded `emb_model`, `nli_tokenizer`, `nli_model` to avoid
    reloading heavy models on every call when evaluating a large batch.
    """
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import re
    import torch

    def extract_claims(text: str) -> List[str]:
        sents = re.split(r"[.!?]\s+", text.strip())
        return [s.strip() for s in sents if len(s.strip()) >= 15]

    claims = extract_claims(generated)
    if not claims:
        return 1.0, 0.0, 0.0

    _emb_model = emb_model if emb_model is not None else SentenceTransformer(model_name)

    all_review_sents = []
    for r in reviews:
        all_review_sents.extend(re.split(r"[.!?]\s+", r.strip()))
    all_review_sents = [s.strip() for s in all_review_sents if len(s.strip()) >= 10]
    if not all_review_sents:
        return 0.0, 1.0, 0.0  # no evidence -> treat as unsupported

    E_claims = _emb_model.encode(claims, convert_to_tensor=True)
    E_reviews = _emb_model.encode(all_review_sents, convert_to_tensor=True)

    # cosine similarity matrix: shape (len(claims), len(all_review_sents))
    sim_matrix = util.cos_sim(E_claims, E_reviews).cpu().numpy()

    # Load NLI model only if not provided
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nli_tokenizer = nli_tokenizer
    _nli_model = nli_model
    if _nli_tokenizer is None or _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        _nli_model.eval()

    supported, unsupported, contradicted = 0, 0, 0
    for j in range(len(claims)):
        sims = sim_matrix[j]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < threshold:
            unsupported += 1
        else:
            premise = all_review_sents[best_idx]
            hypothesis = claims[j]
            label = _nli_entailment(premise, hypothesis, _nli_tokenizer, _nli_model, device)
            if label == "entailment":
                supported += 1
            elif label == "contradiction":
                contradicted += 1
            else:
                unsupported += 1

    n = len(claims)
    return supported / n, unsupported / n, contradicted / n


# ============== 5. Decision–Review Consistency ==============
def decision_review_consistency(
    generated_review: str,
    predicted_decision: Union[int, str],  # 0 = reject, 1 = accept
) -> bool:
    """
    Check if predicted decision (0=reject, 1=accept) is consistent with the tone of generated review.
    Simple heuristic: Reject -> review should have negative cues; Accept -> more positive.
    """
    pred_val = _parse_decision(predicted_decision)
    rev = generated_review.strip().lower() if generated_review else ""

    neg_cues = ["reject", "weak", "insufficient", "lacks", "missing", "concern", "issue", "flaw", "limited", "not novel"]
    pos_cues = ["accept", "strong", "contribution", "novel", "well-written", "convincing", "solid"]

    neg_count = sum(1 for c in neg_cues if c in rev)
    pos_count = sum(1 for c in pos_cues if c in rev)

    if pred_val == 0:  # Reject
        return neg_count >= pos_count or neg_count >= 1
    else:  # Accept (1 or other)
        return pos_count >= neg_count or pos_count >= 1


def decision_review_consistency_rate(
    generated_reviews: List[str],
    predicted_decisions: List[str],
) -> float:
    """Fraction of samples where decision and review are consistent."""
    consistent = [
        decision_review_consistency(rev, dec)
        for rev, dec in zip(generated_reviews, predicted_decisions)
    ]
    return sum(consistent) / len(consistent) if consistent else 0.0


# ============== 6. Final Decision Macro-F1 ==============
def decision_macro_f1(
    gold_decisions: List[Union[int, str]],  # 0 = reject, 1 = accept
    predicted_decisions: List[Union[int, str]],
) -> float:
    """
    Macro-F1 for binary Accept/Reject prediction.
    Input: 0 or 1 (int/str). 0 = reject, 1 = accept.
    """
    def to_binary(lbl) -> int:
        v = _parse_decision(lbl)
        return 1 if v == 1 else 0

    y_true = [to_binary(g) for g in gold_decisions]
    y_pred = [to_binary(p) for p in predicted_decisions]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ============== Batch evaluation ==============
def evaluate_all(
    generated_reviews: List[str],
    gold_meta_reviews: List[str],
    reviews_list: List[List[str]],  # list of review texts per sample
    gold_decisions: List[str],
    predicted_decisions: List[str],
    k: int = 5,
    coverage_threshold: float = 0.80,
    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model_name: str = "facebook/bart-large-mnli",
) -> Dict[str, float]:
    """
    Run all metrics on a batch of samples.

    Models (SentenceTransformer, BART-MNLI) are loaded once and reused across
    all samples for efficiency.
    """
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from bert_score import score as bert_score_fn
    import torch
    from tqdm import tqdm

    n = len(generated_reviews)
    assert n == len(gold_meta_reviews) == len(reviews_list) == len(gold_decisions) == len(predicted_decisions)

    # --- Load heavy models once ---
    print("Loading SentenceTransformer...")
    emb_model = SentenceTransformer(emb_model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NLI model ({nli_model_name}) on {device}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model_obj = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    nli_model_obj.eval()

    # --- BERTScore: batch call (much faster than per-sample) ---
    print("Computing BERTScore (batch)...")
    _, _, F1 = bert_score_fn(generated_reviews, gold_meta_reviews, lang="en", verbose=False)
    bs = F1.tolist()

    # --- Per-sample metrics ---
    r1, r2, rl, cov, supp, unsupp, contrad = [], [], [], [], [], [], []
    for i in tqdm(range(n), desc="Per-sample metrics"):
        rouge_scores = compute_rouge_all(generated_reviews[i], gold_meta_reviews[i])
        r1.append(rouge_scores["rouge1"])
        r2.append(rouge_scores["rouge2"])
        rl.append(rouge_scores["rougeL"])

        cov.append(coverage_at_k(
            generated_reviews[i], gold_meta_reviews[i],
            k=k, threshold=coverage_threshold, emb_model=emb_model,
        ))

        s, u, c = hallucination_rate(
            generated_reviews[i], reviews_list[i],
            threshold=coverage_threshold,
            emb_model=emb_model,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model_obj,
        )
        supp.append(s)
        unsupp.append(u)
        contrad.append(c)

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
        "bertscore": float(np.mean(bs)),
        "coverage@k": float(np.mean(cov)),
        "hallucination_supported_rate": float(np.mean(supp)),
        "hallucination_unsupported_rate": float(np.mean(unsupp)),
        "hallucination_contradicted_rate": float(np.mean(contrad)),
        "decision_review_consistency": decision_review_consistency_rate(generated_reviews, predicted_decisions),
        "decision_macro_f1": decision_macro_f1(gold_decisions, predicted_decisions),
    }
