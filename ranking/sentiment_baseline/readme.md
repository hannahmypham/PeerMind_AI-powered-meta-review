
## Meta-review scoring + top-30 selection pipeline

### Inputs

For each paper, the input CSV contains:

* `paper_id`
* `true_decision` (ground truth: ACCEPT/REJECT)
* `pred_decision` (model decision)
* `true_meta_review` (gold meta-review text)
* `pred_meta_review` (model-generated meta-review text)

### Step 1 — Compute a sentiment-style score for each meta-review

We assign a scalar **sentiment score** to both the **true** and **predicted** meta-review using the same lightweight rubric:

* Maintain two keyword sets:

  * **POS keywords** (signals of strength): rigor (“rigorous”, “robust”), novelty (“novel”, “innovative”), significance (“important”, “impactful”), clarity (“well-written”, “easy to follow”), strong experimental support (“ablation”, “extensive experiments”, “robustness”), etc.
  * **NEG keywords** (signals of weakness): “missing baseline”, “limited evaluation”, “unclear”, “overclaim”, “insufficient evidence”, “major concerns”, etc.
* Use **binary presence**, not raw counts:

  * if a keyword appears anywhere in the meta-review, it contributes **at most once**
  * this avoids score inflation from repetitive generations
* Apply **stronger penalties** for “major/serious” concern markers than minor concern markers.
* Apply small adjustments:

  * “borderline / uncertain” language slightly dampens the score
  * “promising / potential” language slightly boosts the score

This produces:

* `true_meta_review_sentiment_score`
* `pred_meta_review_sentiment_score`

### Step 2 — Filter to predicted accepts

To focus on papers the model would actually recommend, we first filter:

* keep only rows where `pred_decision` contains **ACCEPT**

### Step 3 — Rank accepted papers and select top-30

From the filtered set:

* sort by `pred_meta_review_sentiment_score` (descending)
* select **top 30**
* save the output CSV containing:
  `paper_id, true_decision, pred_decision, true_meta_review, pred_meta_review, true_meta_review_sentiment_score, pred_meta_review_sentiment_score`

---

## Findings (from Flan-T5 / BART / Pegasus outputs)

### 1) Even after filtering to predicted ACCEPT, false positives remain common

In the **BART** and **Pegasus** top-30 lists (filtered to `pred_decision = ACCEPT`), a noticeable fraction of papers still have:

* `true_decision = REJECT` but `pred_decision = ACCEPT`

Example patterns visible in the outputs:

* **BART top-30:** includes several `REJECT → ACCEPT` entries near the top (e.g., `rDRCIvTppL`, `xImTb8mNOr`, `DRf8RpofIN`, etc.).
* **Pegasus top-30:** includes multiple `REJECT → ACCEPT` cases (e.g., `fVgUXaesSS`, `2hbgKYuao1`, `LPXfOxe0zF`, etc.).

**Interpretation:** filtering to predicted accepts reduces noise, but the models still over-accept some papers.

### 2) Generated meta-reviews are systematically more positive than gold meta-reviews

Across many rows, `pred_meta_review_sentiment_score` is **substantially higher** than `true_meta_review_sentiment_score`, sometimes even when:

* the ground truth decision is REJECT
* the gold meta-review score is negative

Example (Pegasus list):

* some papers have **negative true sentiment** but **high predicted sentiment**, suggesting the model-generated meta-review emphasizes strengths more than the gold meta-review does.

**Interpretation:** these models tend to produce **optimistic / strength-heavy meta-reviews**, underweighting weaknesses compared to gold meta-reviews.

### 3) The ranking is best interpreted as “most positive-sounding accepted generations”

Because ranking is by `pred_meta_review_sentiment_score`, the top-30 reflects:

* papers whose **generated meta-reviews contain the strongest set of positive signals**
* not necessarily the papers that truly should be accepted





