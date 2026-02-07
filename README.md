# Spotify Candidate Quality Estimation

Content-Based Candidate Quality Estimation for Music Recommendations

---

## Problem Context

Music recommendation systems must balance **exploration** (surfacing new or less-known tracks) with **user experience**.
Poor-quality or context-misaligned tracks surfaced too early can increase skips and negatively impact downstream personalization.

In early stages of a track’s lifecycle, **behavioral signals are sparse or unavailable**, making it difficult to assess
whether a track should be safely exposed or cautiously handled.

This project builds a **content-based ML system** to estimate *relative skip risk* for tracks **before sufficient user interaction data exists**, enabling safer exploration decisions.

---

## Decision Framing

**User of the model:**  
Recommendation system (not end users)

**Decision supported:**  
Which tracks should be deprioritized or cautiously exposed during early-stage recommendation?

**Model output:**  
A **continuous risk probability score** (0–1), used as a *risk prior* rather than a hard decision.

The model is intentionally **conservative** and designed to support **candidate filtering and routing**, not final ranking.

---

## Data & Target Construction

- Dataset: Public Spotify audio features dataset
- Granularity: Track-level
- No user interaction logs available

### Target Definition (Proxy)
A **genre-relative popularity percentile** is used as a proxy for historical underperformance:
high_skip_risk_proxy = 1 if popularity percentile within genre ≤ 25%

Why this proxy:
- Controls for genre-specific popularity distributions
- Reduces exposure bias compared to global popularity
- Produces a stable, interpretable relative risk signal

---

## Feature Engineering

Features are designed to be **available at inference time** and safe for cold-start scenarios.

### Feature groups:
- **Raw audio features:** acousticness, energy, loudness, tempo, etc.
- **Genre-normalized z-scores:** deviation from genre norms
- **Artist-level priors:** mean audio characteristics (audio-only, leakage-safe)
- **Categorical encodings:** genre one-hot, cyclic encoding for musical key

All feature choices were evaluated through controlled ablation experiments.

---

## Modeling Approach

### Baseline
- Regularized Logistic Regression (interpretable, calibration-friendly)

### Non-linear model
- LightGBM (to capture feature interactions and non-linear effects)

### Validation strategy
- **Artist-disjoint train/test split** to prevent artist-level leakage
- Evaluation framed as a **ranking / filtering task**, not pure classification

---

## Results & Insights

### Evaluation Setup

Performance is evaluated using **Recall@K** and **Lift@K**, reflecting how the system would behave when selecting a small
subset of high-risk tracks for cautious handling.

This aligns with real recommender-system decision constraints rather than optimizing accuracy alone.

---

### Model Performance

**Headline metric (business-aligned):**
- **Lift@5%: 1.57×**

This means the top 5% highest-risk tracks identified by the model contain **57% more truly risky tracks** than random selection.

**Secondary metrics:**
- Lift@10%: 1.43×
- Lift@15%: 1.35×

The model is effective as a **risk prior** for early-stage filtering.

---

### Ablation Study (Feature Contribution)

A controlled A/B/C experiment quantifies the impact of feature engineering:

| Experiment | Features Used | Lift@5% |
|----------|--------------|---------|
| A | Raw audio features | 1.12× |
| B | Audio + genre-normalized features | 1.20× |
| C | Full model (audio + genre z-scores + artist priors) | **1.57×** |

**Key takeaway:**  
Context-aware features (genre-relative deviations and artist priors) substantially improve risk separation compared to raw audio alone.

---

### Explainability (SHAP)

SHAP was applied to the trained LightGBM model to validate feature behavior and interpret risk drivers.

Key observations:
- Genre-relative loudness, energy, and danceability are strong contributors
- Absolute audio features behave differently depending on genre context
- Artist priors act as stabilizing signals rather than dominant predictors

This confirms the model learns **context-aware risk patterns**, not popularity memorization.

---

### Example Scored Outputs

The inference pipeline produces a **risk probability score** for each track:

| track_id | genre | risk_probability |
|--------|-------|------------------|
| 0ST6uPfvaPpJLtQwhE6KfC | Movie | 0.77 |
| 0IuslXpMROHdEPvSl1fTQK | Movie | 0.30 |
| 0BRjO6ga9RKCKjfDqeFgWV | Movie | 0.19 |

These scores are intended for **downstream decision-making**, not standalone judgments.

---

## Limitations & Production Considerations

This model estimates **historical, context-relative risk** and is intentionally conservative.

Edge cases handled outside this model in a full production system include:
- Cold-start tracks with limited exposure
- Genre misclassification or context mismatch
- Viral or promotional popularity spikes
- Temporal shifts in listener preferences

In practice, model outputs would be treated as **weak priors**, combined with exploration logic and real-time feedback.

---

## Reproducibility & Usage

### Training and experiments
```bash
python -m src.train --model lightgbm --experiment full --save-model True
```
Inference on new data
python -m src.run_inference

## Testing

Core training, inference, evaluation, and explainability logic is covered by unit tests.
Tests validate:
- Feature construction and leakage-safe splits
- Training CLI behavior and artifact persistence
- Inference-time feature alignment
- Optional explainability (SHAP) behavior

Tests are designed to run without requiring raw data downloads.


Artifacts (trained model and feature metadata) are persisted to ensure consistent inference and explainability.

Engineering Notes

CLI-driven experimentation

Modular training and inference pipelines

Leakage-aware feature design

Explainability integrated via SHAP

LLMs used as a development accelerator for feature ideation and experiment setup, with outputs validated through tests, metrics, and data checks

## Repo srtucture
spotify-candidate-quality/
├── src/
│   ├── train.py
│   ├── train_lgbm.py
│   ├── train_logistic.py
│   ├── inference.py
│   ├── explain.py
│   └── features.py
├── tests/
│   ├── test_train_common.py
│   ├── test_train_cli.py
│   ├── test_train_artifacts.py
│   ├── test_inference.py
│   ├── test_explain.py
│   └── test_evaluate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── data/
│   ├── raw/              # local only (ignored)
│   └── processed/
├── models/               # saved artifacts (optional, ignored)
├── assets/               # curated plots for README / portfolio
├── .gitignore
└── README.md


