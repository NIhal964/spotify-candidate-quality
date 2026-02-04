# 🎧 Spotify Candidate Quality – Upstream Risk Prior for Music Recommendation

[![CI](https://github.com/NIhal964/spotify-candidate-quality/actions/workflows/ci.yml/badge.svg)](https://github.com/NIhal964/spotify-candidate-quality/actions)

**Elevator (TL;DR):** This project builds a non-personalized upstream risk prior that concentrates risky tracks for early filtering — **Lift@5% = 1.57× (LightGBM)**. Repro: see the `train` command in Usage (deterministic seed included).

## Business Problem

Large-scale music recommendation systems generate **millions of candidate tracks** before final ranking.  
Not all candidates are equally safe to surface — some tracks are **systematically skipped** when shown broadly, even if they are not inherently “bad” music.

**Goal:**  
Build an **upstream risk-scoring model** that estimates whether a track is likely to underperform *if blindly surfaced*, so that downstream ranking models can focus on safer candidates.

This model is **not** a recommender and **not** user-personalized.  
It acts as a **risk prior / guardrail** in the recommendation pipeline.

---

## Why This Matters

In production recommendation systems:

- Recommending a *bad candidate* is often more costly than missing a good one
- Early filtering reduces load on expensive ranking models
- Risk priors improve overall system reliability and user experience

This project simulates that **early-stage decision layer** using only content and metadata signals.

---

## Dataset & Target Construction

**Dataset:** Spotify audio features dataset  
**Key limitation:** No user behavior, no session context, no exposure logs

Because true skip labels are unavailable, a **proxy target** is constructed:

- Tracks are ranked by **popularity percentile within their genre**
- Bottom 25% within each genre are labeled as **high skip risk**
- This controls for genre exposure bias

```text
high_skip_risk = 1 if popularity_percentile_within_genre ≤ 25%
```

This target represents **historical, context-relative underperformance**, not intrinsic quality.

---

## Feature Design

Features are explicitly designed to avoid leakage and reflect what would be available at inference time.

### 1. Raw Audio Features
Absolute musical properties such as loudness, tempo, energy, speechiness, and duration.

### 2. Genre-Normalized Audio Features
Z-scores computed within each genre to capture **deviation from genre norms**  
(e.g., unusually loud or slow tracks for a given genre).

### 3. Artist-Level Priors
Artist-average audio features computed using **audio signals only**.  
These encode historical expectations without using outcome variables.

### 4. Structural Encodings
- One-hot genre encoding
- Cyclic encoding for musical key
- Explicit removal of popularity-derived fields

All feature logic is shared between training and inference.

---

## Train / Test Strategy

To prevent leakage from repeated artists:

- **Group-aware split by artist**
- Ensures no artist appears in both train and test sets

This avoids inflated performance from memorizing artist signatures rather than learning risk patterns.

---

## Evaluation Strategy

Global accuracy metrics are insufficient for this problem.

Instead, evaluation focuses on:

- **Lift@K** – primary operational metric: how well the model concentrates risky tracks at the top
- Recall at small K (early filtering effectiveness)
- ROC-AUC (secondary sanity check; not the primary operational metric)

This mirrors how **risk filters** are evaluated in real recommender systems.

---

## Modeling Approach

### Baseline: Logistic Regression
Used to validate linear signal and establish a transparent baseline.

### Non-Linear Model: LightGBM
Introduced to test whether feature interactions improve early risk detection.

No heavy hyperparameter tuning was performed — the objective was **signal validation**, not leaderboard optimization.

---

## Results

**Headline metric (primary):** **Lift@5% = 1.57×** (LightGBM) — secondaries: 10% = 1.43×, 15% = 1.35×.

**Key numbers:** Data rows = 232,725 | Positive rate ≈ 24.5% | Random seed = 42

### Feature Ablation (Lift@5%)


---

### 1. Training & Reproducing the headline run

To reproduce the headline result deterministically:

```bash
python -m src.train --model lightgbm --experiment full --save-model --random-state 42
```

Artifacts produced (check these after running): `models/model_lgbm.pkl`, `logs/shap_feature_importance.csv`, `logs/feature_importance.csv`

---

| Feature Set | Lift@5% |
|------------|--------|
| Audio-only | 1.12× |
| Audio + Genre Z-Scores | 1.20× |
| Full Feature Set | 1.42× |

### Final Model (LightGBM)

| K% | Recall | Lift |
|---|------|------|
| 5% | 0.079 | **1.57×** |
| 10% | 0.143 | 1.43× |
| 15% | 0.203 | 1.35× |
| 20% | 0.262 | 1.31× |

**Interpretation:**  
Inspecting only the top 5% most risky tracks surfaces **57% more underperforming tracks than random selection**.

Although global AUC is modest, the model is effective as an **upstream risk filter**, which is the intended role.

---

## Model Interpretability (SHAP)

SHAP analysis was applied to the final LightGBM model to understand *why* it improves early risk detection.

**Key drivers:**
- Genre context (e.g., Movie / soundtrack-like content)
- Artist-level priors (duration, energy, speechiness)
- Non-linear interactions between artist expectations and audio features

**Key insight:**  
The model relies more on **artist-level risk priors and contextual mismatch** than per-track audio nuance — exactly what an upstream risk model should capture.

This explains why non-linear models improve **Lift@K** without dramatically increasing global AUC.

---

## System Placement

This model would sit **before personalized ranking**:

```
Candidate Generation
      ↓
Candidate Risk Prior (this model)
      ↓
Behavioral / Personalized Ranking
      ↓
Final Recommendations
```

It reduces exposure of systematically risky tracks while preserving downstream flexibility.

---

## Key Takeaways

- Audio-only signals have limited predictive power without user context
- Genre-relative features and artist priors significantly improve risk concentration
- Non-linear models are valuable for **early filtering**, not final ranking
- Modest AUC does not imply low business value in this setting

---

## Engineering Notes

- Config-driven pipeline
- CLI-based experiment control
- Shared feature logic for training and inference
- Group-aware splits to prevent leakage
- Logging, plots, and artifacts saved deterministically

---

## Usage

### 1. Training & Experiments

Run feature ablation experiments and train a model via CLI:

```bash
python -m src.train --model lightgbm --experiment full
```

Supported options:
- `--model`: `logistic` | `lightgbm`
- `--experiment`: `audio` | `genre` | `full`
- `--save-model`: persist trained model artifacts

Example:

```bash
python -m src.train --model lightgbm --experiment full --save-model
```

This command:
- performs group-aware splits
- evaluates Lift@K and ROC/PR metrics
- writes plots and feature importances to `logs/`
- optionally saves the trained pipeline to `models/`

---

### 2. Model Explanation (SHAP)

Generate global feature explanations for the final LightGBM model:

```bash
python -m src.explain
```

Outputs:
- SHAP summary plot (`logs/shap_summary.png`)
- Feature contribution analysis for interpretability

---

### 3. Offline Inference (Optional)

Run offline risk scoring on new tracks:

```bash
python -m src.inference
```

Outputs:
- risk probability
- binary high-risk flag (threshold-based)

---

## What This Project Is (and Isn’t)

**Is:**
- A realistic simulation of an upstream recommender component
- Focused on system-level impact and evaluation
- Honest about data and modeling limitations

**Isn’t:**
- A personalized recommendation system
- A user behavior model
- An overfit benchmark exercise
