# Content-Based Candidate Quality Estimation for Music Recommendation Systems

## Overview

This project builds a content-based risk scoring model for music recommendation systems.

Rather than predicting engagement directly, the model estimates the probability that a track will underperform (high skip risk) when surfaced to users. The output is a continuous risk score designed to support candidate filtering, gating, and ranking before personalization signals are available.

This mirrors how large-scale recommender systems manage exploration risk during early exposure.

## Business Problem

Music recommendation systems must balance discovery with user experience.

Surfacing low-quality or mismatched tracks early in a session can increase skip rates, reduce session length, and bias downstream personalization models.

The system therefore needs a way to identify risky candidates early, using only content-level information.

## Decision Framing

Who uses this model?  
The recommendation system itself (not end users).

What decision does it support?  
Which candidate tracks should be deprioritized during exploration and how aggressively a track should be exposed under limited recommendation slots.

What does the model output?  
A risk probability (risk_probability ∈ [0,1]). Higher values indicate higher likelihood of underperformance. The score is intentionally rankable, not a hard label.

## Target Definition (Proxy)

True skip outcomes are not available, so a proxy target is constructed:

high_skip_risk_proxy = 1 if track popularity percentile within its genre ≤ 25%

Why this proxy?

It is genre-aware, avoids cross-genre popularity bias, is stable over time, and aligns with business intuition that tracks consistently underperforming relative to peers are riskier to surface.

The model estimates historical, context-relative risk rather than absolute song quality.

## Data

Source: Public Spotify audio features dataset (Kaggle)

Granularity: Track-level

Key fields include audio features (energy, loudness, danceability, etc.), genre, artist, and popularity (used only for target construction).

## Feature Engineering

Features are designed to capture both intrinsic track properties and contextual deviation signals.

Raw audio features represent absolute musical characteristics such as energy, loudness, danceability, tempo, valence, and instrumentalness. These are especially important for cold-start scenarios.

Genre-normalized features are computed as within-genre z-scores for selected audio features including loudness, energy, danceability, tempo, and speechiness. These capture how tracks deviate from genre norms.

Artist-level priors are computed as artist-level averages of audio features using training data only. These provide prior context without leaking outcome information.

Categorical encodings include one-hot encoding for genre and cyclic encoding for musical key.

All feature transformations are persisted as training artifacts and reused at inference time to ensure feature parity.

## Modeling Approach

A regularized logistic regression model is used as a transparent baseline.

A non-linear LightGBM model is trained to capture feature interactions, improve recall under low exposure budgets, and better align with ranking-based evaluation metrics.

## Evaluation Strategy

Traditional metrics such as accuracy or ROC-AUC are insufficient for this problem.

The primary metric is Lift@K, which measures how much better the model is at identifying risky tracks within the top K percent of candidates compared to random selection. This directly reflects how recommendation systems operate under exposure constraints.

Results summary:

Experiment A (raw audio features): Lift@5% = 1.12x  
Experiment B (audio + genre-normalized features): Lift@5% = 1.20x  
Experiment C (full feature set): Lift@5% = 1.57x  

The full model surfaces 57 percent more high-risk tracks in the top 5 percent than random selection.

## Interpretability (SHAP)

SHAP is used to explain why the model flags tracks as risky.

Key observations include genre-relative loudness and energy being stronger predictors than raw values, consistent risk signals from tracks deviating negatively from genre norms, and marginal contribution from artist-level priors.

Artifacts generated include global feature importance tables, per-sample explanations, and SHAP summary plots saved to the logs directory.

## Inference and Scoring

At inference time, no target labels are available. Raw track metadata is transformed using persisted training artifacts and the model outputs a risk probability score.

Example output columns:

track_id  
genre  
risk_probability  

These scores are intended to be ranked, thresholded, or combined with downstream personalization or business rules.

## Usage

Training and experiments:

python -m src.train --model lightgbm --experiment full --save-model True

This runs feature ablation experiments, logs Lift@K metrics, and saves the trained model and artifacts.

Inference:

python -m src.run_inference

This scores tracks and saves results to data/processed/scored_tracks.csv.

Model explainability:

python -m src.explain --model-file models/model_lgbm.pkl

## Limitations and Future Work

The target is a proxy and not true skip behavior. The model does not include user-level or session-level context and is designed for candidate risk estimation rather than final ranking.

Future extensions include incorporating session context, exploration-aware thresholding, and online A/B testing simulation.

## Why This Project Matters

This project focuses on decision support rather than pure prediction.

It demonstrates business-aligned problem framing, feature design under real-world constraints, ranking-aware evaluation, reproducible training and inference, and model interpretability.

The system is intentionally conservative and designed to integrate cleanly into a larger recommendation pipeline.
