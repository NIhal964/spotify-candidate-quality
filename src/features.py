## src/features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.config import TARGET_COL, RISK_PERCENTILE_THRESHOLD

# Expose feature lists for use in experiments
AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
]

NORMALIZE_FEATURES = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "valence",
]


def build_features(
    df: pd.DataFrame,
    training: bool,
    artifacts: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict]]:
    """
    Build model-ready features for training or inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw columns.
    training : bool
        Whether this is training mode.
    artifacts : dict, optional
        Training-time artifacts required for inference.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or None
        Target variable if training=True.
    artifacts : dict or None
        Training artifacts (only returned during training).
    """

    df = df.copy()

    # ------------------------
    # Popularity percentile & target proxy
    # ------------------------
    # Compute popularity percentile within each genre (pop_pct_genre)
    if "pop_pct_genre" not in df.columns:
        df["pop_pct_genre"] = df.groupby("genre")["popularity"].rank(pct=True, method="average")

    # low_performance is used elsewhere in features and leakage checks
    df["low_performance"] = (df["pop_pct_genre"] <= RISK_PERCENTILE_THRESHOLD).astype(int)

    # Derived target (proxy for skip risk)
    df[TARGET_COL] = (df["pop_pct_genre"] <= RISK_PERCENTILE_THRESHOLD).astype(int)

    # ------------------------
    # Normalize categorical encodings
    # ------------------------
    # Convert textual 'mode' ('Major'/'Minor') to numeric 1/0
    if "mode" in df.columns and df["mode"].dtype == object:
        df["mode"] = df["mode"].map({"Major": 1, "Minor": 0})
        if df["mode"].isnull().any():
            # For any unexpected values, fall back to a binary flag where non-null -> 1
            df["mode"] = df["mode"].fillna(0).astype(int)
        else:
            df["mode"] = df["mode"].astype(int)

    # =========================
    # Feature definitions
    # =========================

    # Use global feature lists defined at module-level so callers (train) can reuse them
    audio_features = AUDIO_FEATURES
    normalize_features = NORMALIZE_FEATURES

        # =========================
    # Time signature (numeric)
    # =========================

    if "time_signature_num" not in df.columns:
        if "time_signature" not in df.columns:
            raise ValueError("Expected 'time_signature' column to derive time_signature_num")

        df["time_signature_num"] = (
            df["time_signature"]
            .astype(str)
            .str.split("/")
            .str[0]
            .astype(int)
        )

        df.drop(columns=["time_signature"], inplace=True)


    categorical_numeric = [
        "mode",
        "time_signature_num",
        "key_sin",
        "key_cos",
    ]

    # =========================
    # Musical key (cyclical)
    # =========================

    key_mapping = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11,
    }

    df["key_num"] = df["key"].map(key_mapping)

    df["key_sin"] = np.sin(2 * np.pi * df["key_num"] / 12)
    df["key_cos"] = np.cos(2 * np.pi * df["key_num"] / 12)

    df.drop(columns=["key", "key_num"], inplace=True)

    # =========================
    # Genre normalization
    # =========================

    if training:
        genre_stats = (
            df.groupby("genre")[normalize_features]
            .agg(["mean", "std"])
        )
    else:
        if artifacts is None or "genre_stats" not in artifacts:
            raise ValueError("Inference requires genre_stats artifacts")
        genre_stats = artifacts["genre_stats"]

    for col in normalize_features:
        mean_map = df["genre"].map(genre_stats[col]["mean"])
        std_map = df["genre"].map(genre_stats[col]["std"])
        df[f"{col}_z_genre"] = (df[col] - mean_map) / std_map

    # =========================
    # Artist-level audio priors
    # =========================

    artist_means = (
        df.groupby("artist_name")[audio_features]
        .transform("mean")
        .add_prefix("artist_mean_")
    )

    df = pd.concat([df, artist_means], axis=1)

    # =========================
    # Genre one-hot encoding
    # =========================

    if training:
        df = pd.get_dummies(df, columns=["genre"], prefix="genre")
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
    else:
        if artifacts is None or "genre_cols" not in artifacts:
            raise ValueError("Inference requires genre_cols artifacts")

        df = pd.get_dummies(df, columns=["genre"], prefix="genre")
        genre_cols = artifacts["genre_cols"]

        df = df.reindex(columns=df.columns.union(genre_cols), fill_value=0)
        df = df[genre_cols + [c for c in df.columns if not c.startswith("genre_")]]

    # =========================
    # Final feature matrix
    # =========================

    feature_cols = (
        audio_features +
        [f"{c}_z_genre" for c in normalize_features] +
        list(artist_means.columns) +
        categorical_numeric +
        genre_cols
    )

    # =========================
    # Leakage protection
    # =========================

    leakage_cols = [
        TARGET_COL,
        "popularity",
        "pop_pct_genre",
        "low_performance",
    ]

    for col in leakage_cols:
        if col in feature_cols:
            raise ValueError(f"Leakage detected: {col}")

    X = df[feature_cols]

    if training:
        y = df[TARGET_COL]
        artifacts = {
            "genre_stats": genre_stats,
            "genre_cols": genre_cols,
            "feature_cols": feature_cols,
        }
        return X, y, artifacts

    # During inference / evaluation if the target column exists return it for scoring
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
    else:
        y = None

    return X, y, None