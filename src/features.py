# src/features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.config import TARGET_COL


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

    # =========================
    # Feature definitions
    # =========================

    audio_features = [
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

    normalize_features = [
        "danceability",
        "energy",
        "tempo",
        "loudness",
        "valence",
    ]

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

    return X, None, None
