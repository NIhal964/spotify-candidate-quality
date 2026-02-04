import pandas as pd
from src.features import build_features


def make_sample_df():
    # Create a small dataset with 2 genres and 4 tracks each (so 25% bucket exists)
    data = {
        "genre": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "popularity": [5, 10, 20, 30, 3, 8, 15, 25],
        "time_signature": ["4/4"] * 8,
        "mode": ["Major", "Minor", "Major", "Major", "Major", "Minor", "Major", "Minor"],
        "key": ["C", "D", "E", "F", "G", "A", "B", "C#"],
        "artist_name": ["art1", "art1", "art2", "art2", "art3", "art3", "art4", "art4"],
        # audio features
        "acousticness": [0.1, 0.2, 0.3, 0.12, 0.1, 0.2, 0.15, 0.11],
        "danceability": [0.5, 0.6, 0.55, 0.48, 0.4, 0.45, 0.5, 0.42],
        "energy": [0.3, 0.4, 0.35, 0.32, 0.2, 0.25, 0.3, 0.22],
        "instrumentalness": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.05],
        "liveness": [0.1, 0.2, 0.15, 0.12, 0.05, 0.07, 0.09, 0.06],
        "loudness": [-10, -9, -11, -10, -12, -10, -9, -11],
        "speechiness": [0.02, 0.03, 0.01, 0.025, 0.02, 0.025, 0.02, 0.015],
        "tempo": [100, 110, 105, 102, 120, 115, 108, 112],
        "valence": [0.3, 0.4, 0.35, 0.33, 0.2, 0.25, 0.3, 0.22],
        "duration_ms": [200000, 210000, 190000, 205000, 230000, 220000, 205000, 215000],
    }
    return pd.DataFrame(data)


def test_build_features_training_behavior():
    df = make_sample_df()
    X, y, artifacts = build_features(df, training=True)

    assert isinstance(X, type(df))
    assert "feature_cols" in artifacts
    # X must have columns equal to artifacts['feature_cols']
    assert list(X.columns) == artifacts["feature_cols"]
    # Target should be present and length matches
    assert len(y) == len(df)
    # There should be at least one positive (per-genre low performer)
    assert y.sum() >= 1
