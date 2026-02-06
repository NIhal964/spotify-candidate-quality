import pytest
from src.train_common import get_cols_for_experiment


def test_get_cols_audio_genre_full():
    # Minimal artifacts mock
    artifacts = {"feature_cols": ["f1", "f2", "genre_A", "artist_mean_x"]}

    audio_cols = get_cols_for_experiment("audio", artifacts)
    assert "key_sin" in audio_cols and "key_cos" in audio_cols

    genre_cols = get_cols_for_experiment("genre", artifacts)
    assert any(c.endswith("_z_genre") for c in genre_cols) or len(genre_cols) > len(audio_cols)

    full_cols = get_cols_for_experiment("full", artifacts)
    assert full_cols == artifacts["feature_cols"]

    with pytest.raises(ValueError):
        get_cols_for_experiment("unknown", artifacts)