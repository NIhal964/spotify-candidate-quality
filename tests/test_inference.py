import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import src.inference as inference


def test_run_inference_empty_df():
    with pytest.raises(ValueError):
        inference.run_inference(pd.DataFrame())


def test_run_inference_monkeypatched(monkeypatch):
    # create a small fake df
    df = pd.DataFrame({"a": [1, 2]})

    # fake build_features returns features matching 'f1'
    def fake_build_features(df_in, training, artifacts=None):
        X = pd.DataFrame({"f1": [0.1, 0.2]})
        return X, None, {"feature_cols": ["f1"]}

    class FakeModel:
        def predict_proba(self, X):
            # return probs for two rows
            return np.array([[0.2, 0.8] for _ in range(len(X))])

    monkeypatch.setattr(inference, "build_features", fake_build_features)
    monkeypatch.setattr(inference, "load_model", lambda model_file=None: FakeModel())

    out = inference.run_inference(df, model_file=None)

    assert "risk_probability" in out.columns
    assert "high_skip_risk_prediction" in out.columns
    assert out.shape[0] == 2
    assert np.allclose(out["risk_probability"], 0.8)
    assert set(out["high_skip_risk_prediction"].unique()) <= {0, 1}