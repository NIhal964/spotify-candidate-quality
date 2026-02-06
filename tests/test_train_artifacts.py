import joblib
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

import src.train_logistic as tlog
import src.train_lgbm as tlgbm


def fake_prepare_features_train():
    # return X_train, y_train, X_test, y_test, artifacts
    X_train = pd.DataFrame({"f1": [0, 1]})
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame({"f1": [0]})
    y_test = pd.Series([0])
    artifacts = {"feature_cols": ["f1"]}
    return X_train, y_train, X_test, y_test, artifacts


class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.array([[0.5, 0.5] for _ in range(len(X))])


class FakeLGBM:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.array([[0.5, 0.5] for _ in range(len(X))])


def test_train_logistic_saves_artifacts(monkeypatch):
    monkeypatch.setattr(tlog, "prepare_features", lambda a, b: fake_prepare_features_train())
    monkeypatch.setattr(tlog, "Pipeline", FakePipeline)

    dumped = []

    def fake_dump(obj, path):
        dumped.append((obj, str(path)))

    monkeypatch.setattr(tlog, "joblib", MagicMock(dump=fake_dump))

    tlog.train_logistic(experiment="full", save_model=True)

    # artifacts should have been saved (one of the dump calls contains artifacts dict)
    assert any(isinstance(item[0], dict) and "feature_cols" in item[0] for item in dumped)


def test_train_lgbm_saves_artifacts(monkeypatch):
    monkeypatch.setattr(tlgbm, "prepare_features", lambda a, b: fake_prepare_features_train())
    monkeypatch.setattr(tlgbm, "LGBMClassifier", FakeLGBM)

    dumped = []

    def fake_dump(obj, path):
        dumped.append((obj, str(path)))

    monkeypatch.setattr(tlgbm, "joblib", MagicMock(dump=fake_dump))

    tlgbm.train_lgbm(experiment="full", save_model=True)

    assert any(isinstance(item[0], dict) and "feature_cols" in item[0] for item in dumped)