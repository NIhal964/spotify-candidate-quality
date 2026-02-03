import numpy as np
import math
from src.evaluate import compute_recall_at_k, compute_lift_at_k


def test_recall_basic():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    # top 40% of 5 = ceil(2) = 2 -> picks indices [0,1] ; positives among selected = 1, total positives = 2
    recall = compute_recall_at_k(y_true, y_score, 0.4)
    assert math.isclose(recall, 0.5)


def test_lift_basic():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    # recall 0.5 at k=0.4 -> lift = 0.5 / 0.4 = 1.25
    lift = compute_lift_at_k(y_true, y_score, 0.4)
    assert math.isclose(lift, 1.25)


def test_no_positives():
    y_true = np.zeros(10)
    y_score = np.linspace(0, 1, 10)
    recall = compute_recall_at_k(y_true, y_score, 0.3)
    lift = compute_lift_at_k(y_true, y_score, 0.3)
    assert np.isnan(recall)
    assert np.isnan(lift)


def test_invalid_inputs():
    import pytest
    with pytest.raises(ValueError):
        compute_recall_at_k([], [], 0)
    with pytest.raises(ValueError):
        compute_recall_at_k([1, 0], [0.1], 0.5)
