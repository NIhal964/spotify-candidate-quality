import pytest

shap = pytest.importorskip("shap")

from src.explain import select_explainer


class DummyTree:
    feature_importances_ = [1, 2, 3]


class DummyLinear:
    pass  # name 'DummyLinear' will result in 'dummylinear' and match linear if adjusted


class DummyOther:
    def predict_proba(self, x):
        return x


def test_select_explainer_tree():
    expl, kind = select_explainer(DummyTree())
    assert kind == "tree"


def test_select_explainer_linear():
    # Forge a class with name containing 'Logistic' to trigger linear path
    class LogisticLike:
        pass

    LogisticLike.__name__ = "LogisticRegression"
    expl, kind = select_explainer(LogisticLike())
    assert kind == "linear"


def test_select_explainer_kernel():
    expl, kind = select_explainer(DummyOther())
    assert kind == "kernel" or kind in ("kernel",)
