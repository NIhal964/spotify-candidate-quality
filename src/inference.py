# src/inference.py
import pandas as pd
import joblib
from pathlib import Path

from src.config import MODEL_DIR
from src.features import build_features


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run risk scoring on new tracks.

    Returns:
        DataFrame with risk_probability column
    """

    model_path = MODEL_DIR / "model_lgbm.pkl"
    artifacts_path = MODEL_DIR / "artifacts.pkl"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run training first.")

    if not artifacts_path.exists():
        raise FileNotFoundError("Training artifacts not found. Run training with --save-model True.")

    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)

    # Build inference features
    X, _, _ = build_features(
        df,
        training=False,
        artifacts=artifacts
    )

    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["risk_probability"] = probs

    return out
