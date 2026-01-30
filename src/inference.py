# src/inference.py

import logging
import joblib
import pandas as pd

from src.config import MODEL_FILE, INFERENCE_THRESHOLD
from src.features import build_features

logger = logging.getLogger(__name__)


def load_model():
    """
    Load trained model from disk.
    """
    if not MODEL_FILE.exists():
        logger.error(f"Trained model not found at {MODEL_FILE}")
        raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    logger.info("Model loaded successfully")

    return model


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run offline inference and return risk scores.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    pd.DataFrame
        Input data with risk scores and predictions appended.
    """

    if df.empty:
        raise ValueError("Input DataFrame for inference is empty")

    model = load_model()

    # Build features (no target during inference)
    X, _, _ = build_features(
        df,
        training=False,
        artifacts=artifacts
    )

    # Predict risk probability
    risk_prob = model.predict_proba(X)[:, 1]
    risk_label = (risk_prob >= INFERENCE_THRESHOLD).astype(int)

    output = df.copy()
    output["risk_probability"] = risk_prob
    output["high_skip_risk_prediction"] = risk_label

    logger.info(
        f"Inference completed | samples={len(output)} | threshold={INFERENCE_THRESHOLD}"
    )

    return output
