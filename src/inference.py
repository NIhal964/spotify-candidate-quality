# src/inference.py

import logging
import joblib
import pandas as pd
from typing import Optional

from src.config import MODEL_FILE, INFERENCE_THRESHOLD, MODEL_DIR
from src.features import build_features

logger = logging.getLogger(__name__)


from pathlib import Path
from src.train_common import load_and_split


def load_model(model_file: Optional[str] = None):
    """Load trained model from disk.

    Parameters
    ----------
    model_file : str or Path, optional
        Path to model artifact. If None, `MODEL_FILE` is used.
    """
    model_path = Path(model_file) if model_file is not None else MODEL_FILE

    if not model_path.exists():
        logger.error(f"Trained model not found at {model_path}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")

    return model


def load_artifacts(artifacts_file: Optional[str] = None):
    """Load or recompute training artifacts required for feature building."""
    artifacts_path = Path(artifacts_file) if artifacts_file is not None else MODEL_DIR / "artifacts.pkl"
    if artifacts_path.exists():
        try:
            return joblib.load(artifacts_path)
        except Exception:
            logger.info("Could not load artifacts file; will recompute from training data")

    # Recompute artifacts from training split
    train_df, _ = load_and_split()
    _, _, artifacts = build_features(train_df, training=True)
    return artifacts


def run_inference(df: pd.DataFrame, model_file: Optional[str] = None, artifacts: Optional[dict] = None) -> pd.DataFrame:
    """
    Run offline inference and return risk scores.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    model_file : str or Path, optional
        Optional path to model file to load; if None, default MODEL_FILE is used.
    artifacts : dict, optional
        Precomputed artifacts to use for feature building; if None, artifacts
        will be loaded from disk or recomputed from training split.

    Returns
    -------
    pd.DataFrame
        Input data with risk scores and predictions appended.
    """

    if df.empty:
        raise ValueError("Input DataFrame for inference is empty")

    # Load model and artifacts
    model = load_model(model_file=model_file)
    arts = artifacts if artifacts is not None else load_artifacts()

    # Build features (no target during inference)
    X, _, _ = build_features(
        df,
        training=False,
        artifacts=arts
    )

    # Predict risk probability (use feature_cols ordering if available)
    try:
        feat_cols = arts["feature_cols"]
        X_pred = X[feat_cols]
    except Exception:
        X_pred = X

    if not hasattr(model, "predict_proba"):
        logger.error("Loaded model does not support probability prediction")
        raise AttributeError("Model missing predict_proba")

    risk_prob = model.predict_proba(X_pred)[:, 1]
    risk_label = (risk_prob >= INFERENCE_THRESHOLD).astype(int)

    output = df.copy()
    output["risk_probability"] = risk_prob
    output["high_skip_risk_prediction"] = risk_label

    logger.info(
        f"Inference completed | samples={len(output)} | threshold={INFERENCE_THRESHOLD}"
    )

    return output
