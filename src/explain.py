import argparse
import logging
import joblib
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None


from src.data_loader import load_raw_data
from src.features import build_features
from src.train_common import load_and_split
from src.config import MODEL_DIR, LOG_DIR

logger = logging.getLogger(__name__)


def select_explainer(model):
    """Auto-select a SHAP explainer based on model type.

    Returns a lightweight lazy explainer that defers construction of the
    underlying SHAP explainer until `shap_values` (or other methods) are
    actually called. This avoids instantiating SHAP explainers on toy/dummy
    objects used in unit tests.
    """
    if shap is None:
        raise ImportError(
        "SHAP is not installed. Install it with `pip install shap` to run explanations."
    )

    # Unwrap pipeline
    if hasattr(model, "named_steps"):
        m = model.named_steps.get("model", model)
    else:
        m = model

    class LazyExplainer:
        def __init__(self, maker, kind):
            self._maker = maker
            self._kind = kind
            self._explainer = None

        def _ensure(self):
            if self._explainer is None:
                try:
                    self._explainer = self._maker()
                except Exception:
                    # If construction fails, try to fall back to KernelExplainer
                    if hasattr(m, "predict_proba"):
                        try:
                            self._explainer = shap.KernelExplainer(m.predict_proba, shap.sample(np.zeros((100, 1)), 10))
                        except Exception:
                            raise
                    else:
                        raise

        def shap_values(self, *args, **kwargs):
            self._ensure()
            return self._explainer.shap_values(*args, **kwargs)

        def __getattr__(self, item):
            self._ensure()
            return getattr(self._explainer, item)

    cls = type(m).__name__.lower()
    if "lgbm" in cls or "lightgbm" in cls or hasattr(m, "feature_importances_"):
        return LazyExplainer(lambda: shap.TreeExplainer(m), "tree"), "tree"
    if "logistic" in cls or "linear" in cls or "sgd" in cls:
        return LazyExplainer(lambda: shap.LinearExplainer(m, np.zeros((1, 1))), "linear"), "linear"
    # fallback
    return LazyExplainer(lambda: shap.KernelExplainer(m.predict_proba, shap.sample(np.zeros((100, 1)), 10)), "kernel"), "kernel"


def build_artifacts_if_missing(model_file: Path):
    """Return artifacts for feature building. Prefer persisted artifacts if available, otherwise rebuild from training split."""
    artifacts_file = model_file.parent / "artifacts.pkl"
    if artifacts_file.exists():
        try:
            return joblib.load(artifacts_file)
        except Exception:
            logger.info("Could not load saved artifacts; will recompute from training data")

    # Recompute from training split
    train_df, _ = load_and_split()
    # build_features returns (X, y, artifacts) when training=True
    _, _, artifacts = build_features(train_df, training=True)
    return artifacts


def sample_df(df: pd.DataFrame, X_full: pd.DataFrame, probs: Optional[np.ndarray], sample_size: int, stratify: str = "random") -> pd.DataFrame:
    """Return a sampled subset of df according to stratify strategy."""
    if stratify == "random":
        return df.sample(n=sample_size, random_state=42)

    if stratify == "stratified_by_target":
        # try to stratify by target column if present
        if "high_skip_risk_proxy" in df.columns:
            pos = df[df["high_skip_risk_proxy"] == 1]
            neg = df[df["high_skip_risk_proxy"] == 0]
            n_pos = min(len(pos), sample_size // 2)
            n_neg = sample_size - n_pos
            return pd.concat([pos.sample(n=n_pos, random_state=42), neg.sample(n=n_neg, random_state=42)])
        else:
            return df.sample(n=sample_size, random_state=42)

    if stratify == "top_risk":
        if probs is None:
            raise ValueError("top_risk sampling requires predicted probabilities")
        idx = np.argsort(probs)[::-1][:sample_size]
        return df.iloc[idx]

    raise ValueError("Unknown stratify option")


def run_shap(model_file: Path, sample_size: int = 1000, stratify: str = "top_risk", save_csv: bool = True, top_n: int = 5):
    logger.info(f"Running SHAP analysis (top_n={top_n})")

    # -------------------------
    # Load model
    # -------------------------
    model = joblib.load(model_file)

    # -------------------------
    # Ensure artifacts (genre stats, cols, feature order)
    # -------------------------
    try:
        artifacts = build_artifacts_if_missing(model_file)
    except Exception as exc:
        logger.error(f"Could not prepare artifacts: {exc}")
        raise

    # -------------------------
    # Load data and build features using artifacts
    # -------------------------
    df = load_raw_data()

    # Build features for whole dataset (for top-risk sampling)
    X_all, y_all, _ = build_features(df, training=False, artifacts=artifacts)

    # Predict probabilities if model supports predict_proba
    # Unwrap pipeline to predict on proper columns
    if hasattr(model, "named_steps"):
        base_model = model
    else:
        base_model = model

    if hasattr(base_model, "predict_proba"):
        try:
            probs = base_model.predict_proba(X_all[artifacts["feature_cols"]])[:, 1]
        except Exception:
            # fallback: try predict_proba on X_all directly
            probs = base_model.predict_proba(X_all)[:, 1]
    else:
        probs = None

    # -------------------------
    # Sample for SHAP
    # -------------------------
    df_sample = sample_df(df, X_all, probs, sample_size, stratify=stratify)
    X_sample, y_sample, _ = build_features(df_sample, training=False, artifacts=artifacts)

    # Ensure feature order matches training feature_cols
    feature_cols = artifacts["feature_cols"]
    X_sample = X_sample[feature_cols]

    # -------------------------
    # SHAP explainer selection
    # -------------------------
    explainer, explainer_kind = select_explainer(model)
    logger.info(f"Using SHAP explainer: {explainer_kind}")

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers shap_values may be list-like -> choose positive class
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # -------------------------
    # Produce tabular summaries
    # -------------------------
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    fi = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    fi = fi.sort_values("mean_abs_shap", ascending=False)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fi_path = LOG_DIR / "shap_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    logger.info(f"Saved SHAP feature importance table to {fi_path}")

    if save_csv:
        # Save per-sample SHAP values for the sample (beware large files)
        sv_df = pd.DataFrame(shap_vals, columns=feature_cols)
        sv_df_path = LOG_DIR / "shap_values_sample.csv"
        sv_df.to_csv(sv_df_path, index=False)
        logger.info(f"Saved SHAP sample values to {sv_df_path}")

    # -------------------------
    # Summary plot
    # -------------------------
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_sample, max_display=top_n, show=False)
    plot_path = LOG_DIR / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {plot_path} (top %d features)" % top_n)

    return fi_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", default=str(MODEL_DIR / "model_lgbm.pkl"))
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--stratify", choices=["random", "stratified_by_target", "top_risk"], default="top_risk")
    p.add_argument("--top-n", type=int, default=5, help="Number of top features to show in SHAP summary plot")
    p.add_argument("--save-csv", dest="save_csv", action="store_true")
    p.set_defaults(save_csv=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_shap(Path(args.model_file), sample_size=args.sample_size, stratify=args.stratify, save_csv=args.save_csv, top_n=args.top_n)
