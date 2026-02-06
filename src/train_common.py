import logging
from typing import Dict, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODEL_DIR, LOG_DIR, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_raw_data
from src.features import build_features, AUDIO_FEATURES, NORMALIZE_FEATURES

logger = logging.getLogger(__name__)


def load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data and return train/test DataFrames (group-aware split)."""
    df = load_raw_data()

    if "artist_name" not in df.columns:
        raise ValueError("artist_id column required for group-aware splitting")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["artist_name"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split complete | train_rows={len(train_df)} | test_rows={len(test_df)} | "
        f"train_artists={train_df['artist_name'].nunique()} | test_artists={test_df['artist_name'].nunique()}"
    )

    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Run feature building for train and test and return X/y/artifacts."""
    X_train, y_train, artifacts = build_features(train_df, training=True)
    X_test, y_test, _ = build_features(test_df, training=False, artifacts=artifacts)

    logger.info(f"Train positive rate: {y_train.mean():.3f}")
    logger.info(f"Test positive rate:  {y_test.mean():.3f}")

    return X_train, y_train, X_test, y_test, artifacts


def run_abc_experiments(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, artifacts: dict) -> Dict[str, float]:
    """Run A/B/C experiments (audio / audio+norm / full) and return scores."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    def make_pipeline():
        return Pipeline(steps=[("scaler", StandardScaler()), ("model", LogisticRegression(solver="lbfgs", max_iter=1000))])

    def run_experiment(name: str, cols: List[str]):
        # Ensure we only use columns available in this dataset
        available_cols = [c for c in cols if c in X_train.columns]
        logger.info(f"Running experiment: {name} | requested={len(cols)} | available={len(available_cols)}")
        if len(available_cols) == 0:
            logger.info(f"Skipping experiment {name}; no matching columns available in data")
            return float("nan")

        clf = make_pipeline()
        clf.fit(X_train[available_cols], y_train)
        probs = clf.predict_proba(X_test[available_cols])[:, 1]
        # Report Lift@5% as the primary experiment metric (do not report ROC-AUC)
        try:
            from src.evaluate import compute_metrics_at_ks
            metrics_df = compute_metrics_at_ks(y_test.values, probs, ks=[0.05])
            lift_5 = metrics_df.loc[0.05, "lift"] if 0.05 in metrics_df.index else float("nan")
            logger.info(f"{name} Lift@5%: {lift_5:.2f}x")
        except Exception:
            lift_5 = float("nan")
            logger.info(f"{name} — could not compute Lift@5%")
        return float(lift_5)

    audio_only_cols = AUDIO_FEATURES + ["key_sin", "key_cos", "mode", "time_signature_num"]
    audio_plus_norm_cols = audio_only_cols + [f"{c}_z_genre" for c in NORMALIZE_FEATURES]
    full_cols = artifacts["feature_cols"]

    scores = {
        "A": run_experiment("Audio-only (A)", audio_only_cols),
        "B": run_experiment("Audio + genre z (B)", audio_plus_norm_cols),
        "C": run_experiment("Full model (C)", full_cols),
    }

    logger.info(f"Experiment summary (Lift@5%) | A={scores['A']:.2f}x | B={scores['B']:.2f}x | C={scores['C']:.2f}x")

    return scores


def get_cols_for_experiment(experiment: str, artifacts: dict) -> List[str]:
    """Return feature columns for a named experiment: 'audio', 'genre', or 'full'."""
    audio_only_cols = AUDIO_FEATURES + ["key_sin", "key_cos", "mode", "time_signature_num"]
    audio_plus_norm_cols = audio_only_cols + [f"{c}_z_genre" for c in NORMALIZE_FEATURES]

    experiment = experiment.lower()
    if experiment == "audio":
        return audio_only_cols
    elif experiment == "genre":
        return audio_plus_norm_cols
    elif experiment == "full":
        return artifacts["feature_cols"]
    else:
        raise ValueError("Unknown experiment. Choose from 'audio', 'genre', 'full'.")


def compute_univariate_aucs(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series):
    """Train a simple logistic on each feature and return AUCs."""
    from sklearn.linear_model import LogisticRegression

    aucs = {}
    for col in X_tr.columns:
        try:
            xtr = X_tr[[col]].values
            xte = X_te[[col]].values
            if np.unique(xtr).size <= 1:
                continue
            clf = LogisticRegression(solver="liblinear", max_iter=1000)
            clf.fit(xtr, y_tr)
            probs = clf.predict_proba(xte)[:, 1]
            score = roc_auc_score(y_te, probs)
            aucs[col] = score
        except Exception:
            continue
    return aucs


def persist_feature_diagnostics(
    feature_aucs: Dict[str, float],
    coef_pairs: Optional[List[Tuple[str, float]]],
    model_name: str,
    lgbm_importances: Optional[Dict[str, float]] = None,
):
    """Write feature diagnostics to CSV in LOG_DIR. Returns path."""
    coef_map = {f: c for f, c in (coef_pairs or [])}
    all_features = sorted(set(list(feature_aucs.keys()) + list(coef_map.keys())))

    fi_rows = []
    for feat in all_features:
        fi_rows.append({
            "feature": feat,
            "univariate_auc": feature_aucs.get(feat),
            "coefficient": coef_map.get(feat),
            "model": model_name,
            "lgbm_importance": (lgbm_importances.get(feat) if lgbm_importances else None),
        })

    fi_df = pd.DataFrame(fi_rows)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fi_path = LOG_DIR / "feature_importance.csv"

    # If file exists, append this model's diagnostics; otherwise create
    if fi_path.exists():
        existing = pd.read_csv(fi_path)
        # concat and drop duplicates by (model,feature) keeping new
        combined = pd.concat([existing, fi_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model", "feature"], keep="last")
        combined.to_csv(fi_path, index=False)
    else:
        fi_df.to_csv(fi_path, index=False)

    logger.info(f"Wrote feature importances to {fi_path}")
    return fi_path


def save_roc_pr_plots(y_test: pd.Series, y_pred_prob: np.ndarray, out_prefix: str = "roc"):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        roc_path = LOG_DIR / f"{out_prefix}_roc_curve.png"
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        pr_path = LOG_DIR / f"{out_prefix}_pr_curve.png"
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label='Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()

        logger.info(f"Saved ROC curve to {roc_path} and PR curve to {pr_path}")
        return roc_path, pr_path
    except Exception as exc:
        logger.info(f"Could not save ROC/PR plots: {exc}")
        return None, None