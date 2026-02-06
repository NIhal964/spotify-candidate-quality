# src/train_logistic.py

import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import MODEL_DIR
from src.train_common import (
    load_and_split,
    prepare_features,
    run_abc_experiments,
    compute_univariate_aucs,
    persist_feature_diagnostics,
    save_roc_pr_plots,
    get_cols_for_experiment,
)

logger = logging.getLogger(__name__)


def train_logistic(experiment: str = "full", save_model: bool = True):
    logger.info("Starting logistic training flow")
    train_df, test_df = load_and_split()
    X_train, y_train, X_test, y_test, artifacts = prepare_features(train_df, test_df)

    # Run A/B/C experiments (diagnostics)
    _ = run_abc_experiments(X_train, X_test, y_train, y_test, artifacts)

    # Determine feature set for final model based on experiment flag
    cols = get_cols_for_experiment(experiment, artifacts)
    logger.info(f"Training Logistic on experiment='{experiment}' with {len(cols)} features")

    # Final logistic pipeline
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_train[cols], y_train)

    # Evaluate
    y_pred_prob = pipeline.predict_proba(X_test[cols])[:, 1]
    # Compute multi-K Recall & Lift and log locked headline metric (do NOT log ROC-AUC)
    try:
        from src.evaluate import compute_metrics_at_ks, metrics_table_str
        from src.config import HEADLINE_K, SECONDARY_KS
        ks = [0.05, 0.10, 0.15, 0.20, 0.30]
        metrics_df = compute_metrics_at_ks(y_test.values, y_pred_prob, ks)
        logger.info("Recall & Lift at multiple K:\n" + metrics_table_str(metrics_df))

        primary_k = HEADLINE_K
        if primary_k in metrics_df.index:
            lift_p = metrics_df.loc[primary_k, "lift"]
            logger.info(f"Headline metric | K={int(primary_k*100)}% | Lift={lift_p:.2f}x")

        for sk in SECONDARY_KS:
            if sk in metrics_df.index:
                lift_s = metrics_df.loc[sk, "lift"]
                logger.info(f"Secondary metric | K={int(sk*100)}% | Lift={lift_s:.2f}x")
    except Exception as e:
        logger.info(f"Could not compute multi-K metrics: {e}")

    # Diagnostics
    feature_aucs = compute_univariate_aucs(X_train, X_test, y_train, y_test)

    try:
        model = pipeline.named_steps["model"]
        coefs = model.coef_.ravel()
        coef_pairs = list(zip(cols, coefs))
    except Exception:
        coef_pairs = None

    persist_feature_diagnostics(feature_aucs, coef_pairs, model_name="logistic")

    # Save plots and model
    save_roc_pr_plots(y_test, y_pred_prob, out_prefix="logistic")

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "model_logistic.pkl"
        joblib.dump(pipeline, model_path)
        # Persist training artifacts for reproducible inference/explain
        try:
            joblib.dump(artifacts, MODEL_DIR / "artifacts.pkl")
            logger.info(f"Saved artifacts to {MODEL_DIR / 'artifacts.pkl'}")
        except Exception as exc:
            logger.info(f"Could not save artifacts: {exc}")
        logger.info(f"Saved logistic model to {model_path}")
    else:
        logger.info("save_model=False; skipping model persistence")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["audio", "genre", "full"], default="full")
    p.add_argument("--save-model", type=lambda x: x.lower() in ["1", "t", "true", "y", "yes"], default=True)
    args = p.parse_args()
    train_logistic(experiment=args.experiment, save_model=args.save_model)