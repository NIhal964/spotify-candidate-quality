# src/train_lgbm.py

# src/train_lgbm.py

import logging
import joblib
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

# -------------------------------------------------
# Optional LightGBM dependency (GitHub-safe)
# -------------------------------------------------
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LGBM = False


def train_lgbm(experiment: str = "full", save_model: bool = True):
    """
    Train a non-linear model:
    - Use LightGBM if available
    - Fall back to RandomForest otherwise

    This design ensures portability across environments
    (local dev vs GitHub runners).
    """

    logger.info(
        "Starting non-linear training flow "
        "(LightGBM if available, otherwise RandomForest)"
    )

    # -------------------------
    # Load & split data
    # -------------------------
    train_df, test_df = load_and_split()

    X_train, y_train, X_test, y_test, artifacts = prepare_features(
        train_df, test_df
    )

    # -------------------------
    # Run A/B/C diagnostics
    # -------------------------
    _ = run_abc_experiments(
        X_train, X_test, y_train, y_test, artifacts
    )

    cols = get_cols_for_experiment(experiment, artifacts)
    logger.info(
        f"Training non-linear model on experiment='{experiment}' "
        f"with {len(cols)} features"
    )

    # ============================================================
    # Primary path: LightGBM
    # ============================================================
    if HAS_LGBM:
        logger.info("Training with LightGBM")

        clf = LGBMClassifier(
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train[cols], y_train)
        probs = clf.predict_proba(X_test[cols])[:, 1]

        # -------------------------
        # Multi-K Recall & Lift
        # -------------------------
        try:
            from src.evaluate import compute_metrics_at_ks, metrics_table_str
            from src.config import HEADLINE_K, SECONDARY_KS

            ks = [0.05, 0.10, 0.15, 0.20, 0.30]
            metrics_df = compute_metrics_at_ks(
                y_test.values, probs, ks
            )

            logger.info(
                "Recall & Lift at multiple K:\n"
                + metrics_table_str(metrics_df)
            )

            if HEADLINE_K in metrics_df.index:
                lift_p = metrics_df.loc[HEADLINE_K, "lift"]
                logger.info(
                    f"Headline metric | "
                    f"K={int(HEADLINE_K * 100)}% | "
                    f"Lift={lift_p:.2f}x"
                )

            for sk in SECONDARY_KS:
                if sk in metrics_df.index:
                    lift_s = metrics_df.loc[sk, "lift"]
                    logger.info(
                        f"Secondary metric | "
                        f"K={int(sk * 100)}% | "
                        f"Lift={lift_s:.2f}x"
                    )

        except Exception as e:
            logger.info(f"Could not compute multi-K metrics: {e}")

        # -------------------------
        # Feature diagnostics
        # -------------------------
        try:
            importances = dict(zip(cols, clf.feature_importances_))
        except Exception:
            importances = None

        persist_feature_diagnostics(
            compute_univariate_aucs(
                X_train, X_test, y_train, y_test
            ),
            None,
            model_name="lgbm",
            lgbm_importances=importances,
        )

        save_roc_pr_plots(
            y_test, probs, out_prefix="lgbm"
        )

        # -------------------------
        # Persist artifacts
        # -------------------------
        if save_model:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            joblib.dump(clf, MODEL_DIR / "model_lgbm.pkl")
            logger.info(
                f"Saved LightGBM model to {MODEL_DIR / 'model_lgbm.pkl'}"
            )

            try:
                joblib.dump(
                    artifacts, MODEL_DIR / "artifacts.pkl"
                )
                logger.info(
                    f"Saved artifacts to {MODEL_DIR / 'artifacts.pkl'}"
                )
            except Exception as exc:
                logger.info(
                    f"Could not save artifacts: {exc}"
                )
        else:
            logger.info(
                "save_model=False; skipping model persistence"
            )

    # ============================================================
    # Fallback path: RandomForest
    # ============================================================
    else:
        logger.info(
            "LightGBM not installed; "
            "falling back to RandomForest"
        )

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train[cols], y_train)
        probs = rf.predict_proba(X_test[cols])[:, 1]

        auc = roc_auc_score(y_test, probs)
        logger.info(
            f"RandomForest ROC-AUC (fallback): {auc:.4f}"
        )

        try:
            importances = dict(zip(cols, rf.feature_importances_))
        except Exception:
            importances = None

        persist_feature_diagnostics(
            compute_univariate_aucs(
                X_train, X_test, y_train, y_test
            ),
            None,
            model_name="random_forest",
            lgbm_importances=importances,
        )

        save_roc_pr_plots(
            y_test, probs, out_prefix="rf"
        )

        if save_model:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                rf, MODEL_DIR / "model_random_forest.pkl"
            )
            logger.info(
                f"Saved RandomForest model to "
                f"{MODEL_DIR / 'model_random_forest.pkl'}"
            )

            try:
                joblib.dump(
                    artifacts, MODEL_DIR / "artifacts.pkl"
                )
                logger.info(
                    f"Saved artifacts to {MODEL_DIR / 'artifacts.pkl'}"
                )
            except Exception as exc:
                logger.info(
                    f"Could not save artifacts: {exc}"
                )
        else:
            logger.info(
                "save_model=False; skipping model persistence"
            )


# -------------------------------------------------
# CLI entrypoint (module-safe)
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train non-linear models with fallback"
    )
    parser.add_argument(
        "--experiment",
        choices=["audio", "genre", "full"],
        default="full",
    )
    parser.add_argument(
        "--save-model",
        type=lambda x: x.lower()
        in ["1", "t", "true", "y", "yes"],
        default=True,
    )

    args = parser.parse_args()
    train_lgbm(
        experiment=args.experiment,
        save_model=args.save_model,
    )
