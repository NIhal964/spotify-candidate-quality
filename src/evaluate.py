
import numpy as np
import pandas as pd
from typing import Dict


def compute_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k_percent: float
) -> float:
    """
    Compute recall of positive class when selecting top K% highest-risk items.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (1 = high risk).
    y_score : array-like
        Predicted risk probabilities.
    top_k_percent : float
        Fraction of items to select (e.g., 0.3 for top 30%).

    Returns
    -------
    float
        Recall at K% (np.nan if no positive labels present).
    """
    # Validation
    if not (0 < top_k_percent <= 1):
        raise ValueError("top_k_percent must be in (0, 1]")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    n = len(y_score)
    if n == 0:
        raise ValueError("Input arrays must be non-empty")

    # Ensure at least one selected when top_k_percent > 0
    from math import ceil

    n_select = max(1, int(ceil(n * top_k_percent)))

    idx_sorted = np.argsort(y_score)[::-1]
    selected_idx = idx_sorted[:n_select]

    total_positives = y_true.sum()
    if total_positives == 0:
        # No positives in the data; recall is undefined
        return float("nan")

    recall = y_true[selected_idx].sum() / total_positives
    return float(recall)


def compute_lift_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k_percent: float
) -> float:
    """
    Compute lift over random selection at top K%.

    Lift = Recall_at_K / K
    Returns np.nan if recall is undefined.
    """
    recall_at_k = compute_recall_at_k(y_true, y_score, top_k_percent)
    if np.isnan(recall_at_k):
        return float("nan")
    lift = recall_at_k / top_k_percent
    return float(lift)


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray
) -> pd.DataFrame:
    """
    Compute recall and precision at different probability thresholds.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, recall, precision
    """
    rows = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append({
            "threshold": t,
            "recall": recall,
            "precision": precision
        })

    return pd.DataFrame(rows)


def compute_metrics_at_ks(y_true: np.ndarray, y_score: np.ndarray, ks=None) -> pd.DataFrame:
    """Compute recall and lift at multiple K values.

    Parameters
    ----------
    y_true, y_score : array-like
        Ground truth and predicted risk.
    ks : iterable of floats, optional
        List of K values (fractions) to compute metrics for. Defaults to [0.05, 0.10, 0.15, 0.20, 0.30].

    Returns
    -------
    pd.DataFrame
        Index: k (float fraction). Columns: recall, lift
    """
    if ks is None:
        ks = [0.05, 0.10, 0.15, 0.20, 0.30]

    rows = []
    for k in ks:
        recall = compute_recall_at_k(y_true, y_score, k)
        lift = compute_lift_at_k(y_true, y_score, k)
        rows.append({"k": float(k), "recall": float(recall) if not np.isnan(recall) else np.nan, "lift": float(lift) if not np.isnan(lift) else np.nan})

    df = pd.DataFrame(rows)
    df = df.set_index("k")
    return df


def metrics_table_str(df: pd.DataFrame) -> str:
    """Render multi-K metrics DataFrame as a pretty table string."""
    lines = []
    lines.append(" K% | Recall |  Lift ")
    lines.append("------------------------")
    for k, row in df.iterrows():
        k_pct = int(round(k * 100))
        recall = f"{row['recall']:.3f}" if not np.isnan(row['recall']) else "nan"
        lift = f"{row['lift']:.2f}x" if not np.isnan(row['lift']) else "nan"
        lines.append(f" {k_pct:>2}% | {recall:>6} | {lift:>6}")
    return "\n".join(lines)